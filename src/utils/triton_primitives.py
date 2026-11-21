import torch
import triton
import triton.language as tl


@triton.jit
def _sgd_kernel(
    param_ptr,
    grad_ptr,
    buffer_ptr,
    lr,
    momentum,
    weight_decay,
    nesterov: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad = grad + weight_decay * param

    if momentum:
        buffer = tl.load(buffer_ptr + offsets, mask=mask, other=0.0)
        buffer = momentum * buffer + grad
        tl.store(buffer_ptr + offsets, buffer, mask=mask)
        if nesterov:
            grad = grad + momentum * buffer
        else:
            grad = buffer

    tl.store(param_ptr + offsets, param - lr * grad, mask=mask)


def sgd_kernel(
    parameter,
    grad,
    buffer,
    lr,
    momentum,
    weight_decay,
    nesterov,
):
    n_elements = parameter.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _sgd_kernel[grid](
        parameter,
        grad,
        buffer if buffer is not None else parameter,
        lr,
        momentum,
        weight_decay,
        nesterov=nesterov,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    bias_correction1,
    bias_correction2,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    adamw: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    if adamw:
        param = param * (1.0 - lr * weight_decay)
    else:
        grad = grad + weight_decay * param

    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

    step_size = lr / bias_correction1
    denom = tl.sqrt(exp_avg_sq) / tl.sqrt(bias_correction2) + eps
    tl.store(param_ptr + offsets, param - step_size * exp_avg / denom, mask=mask)


def adam_kernel(
    parameter,
    grad,
    exp_avg,
    exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    adamw,
):
    n_elements = parameter.numel()
    if n_elements == 0:
        return
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _adam_kernel[grid](
        parameter,
        grad,
        exp_avg,
        exp_avg_sq,
        bias_correction1,
        bias_correction2,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        adamw=adamw,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _mse_forward_kernel(diff_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    diff = tl.load(diff_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, diff * diff, mask=mask)


@triton.jit
def _mse_backward_kernel(diff_ptr, grad_output_ptr, grad_ptr, n_elements, reduction: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    diff = tl.load(diff_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr)
    grad = 2.0 * diff
    if reduction == "mean":
        grad = grad * grad_output / n_elements
    elif reduction == "sum":
        grad = grad * grad_output
    else:
        grad = grad * tl.load(grad_output_ptr + offsets, mask=mask)
    tl.store(grad_ptr + offsets, grad, mask=mask)


class _MSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, reduction):
        diff = input - target
        loss = torch.empty_like(diff)
        n_elements = diff.numel()
        if n_elements:
            block_size = 1024
            _mse_forward_kernel[(triton.cdiv(n_elements, block_size),)](
                diff,
                loss,
                n_elements,
                BLOCK_SIZE=block_size,
            )
        ctx.save_for_backward(diff)
        ctx.reduction = reduction
        ctx.numel = diff.numel()
        if reduction == "none":
            return loss
        return loss.sum() if reduction == "sum" else loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        (diff,) = ctx.saved_tensors
        grad = torch.empty_like(diff)
        if diff.numel():
            block_size = 1024
            grad_output_tensor = grad_output.contiguous()
            _mse_backward_kernel[(triton.cdiv(diff.numel(), block_size),)](
                diff,
                grad_output_tensor,
                grad,
                diff.numel(),
                reduction=ctx.reduction,
                BLOCK_SIZE=block_size,
            )
        return grad, None, None


def mse_loss(input, target, reduction="mean"):
    loss = _MSEFunction.apply(input, target, reduction)
    return loss


@triton.jit
def _ce_forward_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    rows,
    classes,
    logits_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    target = tl.load(target_ptr + row)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < classes
    logits = tl.load(logits_ptr + row * logits_stride + offsets, mask=mask, other=-float("inf"))
    max_logit = tl.max(logits, axis=0)
    log_sum_exp = max_logit + tl.log(tl.sum(tl.exp(logits - max_logit), axis=0))
    target_logit = tl.load(
        logits_ptr + row * logits_stride + target,
        mask=target >= 0,
        other=0.0,
    )
    loss = log_sum_exp - target_logit
    tl.store(loss_ptr + row, loss, mask=target >= 0)


@triton.jit
def _ce_backward_kernel(
    logits_ptr,
    target_ptr,
    grad_output_ptr,
    grad_logits_ptr,
    rows,
    classes,
    logits_stride,
    reduction: tl.constexpr,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    grad_scale = tl.load(grad_output_ptr)
    target = tl.load(target_ptr + row)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < classes
    logits = tl.load(logits_ptr + row * logits_stride + offsets, mask=mask, other=-float("inf"))
    max_logit = tl.max(logits, axis=0)
    probs = tl.exp(logits - max_logit)
    probs = probs / tl.sum(probs, axis=0)
    grad = probs
    grad = tl.where(offsets == target, grad - 1.0, grad)
    if reduction == "mean":
        grad_scale = grad_scale / rows
    elif reduction == "none":
        grad_scale = tl.load(grad_output_ptr + row)
    tl.store(
        grad_logits_ptr + row * logits_stride + offsets,
        tl.where(target == ignore_index, 0.0, grad_scale * grad),
        mask=mask,
    )


class _CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, weight, ignore_index, reduction):
        original_shape = logits.shape
        target_shape = target.shape
        flat_logits = logits.reshape(-1, original_shape[-1]) if logits.ndim > 2 else logits
        rows, classes = flat_logits.shape
        block_size = triton.next_power_of_2(classes)
        losses = torch.empty(rows, device=logits.device, dtype=logits.dtype)
        flat_target = target.reshape(-1)
        _ce_forward_kernel[(rows,)](
            flat_logits,
            flat_target,
            losses,
            rows,
            classes,
            flat_logits.stride(0),
            BLOCK_SIZE=block_size,
        )
        valid = flat_target != ignore_index
        ctx.save_for_backward(flat_logits, flat_target)
        ctx.rows = rows
        ctx.classes = classes
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.valid_rows = max(1, int(valid.sum().item()))
        ctx.original_shape = original_shape
        ctx.target_shape = target_shape
        if reduction == "none":
            return losses.reshape(target.shape)
        if reduction == "sum":
            return losses[valid].sum()
        return losses[valid].mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        block_size = triton.next_power_of_2(ctx.classes)
        if ctx.reduction == "none":
            scale = grad_output.reshape(-1).contiguous()
        else:
            scale = grad_output.reshape(1)
        _ce_backward_kernel[(ctx.rows,)](
            logits,
            target,
            scale,
            grad_logits,
            ctx.valid_rows,
            ctx.classes,
            logits.stride(0),
            reduction=ctx.reduction,
            ignore_index=ctx.ignore_index,
            BLOCK_SIZE=block_size,
        )
        grad_logits = grad_logits.reshape(ctx.rows, ctx.classes)
        grad_logits = grad_logits.reshape(ctx.original_shape)
        return grad_logits, None, None, None, None


def cross_entropy_loss(logits, target, weight, ignore_index, reduction):
    if weight is not None:
        raise NotImplementedError("Triton CrossEntropyLoss weight is not implemented")
    return _CrossEntropyFunction.apply(logits, target, weight, ignore_index, reduction)


@triton.jit
def _bce_forward_kernel(input_ptr, target_ptr, loss_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    logits = tl.load(input_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    loss = tl.maximum(logits, 0.0) - logits * target + tl.log(1.0 + tl.exp(-tl.abs(logits)))
    tl.store(loss_ptr + offsets, loss, mask=mask)


@triton.jit
def _bce_backward_kernel(input_ptr, target_ptr, grad_output_ptr, grad_input_ptr, n_elements, reduction: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    logits = tl.load(input_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-logits))
    grad = sigmoid - target
    scale = tl.load(grad_output_ptr)
    if reduction == "mean":
        grad = grad * scale / n_elements
    else:
        grad = grad * scale
    tl.store(grad_input_ptr + offsets, grad, mask=mask)


class _BCEWithLogitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, weight, reduction):
        if weight is not None:
            raise NotImplementedError("Triton BCEWithLogitsLoss weight is not implemented")
        loss = torch.empty_like(logits)
        n_elements = logits.numel()
        block_size = 1024
        _bce_forward_kernel[(triton.cdiv(n_elements, block_size),)](
            logits,
            target,
            loss,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(logits, target)
        ctx.reduction = reduction
        ctx.n_elements = n_elements
        if reduction == "none":
            return loss
        return loss.sum() if reduction == "sum" else loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target = ctx.saved_tensors
        grad = torch.empty_like(logits)
        block_size = 1024
        scale = grad_output.reshape(1)
        _bce_backward_kernel[(triton.cdiv(ctx.n_elements, block_size),)](
            logits,
            target,
            scale,
            grad,
            ctx.n_elements,
            reduction=ctx.reduction,
            BLOCK_SIZE=block_size,
        )
        return grad, None, None, None


def bce_with_logits_loss(logits, target, weight, reduction):
    return _BCEWithLogitsFunction.apply(logits, target, weight, reduction)
