from collections.abc import Iterable
from typing import Literal

import torch
from torch.optim import Optimizer

try:
    from utils.triton_primitives import sgd_kernel, adam_kernel
except ImportError:
    from utils.triton_primitives import sgd_kernel, adam_kernel


def _parameters(params: Iterable[torch.Tensor] | Iterable[dict]) -> list:
    params = list(params)
    if not params:
        raise ValueError("optimizer requires at least one parameter")
    if not all(isinstance(param, (torch.Tensor, dict)) for param in params):
        raise TypeError("optimizer parameters must be tensors or parameter groups")
    if any(isinstance(param, torch.Tensor) and not param.requires_grad for param in params):
        raise ValueError("optimizer cannot include a parameter that does not require gradients")
    return params


class TritonSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(
            _parameters(params),
            dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            momentum = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if not parameter.is_cuda:
                    grad = grad.add(parameter, alpha=group["weight_decay"])
                    if momentum:
                        state = self.state.setdefault(parameter, {})
                        buffer = state.setdefault("momentum_buffer", torch.clone(grad).detach())
                        buffer.mul_(momentum).add_(grad)
                        grad = grad.add(buffer, alpha=momentum) if group["nesterov"] else buffer
                    parameter.add_(grad, alpha=-group["lr"])
                    continue

                state = self.state.setdefault(parameter, {})
                buffer = state.setdefault(
                    "momentum_buffer",
                    torch.zeros_like(parameter) if momentum else None,
                )
                sgd_kernel(
                    parameter,
                    grad,
                    buffer,
                    group["lr"],
                    momentum,
                    group["weight_decay"],
                    group["nesterov"],
                )
        return loss


class TritonAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        adamw: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        super().__init__(
            _parameters(params),
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, adamw=adamw),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state.setdefault(parameter, {})
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1

                if not parameter.is_cuda:
                    grad = parameter.grad
                    if group["adamw"]:
                        parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    else:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(group["eps"])
                    parameter.addcdiv_(exp_avg / bias_correction1, denom, value=-group["lr"])
                    continue

                adam_kernel(
                    parameter,
                    parameter.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    bool(group["adamw"]),
                )
        return loss


class TritonAdamW(TritonAdam):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, adamw=True)


def get_optimizer(name: str, params, **kwargs):
    optimizers = {"sgd": TritonSGD, "adam": TritonAdam, "adamw": TritonAdamW}
    key = name.lower().replace("-", "")
    if key not in optimizers:
        raise ValueError(f"unknown optimizer '{name}'. Available: {', '.join(optimizers)}")
    return optimizers[key](params, **kwargs)


__all__ = ["TritonSGD", "TritonAdam", "TritonAdamW", "get_optimizer"]
