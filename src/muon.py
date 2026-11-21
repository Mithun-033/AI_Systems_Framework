"""
Triton kernels are used for newton-shultz iterations.

The update is split into three kernels so the intermediate matrix products can
be tiled and reused across the transformation:

1. ``ns_kernel_1`` computes the square Gram matrix
    ``A = X @ X.T``. Each program owns a ``BLOCK_M x BLOCK_M`` tile and walks
    the shared dimension in ``BLOCK_K`` chunks. Only the lower-triangular tiles
    are calculated; off-diagonal tiles are copied to the opposite side with a
    transpose because the result is symmetric.

2. ``ns_kernel_2`` computes the next square transform. For each output tile it
    calculates ``A @ A.T`` in the same tiled way, then combines it with the
    corresponding tile of ``A``:
    ``out_2 = b * A + c * (A @ A.T)``. It uses the same lower-triangle and
    mirrored-write optimization as the first kernel.

3. ``ns_kernel_3`` produces the rectangular result. It multiplies the square
    matrix from kernel 2 by ``X`` and adds a scaled copy of ``X``:
    ``out = a * X + out_2 @ X``. Because this result is not symmetric, every
    ``BLOCK_M x BLOCK_N`` tile is computed once.
"""

import math
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn  # noqa: PLR0402
import triton
import triton.language as tl


@triton.jit
def ns_kernel_1(
    a_ptr,
    out_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute X @ X.transpose (symmetric, upper triangle skipped + mirrored)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(M, BLOCK_M)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_n > pid_m:
        return

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = a_ptr + offset_k[:, None] * stride_ak + offset_n[None, :] * stride_am
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = offset_k + k < K

        a = tl.load(a_ptrs, mask=(offset_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offset_n[None, :] < M), other=0.0)
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_ak

    out_ptrs = out_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < M)
    tl.store(out_ptrs, acc, mask=mask)

    if pid_m != pid_n:
        acc_t = tl.trans(acc)
        out_ptrs_t = out_ptr + offset_n[:, None] * stride_cm + offset_m[None, :] * stride_cn
        mask_t = (offset_n[:, None] < M) & (offset_m[None, :] < M)
        tl.store(out_ptrs_t, acc_t, mask=mask_t)

def solve_ns_kernel_1(matrix: torch.Tensor, out: torch.Tensor):
    if matrix.ndim != 2:
        raise AssertionError("Muon only supports 2D tensors")
    M, K = matrix.shape

    BLOCK_M = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(M, BLOCK_M),)
    ns_kernel_1[grid](
        a_ptr=matrix,
        out_ptr=out,
        M=M,
        K=K,
        stride_am=matrix.stride(0),
        stride_ak=matrix.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )

@triton.jit
def ns_kernel_2(
    a_ptr,
    out_ptr,
    b_coeff : tl.constexpr,
    c_coeff : tl.constexpr,
    M,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    A = X @ X.Transpose
    A = (M,M)
    out = bA + c(A @ A.Transpose)  (symmetric, upper triangle skipped + mirrored)
    out = (M,M)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(M, BLOCK_M)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_n > pid_m:
        return

    offset_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    offset_n = BLOCK_M * pid_n + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    at_ptrs = a_ptr + offset_k[:, None] * stride_ak + offset_n[None, :] * stride_am

    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    for k in range(0, M, BLOCK_K):
        mask_k = k + offset_k < M
        a = tl.load(a_ptrs, mask=(offset_m[:, None] < M) & mask_k[None, :], other=0.0)
        at = tl.load(
            at_ptrs, mask=(mask_k[:, None]) & (offset_n[None, :] < M), other=0.0
        )

        acc = tl.dot(a, at, acc)

        a_ptrs += BLOCK_K * stride_ak
        at_ptrs += BLOCK_K * stride_ak

    A_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_n[None, :] * stride_ak
    A = tl.load(A_ptrs, mask=(offset_m[:, None] < M) & (offset_n[None, :] < M), other=0.0)
    out = b_coeff * A + c_coeff * acc

    out_ptrs = out_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    out_mask = (offset_m[:, None] < M) & (offset_n[None, :] < M)
    tl.store(out_ptrs, out, mask=out_mask)

    if pid_m != pid_n:
        out_t = tl.trans(out)
        out_ptrs_t = out_ptr + offset_n[:, None] * stride_cm + offset_m[None, :] * stride_cn
        mask_t = (offset_n[:, None] < M) & (offset_m[None, :] < M)
        tl.store(out_ptrs_t, out_t, mask=mask_t)


def solve_ns_kernel_2(matrix: torch.Tensor, b: float, c: float, out: torch.Tensor):
    M = matrix.shape[0]

    BLOCK_M = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(M, BLOCK_M),)
    ns_kernel_2[grid](
        a_ptr=matrix,
        out_ptr=out,
        b_coeff=b,
        c_coeff=c,
        M=M,
        stride_am=matrix.stride(0),
        stride_ak=matrix.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )


@triton.jit
def ns_kernel_3(
    a_ptr,
    b_ptr,
    out_ptr,
    a_coeff : tl.constexpr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    X = (M,N)
    Compute out = aX + out_2 @ X where,
    out_2 = bA + cA @ A.T, A = X @ X.T, out_2 = (M,M)
    out = (M,N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(GROUP_M, num_pid_m - first_pid_m)

    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, M, BLOCK_K):
        mask_k = offset_k + k < M

        a = tl.load(a_ptrs, mask=(offset_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None]) & (offset_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    x_ptrs = b_ptr + offset_m[:, None] * stride_bk + offset_n[None, :] * stride_bn
    mask_x = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    X = tl.load(x_ptrs, mask=mask_x, other=0.0)

    out = a_coeff * X + acc
    out_ptrs = out_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn

    tl.store(out_ptrs, out, mask=mask_x)


def solve_ns_kernel_3(
    X: torch.Tensor, out_2: torch.Tensor, a: float, out: torch.Tensor
):
    M, N = X.shape

    BLOCK_M = 64
    BLOCK_K = 32
    BLOCK_N = 64
    GROUP_M = 8

    grid = ((triton.cdiv(M, BLOCK_M)) * triton.cdiv(N, BLOCK_N),)

    ns_kernel_3[grid](
        a_ptr=out_2,
        b_ptr=X,
        out_ptr=out,
        a_coeff=a,
        M=M,
        N=N,
        stride_am=out_2.stride(0),
        stride_ak=out_2.stride(1),
        stride_bk=X.stride(0),
        stride_bn=X.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )


def newton_shultz_step(
    X: torch.Tensor, a: float, b: float, c: float, out: torch.Tensor
):
    '''
    Combines all three kernels to compute the Newton-Schultz step in one function.
    '''
    M, _ = X.shape
    A = torch.empty((M, M), device=X.device, dtype=X.dtype)
    out_2 = torch.empty((M, M), device=X.device, dtype=X.dtype)

    solve_ns_kernel_1(X, A)
    solve_ns_kernel_2(A, b, c, out_2)
    solve_ns_kernel_3(X, out_2, a, out)

def muon_step(
    weights: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float,
    a: float,
    b: float,
    c: float,
    lr: float,
    scaled_lr : float,
    weight_decay: float,
    steps: int,
    eps: float,
    nesterov : bool,
    out: torch.Tensor,
):
    '''
    Performs one iteration of muon optimizer step
    '''
    momentum.mul_(beta).add_(grad, alpha=1 - beta)

    if nesterov:
        momentum = grad + beta * (momentum - grad)  

    X = momentum.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T.contiguous()

    X = X / X.norm().clamp(min=eps)

    ns_out = torch.empty_like(X)
    ns_buf = torch.empty_like(X)
    newton_shultz_step(X, a, b, c, ns_out)
    for _ in range(steps - 1):
        newton_shultz_step(ns_out, a, b, c, ns_buf)
        ns_out, ns_buf = ns_buf, ns_out

    if transposed:
        ns_out = ns_out.T

    out.copy_(ns_out)

    weights.sub_(lr * weight_decay * weights)

    weights.sub_(scaled_lr * out)


class TritonMuon:
    def __init__(
        self,
        params: Iterable[nn.Parameter] | list[nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        coeffs: tuple[float, float, float] = (3.445, -4.775, 2.0315),
        nesterov : bool= True,
        beta: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn : Literal["original", "match_rms_adamw", "spectral_unclamped"] = "original"
    ):
        '''
        Triton implementation of Muon optimizer.
        '''

        if adjust_lr_fn not in ["original", "match_rms_adamw", "spectral_unclamped"]:
            raise AssertionError(f"Invalid adjust_lr_fn: {adjust_lr_fn}")
        
        self.model_params = list(params)
        self.a = coeffs[0]
        self.b = coeffs[1]
        self.c = coeffs[2]
        self.is_nesterov = nesterov
        self.beta = beta
        self.lr = lr
        self.adjust_lr_fn = adjust_lr_fn
        self.weight_decay = weight_decay
        self.steps = ns_steps
        self.eps = eps

        self.momentum = {}

        for param in self.model_params:
            self.momentum[param] = torch.zeros_like(
                param, device=param.device, dtype=param.dtype
            )

    @torch.no_grad()
    def step(self):
        '''
        Optimizer step for Muon optimizer.
        '''
        for param in self.model_params:
            if param.grad is None:
                continue
            out = torch.empty_like(param)

            if self.adjust_lr_fn == "original":
                scaled_lr = self.lr * math.sqrt(max(1, param.shape[0] / param.shape[1]))

            elif self.adjust_lr_fn == "match_rms_adamw":
                scaled_lr = 0.2 * self.lr * math.sqrt(max(param.shape[0], max(param.shape[1])))

            else:
                scaled_lr = self.lr * math.sqrt(param.shape[0] / param.shape[1])

            muon_step(
                weights=param.data,
                grad=param.grad,
                momentum=self.momentum[param],
                beta=self.beta,
                a=self.a,
                b=self.b,
                c=self.c,
                lr=self.lr,
                scaled_lr = scaled_lr,
                weight_decay=self.weight_decay,
                steps=self.steps,
                eps=self.eps,
                nesterov = self.is_nesterov,
                out=out,
            )

    def zero_grad(self):
        '''
        Resets gradients back to zero.
        '''
        for param in self.model_params:
            if param.grad is not None:
                param.grad.zero_()
    
        