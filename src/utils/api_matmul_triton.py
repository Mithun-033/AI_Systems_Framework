import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from utils.fwd_matmul_triton import solve_standard_kernel_2d, solve_gram_kernel_2d
    from utils.bwd_matmul_triton import solve_a_grad_func, solve_b_grad_func, bwd_aat
except ImportError:
    from fwd_matmul_triton import solve_standard_kernel_2d, solve_gram_kernel_2d
    from bwd_matmul_triton import solve_a_grad_func, solve_b_grad_func, bwd_aat


# Autograd Functions
class LinearMatmulFunction(torch.autograd.Function):
    """
    Autograd Function for standard linear matrix multiplication:
        C = A @ B + bias (optional)
    Backward passes:
        dA = dC @ B.T
        dB = A.T @ dC
        dbias = sum(dC, dim=0)
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert a.ndim == 2 and b.ndim == 2, "All tensors must be 2D"
        assert a.is_cuda and b.is_cuda, "All tensors must be on CUDA"
        assert a.shape[1] == b.shape[0], (
            f"Inner dimensions must match: a has shape {a.shape}, b has shape {b.shape}"
        )

        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        solve_standard_kernel_2d(a, b, c)

        if bias is not None:
            assert bias.is_cuda, "Bias must be on CUDA"
            assert bias.ndim in (1, 2), "Bias must be 1D or 2D"
            c = c + bias

        ctx.save_for_backward(a, b)
        ctx.has_bias = bias is not None
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        a, b = ctx.saved_tensors
        grad_a = grad_b = grad_bias = None

        grad_output = grad_output.contiguous()

        # dA = dC @ B.T
        if ctx.needs_input_grad[0]:
            grad_a = torch.empty_like(a)
            solve_a_grad_func(grad_output, b, grad_a)

        # dB = A.T @ dC
        if ctx.needs_input_grad[1]:
            grad_b = torch.empty_like(b)
            solve_b_grad_func(grad_output, a, grad_b)

        # dbias = sum(dC, dim=0)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_a, grad_b, grad_bias


class GramMatmulFunction(torch.autograd.Function):
    """
    Autograd Function for Gram matrix multiplication:
        C = A @ A.T
    Backward pass:
        dA = (dC + dC.T) @ A
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2, "Matrix must be 2D"
        assert a.is_cuda, "Tensor must be present in CUDA"

        M, _ = a.shape
        c = torch.empty((M, M), device=a.device, dtype=a.dtype)
        solve_gram_kernel_2d(a, c)

        ctx.save_for_backward(a)
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        (a,) = ctx.saved_tensors
        grad_a = None

        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            grad_a = bwd_aat(grad_output, a)

        return grad_a


def linear_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes A @ B (+ bias) using Triton autograd."""
    return LinearMatmulFunction.apply(a, b, bias)


def gram_matmul(a: torch.Tensor) -> torch.Tensor:
    """Computes A @ A.T using Triton autograd."""
    return GramMatmulFunction.apply(a)

class LinearMatmul(nn.Module):
    """
    Linear layer using custom Triton kernels for both forward and backward passes.
    Supports input tensors of shape (M, in_features) or higher-dimensional (*, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 2:
            return LinearMatmulFunction.apply(input, self.weight, self.bias)

        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        out_2d = LinearMatmulFunction.apply(input_2d, self.weight, self.bias)
        return out_2d.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# Alias for convenience
Linear = LinearMatmul

class GramMatmul(nn.Module):
    """
    Gram matrix multiplication module computing A @ A.T using Triton autograd.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return GramMatmulFunction.apply(a)


def test_tolerance_linear_matmul():
    M, K, N = 128, 64, 96
    a = torch.randn(M, K, device="cuda", requires_grad=True)
    b = torch.randn(K, N, device="cuda", requires_grad=True)
    bias = torch.randn(N, device="cuda", requires_grad=True)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    c = linear_matmul(a, b, bias)
    expected_c = a_ref @ b_ref + bias_ref

    fwd_atol = (c - expected_c).abs().max().item()
    fwd_rtol = ((c - expected_c).abs() / (expected_c.abs() + 1e-7)).max().item()
    assert torch.allclose(c, expected_c, atol=1e-2, rtol=1e-2), (
        f"linear_matmul forward failed: atol={fwd_atol:.4e}, rtol={fwd_rtol:.4e}"
    )

    grad_output = torch.randn_like(c)
    c.backward(grad_output)
    expected_c.backward(grad_output)

    grad_a_atol = (a.grad - a_ref.grad).abs().max().item()
    grad_a_rtol = ((a.grad - a_ref.grad).abs() / (a_ref.grad.abs() + 1e-7)).max().item()
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-2, rtol=1e-2), (
        f"linear_matmul grad_a failed: atol={grad_a_atol:.4e}, rtol={grad_a_rtol:.4e}"
    )

    grad_b_atol = (b.grad - b_ref.grad).abs().max().item()
    grad_b_rtol = ((b.grad - b_ref.grad).abs() / (b_ref.grad.abs() + 1e-7)).max().item()
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-2, rtol=1e-2), (
        f"linear_matmul grad_b failed: atol={grad_b_atol:.4e}, rtol={grad_b_rtol:.4e}"
    )

    grad_bias_atol = (bias.grad - bias_ref.grad).abs().max().item()
    assert torch.allclose(bias.grad, bias_ref.grad, atol=1e-2, rtol=1e-2), (
        f"linear_matmul grad_bias failed: atol={grad_bias_atol:.4e}"
    )

    print(
        f"Linear Matmul Autograd Tolerance  : Passed "
        f"(fwd_atol={fwd_atol:.2e}, grad_a_atol={grad_a_atol:.2e}, grad_b_atol={grad_b_atol:.2e}, grad_bias_atol={grad_bias_atol:.2e})"
    )


def test_tolerance_gram_matmul():
    M, K = 128, 64
    a = torch.randn(M, K, device="cuda", requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)

    c = gram_matmul(a)
    expected_c = a_ref @ a_ref.T

    fwd_atol = (c - expected_c).abs().max().item()
    fwd_rtol = ((c - expected_c).abs() / (expected_c.abs() + 1e-7)).max().item()
    assert torch.allclose(c, expected_c, atol=1e-1, rtol=1e-2), (
        f"gram_matmul forward failed: atol={fwd_atol:.4e}, rtol={fwd_rtol:.4e}"
    )

    grad_output = torch.randn_like(c)
    c.backward(grad_output)
    expected_c.backward(grad_output)

    grad_a_atol = (a.grad - a_ref.grad).abs().max().item()
    grad_a_rtol = ((a.grad - a_ref.grad).abs() / (a_ref.grad.abs() + 1e-7)).max().item()
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-1, rtol=1e-2), (
        f"gram_matmul backward failed: atol={grad_a_atol:.4e}, rtol={grad_a_rtol:.4e}"
    )

    print(
        f"Gram Matmul Autograd Tolerance    : Passed "
        f"(fwd_atol={fwd_atol:.2e}, grad_a_atol={grad_a_atol:.2e})"
    )


def test_tolerance_linear_module():
    M, in_features, out_features = 64, 128, 64
    custom_layer = LinearMatmul(in_features, out_features, bias=True, device=torch.device("cuda"))

    ref_layer = nn.Linear(in_features, out_features, bias=True, device="cuda")
    with torch.no_grad():
        ref_layer.weight.copy_(custom_layer.weight.T)
        ref_layer.bias.copy_(custom_layer.bias)

    x = torch.randn(M, in_features, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    out = custom_layer(x)
    out_ref = ref_layer(x_ref)

    fwd_atol = (out - out_ref).abs().max().item()
    assert torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2), (
        f"LinearMatmul module forward failed: atol={fwd_atol:.4e}"
    )

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    grad_x_atol = (x.grad - x_ref.grad).abs().max().item()
    grad_w_atol = (custom_layer.weight.grad - ref_layer.weight.grad.T).abs().max().item()
    grad_b_atol = (custom_layer.bias.grad - ref_layer.bias.grad).abs().max().item()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(custom_layer.weight.grad, ref_layer.weight.grad.T, atol=1e-2, rtol=1e-2)
    assert torch.allclose(custom_layer.bias.grad, ref_layer.bias.grad, atol=1e-2, rtol=1e-2)

    print(
        f"LinearMatmul Module Tolerance     : Passed "
        f"(fwd_atol={fwd_atol:.2e}, grad_x_atol={grad_x_atol:.2e}, grad_w_atol={grad_w_atol:.2e}, grad_b_atol={grad_b_atol:.2e})"
    )

def benchmark_cuda(fn, warmup=50, iters=200) -> float:
    """Accurately measures pure GPU kernel runtime in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def run_benchmark(name: str, custom_fn, torch_fn, m: int, k: int, n: int, is_fwd_bwd: bool = False):
    t_custom = benchmark_cuda(custom_fn)
    t_torch = benchmark_cuda(torch_fn)
    speedup = t_torch / t_custom

    # FLOPs for GEMM is 2 * M * N * K; for fwd+bwd it is 3x (2x for bwd)
    factor = 6.0 if is_fwd_bwd else 2.0
    flops = factor * m * n * k
    tflops_custom = (flops / (t_custom * 1e-3)) / 1e12
    tflops_torch = (flops / (t_torch * 1e-3)) / 1e12

    tag = " (Fwd + Bwd)" if is_fwd_bwd else " (Forward)"
    print(f"=== {name}{tag} (M={m}, K={k}, N={n}) ===")
    print(f"  Custom Autograd : {t_custom * 1e3:7.2f} µs | {tflops_custom:6.2f} TFLOP/s")
    print(f"  PyTorch Native  : {t_torch * 1e3:7.2f} µs | {tflops_torch:6.2f} TFLOP/s")
    print(f"  Speedup         : {speedup:7.2f}x {'faster' if speedup >= 1.0 else 'slower'} vs torch\n")


def bench_linear_matmul():
    M, K, N = 1024, 512, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward benchmark
    run_benchmark(
        "Linear Matmul",
        lambda: linear_matmul(a, b, bias),
        lambda: torch.matmul(a, b) + bias,
        M, K, N,
        is_fwd_bwd=False,
    )

    # Forward + Backward benchmark
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    def custom_fwd_bwd():
        c = linear_matmul(a, b, bias)
        c.backward(grad_out, retain_graph=True)

    def torch_fwd_bwd():
        c = torch.matmul(a, b) + bias
        c.backward(grad_out, retain_graph=True)

    run_benchmark(
        "Linear Matmul",
        custom_fwd_bwd,
        torch_fwd_bwd,
        M, K, N,
        is_fwd_bwd=True,
    )


def bench_gram_matmul():
    M, K = 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward benchmark
    run_benchmark(
        "Gram Matmul",
        lambda: gram_matmul(a),
        lambda: torch.matmul(a, a.T),
        M, K, M,
        is_fwd_bwd=False,
    )

    # Forward + Backward benchmark
    grad_out = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)

    def custom_fwd_bwd():
        c = gram_matmul(a)
        c.backward(grad_out, retain_graph=True)

    def torch_fwd_bwd():
        c = torch.matmul(a, a.T)
        c.backward(grad_out, retain_graph=True)

    run_benchmark(
        "Gram Matmul",
        custom_fwd_bwd,
        torch_fwd_bwd,
        M, K, M,
        is_fwd_bwd=True,
    )


def bench_linear_module():
    M, in_features, out_features = 1024, 512, 1024
    x = torch.randn(M, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    layer = LinearMatmul(in_features, out_features, bias=True, device=torch.device("cuda"), dtype=torch.bfloat16)
    ref_layer = nn.Linear(in_features, out_features, bias=True, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref_layer.weight.copy_(layer.weight.T)
        ref_layer.bias.copy_(layer.bias)

    grad_out = torch.randn(M, out_features, device="cuda", dtype=torch.bfloat16)

    def custom_fwd_bwd():
        out = layer(x)
        out.backward(grad_out, retain_graph=True)

    def ref_fwd_bwd():
        out = ref_layer(x)
        out.backward(grad_out, retain_graph=True)

    run_benchmark(
        "LinearMatmul Layer",
        custom_fwd_bwd,
        ref_fwd_bwd,
        M, in_features, out_features,
        is_fwd_bwd=True,
    )


if __name__ == "__main__":
    print("Running numerical tolerance tests (atol / rtol)...")
    test_tolerance_linear_matmul()
    test_tolerance_gram_matmul()
    test_tolerance_linear_module()
    print("All numerical tolerance tests passed!\n")

    print("Running performance benchmarks (speedups vs PyTorch)...")
    bench_linear_matmul()
    bench_gram_matmul()
    bench_linear_module()
