import torch
import torch.nn as nn
from matmul_triton import (
    solve_gram_kernel_2d,
    solve_standard_kernel_2d,
    solve_tall_matmul,
)


def test_tolerance_solve_gram_kernel_2d():
    M, K = 128, 64
    a = torch.randn(M, K, device="cuda")
    c = torch.empty(M, M, device="cuda")

    solve_gram_kernel_2d(a, c)

    expected_c = a @ a.T
    assert torch.allclose(c, expected_c, atol=1e-2, rtol = 1e-2), "solve_gram_kernel_2d failed"

def test_tolerance_solve_standard_kernel_2d():
    M, K, N = 128, 64, 64
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda")

    solve_standard_kernel_2d(a, b, c)

    expected_c = a @ b
    assert torch.allclose(c, expected_c, atol=1e-2, rtol=1e-2), "solve_standard_kernel_2d failed"

def test_tolerance_solve_tall_matmul():
    M, K, N = 8, 128, 16
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.zeros(M, N, device="cuda")

    solve_tall_matmul(a, b, c)

    expected_c = a @ b
    assert torch.allclose(c, expected_c, atol=1e-2, rtol=1e-2), "solve_tall_matmul failed"

def benchmark_cuda(fn, warmup=50, iters=200):
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

def run_benchmark(name, custom_fn, torch_fn, m, k, n):
    t_custom = benchmark_cuda(custom_fn)
    t_torch = benchmark_cuda(torch_fn)
    speedup = t_torch / t_custom

    flops = 2.0 * m * n * k
    tflops_custom = (flops / (t_custom * 1e-3)) / 1e12
    tflops_torch = (flops / (t_torch * 1e-3)) / 1e12

    print(f"=== {name} (M={m}, K={k}, N={n}) ===")
    print(f"  Custom Kernel : {t_custom * 1e3:7.2f} µs | {tflops_custom:6.2f} TFLOP/s")
    print(f"  torch.matmul  : {t_torch * 1e3:7.2f} µs | {tflops_torch:6.2f} TFLOP/s")
    print(f"  Speedup       : {speedup:7.2f}x {'faster' if speedup >= 1.0 else 'slower'} vs torch\n")


def bench_gram_kernel_2d():
    M, K = 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype = torch.bfloat16)
    c = torch.empty(M, M, device="cuda")

    run_benchmark(
        "Gram Kernel 2D",
        lambda: solve_gram_kernel_2d(a, c),
        lambda: torch.matmul(a, a.T),
        M, K, M
    )

def bench_standard_kernel_2d():
    M, K, N = 1024, 512, 1024
    a = torch.randn(M, K, device="cuda", dtype = torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype = torch.bfloat16)
    c = torch.empty(M, N, device="cuda")

    run_benchmark(
        "Standard Kernel 2D",
        lambda: solve_standard_kernel_2d(a, b, c),
        lambda: torch.matmul(a, b),
        M, K, N
    )

def bench_tall_matmul():
    M, K, N = 128, 4096,128
    a = torch.randn(M, K, device="cuda", dtype = torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype = torch.bfloat16)
    c = torch.empty(M, N, device="cuda")

    run_benchmark(
        "Tall Matmul",
        lambda: solve_tall_matmul(a, b, c),
        lambda: torch.matmul(a, b),
        M, K, N
    )


if __name__ == "__main__":
    test_tolerance_solve_gram_kernel_2d()
    test_tolerance_solve_standard_kernel_2d()
    test_tolerance_solve_tall_matmul()
    print("All numerical tolerance tests passed!")

    bench_gram_kernel_2d()
    bench_standard_kernel_2d()
    bench_tall_matmul()