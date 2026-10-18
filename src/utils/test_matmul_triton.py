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
    c = torch.empty(M, N, device="cuda")

    solve_tall_matmul(a, b, c)

    expected_c = a @ b
    assert torch.allclose(c, expected_c, atol=1e-2, rtol=1e-2), "solve_tall_matmul failed"

if __name__ == "__main__":
    test_tolerance_solve_gram_kernel_2d()
    test_tolerance_solve_standard_kernel_2d()
    test_tolerance_solve_tall_matmul()
    print("All tests passed!")