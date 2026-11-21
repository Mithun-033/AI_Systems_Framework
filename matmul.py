import torch
import triton
import triton.language as tl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@triton.jit
def mult_kernel(
    a, b, c,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    block_offsetx = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_offsety = pid_y * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_offsetk = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a + (block_offsetx[:, None] * stride_am + block_offsetk[None, :] * stride_ak)
    b_ptr = b + (block_offsetk[:, None] * stride_bk + block_offsety[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining = K - k * BLOCK_SIZE_K
        a_mask = (block_offsetx[:, None] < M) & (block_offsetk[None, :] < remaining)
        b_mask = (block_offsetk[:, None] < remaining) & (block_offsety[None, :] < N)
        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        a_ptr = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptr = b_ptr + BLOCK_SIZE_K * stride_bk

    #c-prt
    c_ptr = c + (block_offsetx[:, None] * stride_cm + block_offsety[None, :] * stride_cn)
    c_mask = (block_offsetx[:, None] < M) & (block_offsety[None, :] < N)
    tl.store(c_ptr, accumulator.to(c.dtype.element_ty), mask=c_mask)


def matmul(a, b, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
    assert a.is_cuda and b.is_cuda, "tensors must be on the GPU"
    assert a.shape[1] == b.shape[0], "inner dimensions must match"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    mult_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


if __name__ == "__main__":
    if device.type != "cuda":
        raise SystemExit("Triton kernels need a CUDA GPU; none was found.")
    torch.manual_seed(0)
    for M, K, N in [(128, 64, 128), (130, 70, 90), (257, 33, 65)]:
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_triton = matmul(a, b)
        out_torch = torch.matmul(a, b)

        ok = torch.allclose(out_triton, out_torch, atol=1e-2, rtol=1e-2)
        max_err = (out_triton - out_torch).abs().max().item()
        print(f"{M}x{K} @ {K}x{N}: match={ok}, max abs error={max_err:.5f}")