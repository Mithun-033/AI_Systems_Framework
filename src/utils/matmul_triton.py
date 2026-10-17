import torch
import triton
import triton.language as tl


@triton.jit
def standard_kernel_2d(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
): 
    '''Kernel for any standard 2D matrix, uses grouped-id matmul kernel for better L2 cache use'''
    pid = tl.program_id(axis = 0)
    num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M,BLOCK_SIZE_M)

    num_pid_in_group = num_pid_n  * GROUP_SIZE_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(GROUP_SIZE_M, num_pid_m - first_pid_m)

    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    offset_k = tl.arange(0,BLOCK_SIZE_K)

    a_ptrs = a_ptr + offset_m[:,None] * stride_am + offset_k[None,:] * stride_ak
    b_ptrs = b_ptr + offset_k[:,None] * stride_bk + offset_n[None,:] * stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)

    for k in range(0,K,BLOCK_SIZE_K):
        mask_k = offset_k + k < K
        
        a = tl.load(a_ptrs, mask = (offset_m[:,None] < M) & mask_k[None,:], other = 0.0)
        b = tl.load(b_ptrs, mask = mask_k[:,None] & (offset_n[None,:] < N), other = 0.0)
        acc = tl.dot(a,b,acc)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offset_m[:,None] * stride_cm + offset_n[None,:] * stride_cn
    mask_c = (offset_m[:,None] < M) & (offset_n[None,:] < N)
    tl.store(c_ptrs, acc, mask = mask_c)

def solve_standard_kernel_2d(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2 and c.ndim == 2, "All tensors must be 2D"
    assert a.is_cuda and b.is_cuda and c.is_cuda, "All tensors must be on CUDA"
    assert a.shape[1] == b.shape[0], (
        "Inner dimensions must match for matrix multiplication"
    )
    assert c.shape[0] == a.shape[0] and c.shape[1] == b.shape[1], (
        "Output tensor shape must match the result of matrix multiplication"
    )

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    M, K = a.shape
    K, N = b.shape

    grid = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)

    standard_kernel_2d[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        M=M,
        K=K,
        N=N,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

