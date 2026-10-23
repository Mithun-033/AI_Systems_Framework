import torch
import triton
import triton.language as tl


@triton.jit
def a_grad_func(
    dc_ptr,
    b_ptr, 
    a_grad_ptr, 
    M,K,N,
    stride_dcm, stride_dcn,
    stride_bk, stride_bn,
    stride_am, stride_ak,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_K : tl.constexpr,
    GROUP_M : tl.constexpr
):
    '''Computes gradient for matrix A from upstream gradient matrix dC
    dA = dC @ B.T
    '''
    pid = tl.program_id(axis = 0)

    num_pid_k = tl.cdiv(N,BLOCK_K)
    num_pid_m = tl.cdiv(M,BLOCK_M)
    num_pid_in_group = num_pid_k * GROUP_M

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + pid % group_size_m
    pid_k = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
    offset_k = pid_k * BLOCK_N + tl.arange(0,BLOCK_K)
    offset_n = tl.arange(0,BLOCK_N)

    dc_ptrs = dc_ptr + offset_m[:,None] * stride_dcm + offset_n[None,:] * stride_dcn
    b_ptrs = b_ptr + offset_n[:,None] * stride_bn + offset_k[None,:] * stride_bk

    acc = tl.zeros((BLOCK_M,BLOCK_K), dtype = tl.float32)

    for n in range(0,N,BLOCK_N):
        n_mask = offset_n + n < N

        dc = tl.load(dc_ptrs, mask = (offset_m[:,None] < M) & n_mask[None,:], other = 0.0)
        b = tl.load(b_ptrs, mask = n_mask[:,None] & (offset_k[None,:] < K), other = 0.0)
        acc += tl.dot(dc,b)

        dc += BLOCK_N * stride_dcn
        b += BLOCK_N * stride_bn

    out = a_grad_ptr + offset_m[:,None] * stride_am + offset_k[None,:] * stride_ak
    out_mask =  (offset_m[:,None] < M) & (offset_k[None,:] < K)
    tl.store(out, acc, mask = out_mask)



def solve_a_grad_func(dc : torch.Tensor, b : torch.Tensor, a_grad : torch.Tensor):
    assert dc.ndim == 2 and b.ndim == 2 and a_grad.ndim == 2, "All matrices must have dim = 2"

    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_N = 32
    GROUP_M = 8

    M,K = a_grad.shape
    _,N = b.shape
    grid = (triton.cdiv(M,BLOCK_M) *  triton.cdiv(N,BLOCK_N),)

    a_grad_func[grid](
        dc_ptr = dc,
        b_ptr = b,
        a_grad_ptr = a_grad,
        M = M, K = K, N = N,
        stride_dcm = dc.stride(0),
        stride_dcn = dc.stride(1),
        stride_bk = b.stride(0),
        stride_bn = b.stride(1),
        stride_am = a_grad.stride(0),
        stride_ak = a_grad.stride(1),
        BLOCK_M = BLOCK_M,
        BLOCK_K = BLOCK_K,
        BLOCK_N = BLOCK_N,
        GROUP_M = GROUP_M
    )