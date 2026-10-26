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

@triton.jit
def b_grad_func(
    dc_ptr,
    a_ptr,
    b_grad_ptr,
    M,K,N,
    stride_dcm, stride_dcn,
    stride_ak, stride_an,
    stride_bk, stride_bn,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_K : tl.constexpr,
    GROUP_M : tl.constexpr
):
    '''Computes gradient for matrix B from upstream gradient matrix dC
    dB = A.T @ dC
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
    offset_k = pid_k * BLOCK_K + tl.arange(0,BLOCK_K)
    offset_n = tl.arange(0,BLOCK_N)

    dc_ptrs = dc_ptr + offset_m[:,None] * stride_dcm + offset_n[None,:] * stride_dcn
    a_ptrs = a_ptr + offset_k[:,None] * stride_ak + offset_n[None,:] * stride_an

    acc = tl.zeros((BLOCK_K,BLOCK_N), dtype = tl.float32)

    for m in range(0,M,BLOCK_M):
        m_mask = offset_m + m < M

        dc = tl.load(dc_ptrs, mask = m_mask[:,None] & (offset_n[None,:] < N), other = 0.0)
        a = tl.load(a_ptrs, mask = (offset_k[:,None] < K) & m_mask[None,:], other = 0.0)
        acc += tl.dot(a.T, dc)

        dc += BLOCK_M * stride_dcm
        a += BLOCK_M * stride_ak

    out = b_grad_ptr + offset_k[:,None] * stride_bk + offset_n[None,:] * stride_bn
    out_mask =  (offset_k[:,None] < K) & (offset_n[None,:] < N)
    tl.store(out, acc, mask = out_mask)

def solve_b_grad_func(dc : torch.Tensor, a : torch.Tensor, b_grad : torch.Tensor):
    assert dc.ndim == 2 and a.ndim == 2 and b_grad.ndim == 2, "All matrices must have dim = 2"

    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_N = 32
    GROUP_M = 8

    M,K = a.shape
    _,N = b_grad.shape
    grid = (triton.cdiv(M,BLOCK_M) *  triton.cdiv(N,BLOCK_N),)

    b_grad_func[grid](
        dc_ptr = dc,
        a_ptr = a,
        b_grad_ptr = b_grad,
        M = M, K = K, N = N,
        stride_dcm = dc.stride(0),
        stride_dcn = dc.stride(1),
        stride_ak = a.stride(0),
        stride_an = a.stride(1),
        stride_bk = b_grad.stride(0),
        stride_bn = b_grad.stride(1),
        BLOCK_M = BLOCK_M,
        BLOCK_K = BLOCK_K,
        BLOCK_N = BLOCK_N,
        GROUP_M = GROUP_M
    )

@triton.jit
def bwd_aat_kernel(
    dc_ptr,
    a_ptr,
    da_ptr,
    M, K,
    stride_dcm, stride_dcn,  
    stride_am,  stride_ak,   
    stride_dam, stride_dak,  
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,  
    GROUP_M: tl.constexpr,
):
    """
    Computes dA = (dC + dC.T) @ A
    Output dA has shape (M, K)
    Reduction dimension is M
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = num_pid_k * GROUP_M

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)

    m_mask = offs_m < M
    k_mask = offs_k < K

    # dc_ptrs loads dC[m, r] -> shape (BLOCK_M, BLOCK_R)
    dc_ptrs = dc_ptr + (offs_m[:, None] * stride_dcm + offs_r[None, :] * stride_dcn)
    # dc_t_ptrs loads dC[r, m] -> shape (BLOCK_R, BLOCK_M) 
    dc_t_ptrs = dc_ptr + (offs_r[:, None] * stride_dcm + offs_m[None, :] * stride_dcn)
    # Pointers for A: A[r, k] -> shape (BLOCK_R, BLOCK_K)
    a_ptrs = a_ptr + (offs_r[:, None] * stride_am + offs_k[None, :] * stride_ak)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for r in range(0, M, BLOCK_R):
        r_mask = (r + offs_r) < M
        dc = tl.load(dc_ptrs, mask=m_mask[:, None] & r_mask[None, :], other=0.0)
        dc_t_raw = tl.load(dc_t_ptrs, mask=r_mask[:, None] & m_mask[None, :], other=0.0)
        dc_t = tl.trans(dc_t_raw)

        s = dc + dc_t
        a = tl.load(a_ptrs, mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(s, a)

        dc_ptrs += BLOCK_R * stride_dcn
        dc_t_ptrs += BLOCK_R * stride_dcm
        a_ptrs += BLOCK_R * stride_am

    da = acc.to(a_ptr.dtype.element_ty)
    out_ptrs = da_ptr + (offs_m[:, None] * stride_dam + offs_k[None, :] * stride_dak)
    tl.store(out_ptrs, da, mask=m_mask[:, None] & k_mask[None, :])


def bwd_aat(dc: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    assert dc.is_cuda and a.is_cuda
    M, K = a.shape
    assert dc.shape == (M, M), f"Expected dC of shape ({M}, {M}), got {dc.shape}"

    da = torch.empty_like(a)

    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_R = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K),)

    bwd_aat_kernel[grid](
        dc_ptr=dc,
        a_ptr=a,
        da_ptr=da,
        M=M, K=K,
        stride_dcm=dc.stride(0), stride_dcn=dc.stride(1),
        stride_am=a.stride(0),   stride_ak=a.stride(1),
        stride_dam=da.stride(0), stride_dak=da.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
        GROUP_M=GROUP_M,
    )
    return da