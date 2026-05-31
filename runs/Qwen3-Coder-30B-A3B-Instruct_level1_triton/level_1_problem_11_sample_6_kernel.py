import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,
    B_ptr,
    Out_ptr,
    b, i, j, k, l,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_ob, stride_oi, stride_oj, stride_ok,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create masks for valid indices
    m_mask = m_start + tl.arange(0, BLOCK_SIZE_M) < i
    n_mask = n_start + tl.arange(0, BLOCK_SIZE_N) < j
    
    # Loop over K dimension
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, l, BLOCK_SIZE_K):
        # Load A fragment
        k_offset = k_start + tl.arange(0, BLOCK_SIZE_K)
        a_mask = k_offset < l
        
        # Broadcast m and n dimensions
        a_ptrs = A_ptr + pid_b * stride_ab + \
                 tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_ai + \
                 tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_aj + \
                 k_offset[None, :] * stride_al
        
        a = tl.load(a_ptrs, mask=(m_mask[:, None] & n_mask[None, :] & a_mask[None, :]), other=0.0)
        
        # Load B fragment
        b_ptrs = B_ptr + k_offset[:, None] * stride_bl + \
                 tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bk
        
        b = tl.load(b_ptrs, mask=(a_mask[:, None] & n_mask[None, :]), other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
    
    # Write back the result
    out_ptrs = Out_ptr + pid_b * stride_ob + \
               tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_oi + \
               tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_oj
    
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, accumulator, mask=out_mask)

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton implementation of torch.einsum("bijl,lk->bijk", A, B)
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Create output tensor
    out = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid_m = (i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (j + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = b
    
    # Launch kernel
    grid = (grid_m, grid_n, grid_b)
    
    # Get strides
    stride_ab, stride_ai, stride_aj, stride_al = A.stride()
    stride_bl, stride_bk = B.stride()
    stride_ob, stride_oi, stride_oj, stride_ok = out.stride()
    
    einsum_bijl_lk_bijk_kernel[grid](
        A, B, out,
        b, i, j, k, l,
        stride_ab, stride_ai, stride_aj, stride_al,
        stride_bl, stride_bk,
        stride_ob, stride_oi, stride_oj, stride_ok,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)