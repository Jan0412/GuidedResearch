import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,
    B_ptr,
    Out_ptr,
    b,
    i,
    j,
    k,
    l,
    stride_ab,
    stride_ai,
    stride_aj,
    stride_al,
    stride_bl,
    stride_bk,
    stride_ob,
    stride_oi,
    stride_oj,
    stride_ok,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute starting indices for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create pointers for A and B
    offs_am = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    mask_m = offs_am < i
    mask_n = offs_bn < j
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, l, BLOCK_SIZE_K):
        # Create pointers for current tile
        a_ptrs = A_ptr + (
            pid_b * stride_ab +
            offs_am[:, None] * stride_ai +
            tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_al
        )
        
        b_ptrs = B_ptr + (
            tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bl +
            offs_bn[None, :] * stride_bk
        )
        
        # Load tiles
        a_tile = tl.load(a_ptrs, mask=(mask_m[:, None] & (offs_k[None, :] < l - k_start)))
        b_tile = tl.load(b_ptrs, mask=((offs_k[:, None] < l - k_start) & mask_n[None, :]))
        
        # Accumulate
        acc += tl.dot(a_tile, b_tile)
    
    # Write result
    out_ptrs = Out_ptr + (
        pid_b * stride_ob +
        offs_am[:, None] * stride_oi +
        offs_bn[None, :] * stride_oj
    )
    
    tl.store(out_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton kernel for einsum operation: "bijl,lk->bijk"
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Both tensors must be FP32"
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    out = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(i, BLOCK_SIZE_M)
    grid_n = triton.cdiv(j, BLOCK_SIZE_N)
    grid_b = b
    
    grid = (grid_m, grid_n, grid_b)
    
    # Launch kernel
    einsum_bijl_lk_bijk_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        Out_ptr=out.data_ptr(),
        b=b,
        i=i,
        j=j,
        k=k,
        l=l,
        stride_ab=A.stride(0),
        stride_ai=A.stride(1),
        stride_aj=A.stride(2),
        stride_al=A.stride(3),
        stride_bl=B.stride(0),
        stride_bk=B.stride(1),
        stride_ob=out.stride(0),
        stride_oi=out.stride(1),
        stride_oj=out.stride(2),
        stride_ok=out.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)