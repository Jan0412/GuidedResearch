import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    b, i, j, k, l,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute the starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create pointers for the blocks of A and B
    offs_am = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Ensure we don't go out of bounds
    mask_m = offs_am < i
    mask_n = offs_bn < j
    mask_k = offs_k < l
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_iter in range(0, l, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = A_ptr + pid_b * stride_ab + \
                 offs_am[:, None] * stride_ai + \
                 offs_k[None, :] * stride_al
        a_mask = (offs_am[:, None] < i) & (offs_k[None, :] < l)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block
        b_ptrs = B_ptr + offs_k[:, None] * stride_bl + \
                 offs_bn[None, :] * stride_bk
        b_mask = (offs_k[:, None] < l) & (offs_bn[None, :] < k)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute dot product
        acc += tl.dot(a, b)
    
    # Write back the result
    c_ptrs = C_ptr + pid_b * stride_cb + \
             offs_am[:, None] * stride_ci + \
             offs_bn[None, :] * stride_cj
    c_mask = (offs_am[:, None] < i) & (offs_bn[None, :] < j)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Optimized einsum operation: "bijl,lk->bijk"
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be FP32"
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Create output tensor
    C = torch.empty(b, i, j, k, device=A.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (j + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = b
    
    # Launch kernel
    grid = (grid_m, grid_n, grid_b)
    
    einsum_bijl_lk_bijk_kernel[grid](
        A,
        B,
        C,
        b, i, j, k, l,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)