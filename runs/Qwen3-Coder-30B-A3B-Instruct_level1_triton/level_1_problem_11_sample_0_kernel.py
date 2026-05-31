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
    l,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute the starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create the accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension (l dimension in original tensor)
    for k_start in range(0, l, BLOCK_SIZE_K):
        # Load tiles from A and B
        a_tile = tl.load(
            A_ptr + 
            pid_b * i * j * l +
            m_start * l + 
            k_start,
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < i) &
                  (k_start + tl.arange(0, BLOCK_SIZE_K)[None, :] < l),
            other=0.0
        )
        
        b_tile = tl.load(
            B_ptr + 
            k_start * k + 
            n_start,
            mask=(k_start + tl.arange(0, BLOCK_SIZE_K)[:, None] < l) &
                  (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < k),
            other=0.0
        )
        
        # Perform the matrix multiplication for this tile
        acc += tl.dot(a_tile, b_tile)
    
    # Write back the results
    out_ptr = Out_ptr + pid_b * i * j * k + m_start * k + n_start
    tl.store(
        out_ptr,
        acc,
        mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < i) &
              (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < k)
    )

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton implementation of einsum("bijl,lk->bijk")
    """
    assert A.dim() == 4 and B.dim() == 2
    assert A.shape[3] == B.shape[0]
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    out = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (k + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = b
    
    # Launch kernel
    einsum_bijl_lk_bijk_kernel[
        (grid_m, grid_n, grid_b),
        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    ](
        A, B, out, b, i, j, l, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)