import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    b_size,
    i_size,
    j_size,
    l_size,
    k_size,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    k_idx = tl.program_id(3)
    
    # Calculate base offsets for this program
    batch_offset = batch_idx * i_size * j_size * l_size
    i_offset = i_idx * j_size * l_size
    j_offset = j_idx * l_size
    k_offset = k_idx * 1
    
    # Shared memory for B
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_L, BLOCK_SIZE_K))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over l dimension
    for l_start in range(0, l_size, BLOCK_SIZE_L):
        # Load B tile
        l_block = l_start + tl.arange(0, BLOCK_SIZE_L)
        k_block = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Bounds checking for B
        b_mask = (l_block[:, None] < l_size) & (k_block[None, :] < k_size)
        B_tile = tl.load(B_ptr + l_block[:, None] * k_size + k_block[None, :], mask=b_mask, other=0.0)
        tl.store(B_shared, B_tile)
        
        # Load A slice
        a_block = l_start + tl.arange(0, BLOCK_SIZE_L)
        a_mask = a_block < l_size
        A_slice = tl.load(A_ptr + batch_offset + i_offset + j_offset + a_block, mask=a_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(A_slice[:, None] * B_shared[:BLOCK_SIZE_L, :BLOCK_SIZE_K], axis=0)
    
    # Write result
    c_offset = batch_idx * i_size * j_size * k_size + i_idx * j_size * k_size + j_idx * k_size + k_idx
    tl.store(C_ptr + c_offset, acc)

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton implementation of einsum("bijl,lk->bijk")
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be FP32"
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, "Dimension mismatch between A and B"
    
    # Prepare output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid = (
        b,      # batch
        i,      # i
        j,      # j  
        k       # k
    )
    
    # Launch kernel
    einsum_bijl_lk_bijk_kernel[grid](
        A,
        B,
        C,
        b,
        i,
        j,
        l,
        k,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)