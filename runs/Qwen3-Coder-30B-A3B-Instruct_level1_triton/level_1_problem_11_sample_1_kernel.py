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
    k_offset = k_idx * l_size
    
    # Shared memory for B transpose
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(l_size, BLOCK_SIZE_K))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over l dimension with reduction
    for l_block_start in range(0, l_size, BLOCK_SIZE_L):
        # Load A fragment
        a_offsets = batch_offset + i_offset + j_offset + l_block_start + tl.arange(0, BLOCK_SIZE_L)
        a_mask = (l_block_start + tl.arange(0, BLOCK_SIZE_L)) < l_size
        a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B fragment
        b_offsets = l_block_start + tl.arange(0, BLOCK_SIZE_L) + k_idx * l_size
        b_mask = (l_block_start + tl.arange(0, BLOCK_SIZE_L)) < l_size
        b_vals = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(a_vals * b_vals)
    
    # Store result
    if k_idx < k_size:
        c_offset = batch_idx * i_size * j_size * k_size + i_idx * j_size * k_size + j_idx * k_size + k_idx
        tl.store(C_ptr + c_offset, acc)

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton implementation of einsum("bijl,lk->bijk")
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.dim() == 4 and B.dim() == 2, "A must be 4D and B must be 2D"
    assert A.shape[3] == B.shape[0], "Last dimension of A must match first dimension of B"
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Prepare output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid = (
        b,      # batch dimension
        i,      # i dimension  
        j,      # j dimension
        min(k, 1024)  # k dimension (capped for practicality)
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