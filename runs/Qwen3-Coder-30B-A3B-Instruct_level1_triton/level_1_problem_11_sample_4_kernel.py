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
    batch_id = tl.program_id(0)
    i_id = tl.program_id(1)
    j_id = tl.program_id(2)
    k_id = tl.program_id(3)
    
    # Calculate base offsets for this program
    batch_offset = batch_id * i_size * j_size * l_size
    i_offset = i_id * j_size * l_size
    j_offset = j_id * l_size
    k_offset = k_id * l_size
    
    # Shared memory for B
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_L, BLOCK_SIZE_K))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over l dimension in chunks
    for l_block_start in range(0, l_size, BLOCK_SIZE_L):
        # Load B chunk into shared memory
        b_offsets = l_block_start + tl.arange(0, BLOCK_SIZE_L)
        b_mask = b_offsets < l_size
        
        # Load B[l, k] values
        b_k_offsets = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_k_mask = b_k_offsets < k_size
        
        # Broadcast k dimension for B
        b_load_offsets = b_offsets[:, None] * k_size + b_k_offsets[None, :]
        b_load_mask = b_mask[:, None] & b_k_mask[None, :]
        
        # Load B values
        b_vals = tl.load(B_ptr + b_load_offsets, mask=b_load_mask, other=0.0)
        tl.store(B_shared, b_vals)
        
        # Load A values
        a_offsets = batch_offset + i_offset + j_offset + l_block_start + tl.arange(0, BLOCK_SIZE_L)
        a_mask = (l_block_start + tl.arange(0, BLOCK_SIZE_L)) < l_size
        a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result
    c_offset = batch_id * i_size * j_size * k_size + i_id * j_size * k_size + j_id * k_size + k_id
    tl.store(C_ptr + c_offset, acc[0])

@triton.jit
def einsum_bijl_lk_bijk_fused_kernel(
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
    batch_id = tl.program_id(0)
    i_id = tl.program_id(1)
    j_id = tl.program_id(2)
    k_id = tl.program_id(3)
    
    # Calculate base offsets for this program
    batch_offset = batch_id * i_size * j_size * l_size
    i_offset = i_id * j_size * l_size
    j_offset = j_id * l_size
    k_offset = k_id * l_size
    
    # Shared memory for B
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_L, BLOCK_SIZE_K))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over l dimension in chunks
    for l_block_start in range(0, l_size, BLOCK_SIZE_L):
        # Load B chunk into shared memory
        b_offsets = l_block_start + tl.arange(0, BLOCK_SIZE_L)
        b_mask = b_offsets < l_size
        
        # Load B[l, k] values
        b_k_offsets = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_k_mask = b_k_offsets < k_size
        
        # Broadcast k dimension for B
        b_load_offsets = b_offsets[:, None] * k_size + b_k_offsets[None, :]
        b_load_mask = b_mask[:, None] & b_k_mask[None, :]
        
        # Load B values
        b_vals = tl.load(B_ptr + b_load_offsets, mask=b_load_mask, other=0.0)
        tl.store(B_shared, b_vals)
        
        # Load A values
        a_offsets = batch_offset + i_offset + j_offset + l_block_start + tl.arange(0, BLOCK_SIZE_L)
        a_mask = (l_block_start + tl.arange(0, BLOCK_SIZE_L)) < l_size
        a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result
    c_offset = batch_id * i_size * j_size * k_size + i_id * j_size * k_size + j_id * k_size + k_id
    tl.store(C_ptr + c_offset, acc[0])

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Optimized implementation of torch.einsum("bijl,lk->bijk", A, B)
    using Triton kernels.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
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
        k       # k dimension
    )
    
    # Launch kernel
    einsum_bijl_lk_bijk_fused_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)