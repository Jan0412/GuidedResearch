import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def einsum_kernel(
    A_ptr,  # Pointer to 4D tensor A of shape (b, i, j, l)
    B_ptr,  # Pointer to matrix B of shape (l, k)
    C_ptr,  # Pointer to output tensor C of shape (b, i, j, k)
    B_dim0,  # l dimension of B
    B_dim1,  # k dimension of B
    A_stride_b, A_stride_i, A_stride_j, A_stride_l,
    B_stride_l, B_stride_k,
    C_stride_b, C_stride_i, C_stride_j, C_stride_k,
    total_batches,  # b * i * j
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows (i dimension)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns (k dimension)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction dimension (l dimension)
):
    # Program IDs
    batch_id = tl.program_id(0)
    block_i = tl.program_id(1)
    block_k = tl.program_id(2)
    
    # Compute the actual batch index (flattened b, i, j)
    # We'll process one (b,i,j) combination per program_id(0)
    # For simplicity, we'll handle one (b,i,j) at a time
    
    # Compute offsets for output block
    # Output block starts at [batch_id, block_i * BLOCK_SIZE_M, block_k * BLOCK_SIZE_N]
    # But since batch_id is already flattened (b*i*j), we need to reconstruct i,j
    # Actually, let's change strategy: process one (b,i,j) at a time, with multiple programs for k
    
    # Compute the base offset for the current (b,i,j) position
    a_base_offset = batch_id * (A_stride_b + A_stride_i + A_stride_j)
    
    # Compute k offsets
    k_offsets = block_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_mask = k_offsets < B_dim1
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the reduction dimension l in blocks
    for l_start in range(0, B_dim0, BLOCK_SIZE_K):
        l_offsets = l_start + tl.arange(0, BLOCK_SIZE_K)
        l_mask = l_offsets < B_dim0
        
        # Load A[block_i * BLOCK_SIZE_M : (block_i+1) * BLOCK_SIZE_M, l_start : l_start + BLOCK_SIZE_K]
        # But since A has shape (b,i,j,l) and we're processing one (b,i,j), A is 1D of length l
        # So we need to load A[l_start : l_start + BLOCK_SIZE_K]
        a_ptr = A_ptr + a_base_offset + l_offsets * A_stride_l
        a_block = tl.load(a_ptr, mask=l_mask, other=0.0)  # shape (BLOCK_SIZE_K,)
        
        # Load B[l_start : l_start + BLOCK_SIZE_K, block_k * BLOCK_SIZE_N : (block_k+1) * BLOCK_SIZE_N]
        b_ptr = B_ptr + l_offsets * B_stride_l + k_offsets * B_stride_k
        b_block = tl.load(b_ptr, mask=l_mask[:, None] & k_mask[None, :], other=0.0)  # shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        
        # Compute partial product
        # a_block is (BLOCK_SIZE_K,) -> needs to broadcast to (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # But we only have one row in the i dimension for this batch_id
        # So we need to handle the i dimension properly
        
        # For now, assume BLOCK_SIZE_M = 1 since for each (b,i,j) we're doing a vector-matrix multiply
        # So let's simplify: we're doing one vector (1×l) times matrix (l×k) = (1×k)
        # Therefore, we should set BLOCK_SIZE_M = 1
    
    # Actually, let's redesign: since for each (b,i,j) we do a 1×l * l×k = 1×k,
    # we should have one program handle one (b,i,j) and multiple programs for k
    
    # Let me rewrite with better understanding:
    # - batch_id processes one (b,i,j) position
    # - We want to compute output[k] = sum_l A[l] * B[l,k]
    
    # Since BLOCK_SIZE_M should be 1 for this specific einsum pattern
    # Let's assume BLOCK_SIZE_M = 1 for optimal performance
    
    # Actually, I'll implement a more general version that can handle multiple rows if needed
    # But for this specific einsum, it's really vector-matrix multiply per (b,i,j)
    
    # Let me implement the optimized version for the specific einsum pattern:
    # "bijl,lk->bijk" means for each b,i,j: 1×l vector × l×k matrix = 1×k result
    
    # Since this is a vector-matrix multiply, let's simplify:
    BLOCK_SIZE_M = 1  # We're computing one row at a time
    
    # Compute offsets
    k_offsets = block_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_mask = k_offsets < B_dim1
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over l dimension
    for l_start in range(0, B_dim0, BLOCK_SIZE_K):
        l_offsets = l_start + tl.arange(0, BLOCK_SIZE_K)
        l_mask = l_offsets < B_dim0
        
        # Load A[l_start:l_start+BLOCK_SIZE_K] for current (b,i,j)
        a_ptr = A_ptr + batch_id * (A_stride_b + A_stride_i + A_stride_j) + l_offsets * A_stride_l
        a_vals = tl.load(a_ptr, mask=l_mask, other=0.0)
        
        # Load B[l_start:l_start+BLOCK_SIZE_K, k_block_start:k_block_end]
        b_ptr = B_ptr + l_offsets[:, None] * B_stride_l + k_offsets[None, :] * B_stride_k
        b_vals = tl.load(b_ptr, mask=l_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate: a_vals is (BLOCK_SIZE_K,), b_vals is (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Result should be (BLOCK_SIZE_N,)
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result
    if block_i == 0:  # Since we only compute one row per batch_id
        c_ptr = C_ptr + batch_id * (C_stride_b + C_stride_i + C_stride_j) + k_offsets * C_stride_k
        tl.store(c_ptr, acc.to(tl.float32), mask=k_mask)


# Actually, let me implement a cleaner version that's optimized for this specific case
@triton.jit
def einsum_v2_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B_l: tl.constexpr,
    B_k: tl.constexpr,
    total_batches: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # One program per (batch, k_block)
    batch_id = tl.program_id(0)
    block_k = tl.program_id(1)
    
    # Compute k offsets for this block
    k_offsets = block_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_mask = k_offsets < B_k
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over l dimension
    for l_start in range(0, B_l, BLOCK_SIZE_K):
        l_offsets = l_start + tl.arange(0, BLOCK_SIZE_K)
        l_mask = l_offsets < B_l
        
        # Load A[b, i, j, l_start:l_start+BLOCK_SIZE_K]
        # Since we're at fixed (b,i,j) = batch_id, A is just a 1D array of length B_l
        a_vals = tl.load(A_ptr + batch_id * B_l + l_offsets, mask=l_mask, other=0.0)
        
        # Load B[l_start:l_start+BLOCK_SIZE_K, k_block_start:k_block_end]
        b_vals = tl.load(
            B_ptr + l_offsets[:, None] * B_k + k_offsets[None, :], 
            mask=l_mask[:, None] & k_mask[None, :], 
            other=0.0
        )
        
        # Accumulate: sum_l A[l] * B[l,k] for each k in this block
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result to C[b, i, j, k_block_start:k_block_end]
    tl.store(C_ptr + batch_id * B_k + k_offsets, acc.to(tl.float32), mask=k_mask)


def triton_einsum(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, f"Dimension mismatch: A has l={l}, B has l={l2}"
    
    # Output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Flatten the first three dimensions (b, i, j) into total_batches
    total_batches = b * i * j
    
    # Configure grid and block sizes
    BLOCK_SIZE_K = 64  # Block size for reduction dimension
    BLOCK_SIZE_N = 128  # Block size for output k dimension
    
    # Grid: (total_batches, ceil(k / BLOCK_SIZE_N))
    grid = (total_batches, triton.cdiv(k, BLOCK_SIZE_N))
    
    # Launch kernel
    einsum_v2_kernel[grid](
        A, B, C,
        B_l=l, B_k=k, total_batches=total_batches,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for the einsum operation.
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_einsum(A, B)