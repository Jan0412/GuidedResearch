import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_vector_kernel(
    A_ptr,  # Pointer to matrix A (M, K)
    B_ptr,  # Pointer to vector B (K,)
    output_ptr,  # Pointer to output (M,)
    M,  # Number of rows in A
    K,  # Number of columns in A / length of B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one output row
    pid_m = tl.program_id(0)
    
    # Offset for the current row in A
    a_offset = pid_m * K
    
    # Initialize accumulator for dot product
    accumulator = tl.zeros([1], dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        # Mask for K dimension to handle edge cases
        k_mask = k_offsets < K
        
        # Load a row of A: shape (BLOCK_SIZE_K,)
        a_row = tl.load(A_ptr + a_offset + k_offsets, mask=k_mask, other=0.0)
        
        # Load B: shape (BLOCK_SIZE_K,)
        b_vec = tl.load(B_ptr + k_offsets, mask=k_mask, other=0.0)
        
        # Accumulate partial dot product
        accumulator += tl.sum(a_row * b_vec, axis=0)
    
    # Store result
    output_offset = pid_m
    tl.store(output_ptr + output_offset, accumulator)


def triton_matmul_vector(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K,)
    
    Returns:
        Output vector of shape (M,)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    assert B.shape == (K,), f"Expected B shape (K,), got {B.shape}"
    
    # Prepare output tensor
    output = torch.empty(M, dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 1  # Each program handles one row
    BLOCK_SIZE_K = 256  # Block size for K dimension
    
    # Grid: one program per row
    grid = (M,)
    
    # Launch the Triton kernel
    matmul_vector_kernel[grid](
        A, B, output,
        M, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape output to (M, 1) to match expected output shape
    return output.view(M, 1)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.
        
        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K,).
        
        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul_vector(A, B)