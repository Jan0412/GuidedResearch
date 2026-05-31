import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr,  # Pointer to input matrix A
    B_ptr,  # Pointer to input matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Size of the matrices (N x N)
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create row and column offsets
    row_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices within the matrix bounds
    row_mask = row_offsets < N
    col_mask = col_offsets < N
    full_mask = row_mask[:, None] & col_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Perform matrix multiplication for upper triangular part only
    for k in range(0, N, BLOCK_SIZE):
        # Offsets for the K dimension
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < N
        
        # Create masks for A and B
        a_mask = row_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & col_mask[None, :]
        
        # Load A and B tiles (only from upper triangular part)
        a_tile = tl.load(A_ptr + row_offsets[:, None] * stride_a + k_offsets[None, :], 
                         mask=a_mask, other=0.0)
        b_tile = tl.load(B_ptr + k_offsets[:, None] * stride_b + col_offsets[None, :], 
                         mask=b_mask, other=0.0)
        
        # Only compute if both indices are in upper triangular part
        # We need A[i, k] where i <= k and B[k, j] where k <= j for the result C[i,j] to be non-zero
        # For C[i,j] (where i <= j), we need sum over k where i <= k and k <= j
        # So we only accumulate if k >= i and k <= j
        # Create mask for valid k for this (i,j) pair
        i_indices = row_offsets[:, None]
        j_indices = col_offsets[None, :]
        k_indices = k_offsets[None, :]
        # Only accumulate if i <= k and k <= j
        valid_k_mask = (i_indices <= k_indices) & (k_indices <= j_indices)
        
        # Multiply and accumulate with proper masking
        acc += tl.where(valid_k_mask, a_tile * b_tile, 0.0)
    
    # Convert to float16 if needed, but keeping as float32 for precision
    c_out = acc.to(tl.float32)
    
    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * stride_c + col_offsets[None, :], 
             c_out, mask=full_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes the upper triangular part of matrix multiplication C = A * B,
    where A and B are upper triangular matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Block size for tiling (tunable parameter)
    BLOCK_SIZE = 64
    
    # Grid dimensions: (num_blocks_m, num_blocks_n)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the Triton kernel
    triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), B.stride(0), C.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)