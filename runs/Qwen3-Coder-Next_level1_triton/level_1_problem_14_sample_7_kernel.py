import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Custom Triton kernel for matrix multiplication of upper triangular matrices.
    Only computes the upper triangular part of the result.
    """
    # Get the row and column indices for this block
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Create offsets for the row and column
    row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for the result
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Compute matrix multiplication for this block
    # We only need to compute when row <= col (upper triangular part)
    # But we'll compute all and mask later for simplicity
    
    # Loop over the K dimension (inner dimension of matrix multiplication)
    for k in range(0, N, BLOCK_SIZE):
        # Load blocks from A and B
        # For A: only load if column index >= row index (upper triangular)
        a_offsets = row_offsets[:, None] * stride_a + (k + tl.arange(0, BLOCK_SIZE))[None, :]
        a_mask = (row_offsets[:, None] < N) & ((k + tl.arange(0, BLOCK_SIZE))[None, :] < N)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # For B: only load if row index <= column index (upper triangular)
        b_offsets = (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_b + col_offsets[None, :]
        b_mask = ((k + tl.arange(0, BLOCK_SIZE))[:, None] < N) & (col_offsets[None, :] < N)
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate the product
        acc += tl.dot(a, b)
    
    # Store result only for upper triangular part
    c_offsets = row_offsets[:, None] * stride_c + col_offsets[None, :]
    c_mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < N) & (row_offsets[:, None] <= col_offsets[None, :])
    
    # Convert to float32 for storage
    acc = acc.to(tl.float32)
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication for upper triangular matrices using Triton kernel.
    
    Args:
        A: Upper triangular matrix of shape (N, N)
        B: Upper triangular matrix of shape (N, N)
    
    Returns:
        Upper triangular matrix C = A * B of shape (N, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.zeros_like(A)
    
    # Block size for kernel
    BLOCK_SIZE = 128
    
    # Grid dimensions: (num_rows, num_cols) for the output matrix
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
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
        Performs optimized matrix multiplication for upper triangular matrices.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
        
        Returns:
            torch.Tensor: The product of A and B, upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)