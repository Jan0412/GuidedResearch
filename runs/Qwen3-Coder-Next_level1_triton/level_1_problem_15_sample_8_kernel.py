import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_triangular_matmul_kernel(
    A_ptr,  # Pointer to lower triangular matrix A
    B_ptr,  # Pointer to lower triangular matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Dimension size
    stride_a,  # Stride for rows in A
    stride_b,  # Stride for rows in B
    stride_c,  # Stride for rows in C
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the output matrix
    row_idx = tl.program_id(0)
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Only process columns where j <= row_idx (lower triangular part)
    mask = col_offsets <= row_idx
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute dot product for C[row_idx, j] where j <= row_idx
    for k in range(N):
        # A[row_idx, k] is non-zero only if k <= row_idx
        if k <= row_idx:
            # Load A[row_idx, k]
            a_val = tl.load(A_ptr + row_idx * stride_a + k)
            
            # B[k, j] is non-zero only if j <= k
            # So for j in [0, min(k, BLOCK_SIZE-1)], B[k,j] is valid
            # We need B[k, col_offsets] where col_offsets <= k
            b_mask = col_offsets <= k
            b_val = tl.load(B_ptr + k * stride_b + col_offsets, mask=b_mask, other=0.0)
            
            # Accumulate only where both conditions hold (j <= row_idx and j <= k)
            combined_mask = mask & b_mask
            acc += a_val * b_val * combined_mask
    
    # Store result
    tl.store(C_ptr + row_idx * stride_c + col_offsets, acc.to(tl.float32), mask=mask)


def triton_lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs optimized matrix multiplication of lower triangular matrices A and B.
    
    Args:
        A (torch.Tensor): Lower triangular matrix of shape (N, N)
        B (torch.Tensor): Lower triangular matrix of shape (N, N)
        
    Returns:
        torch.Tensor: The result of matrix multiplication C of shape (N, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    
    N = A.shape[0]
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1], "Matrices must be square."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Strides
    stride_a = A.stride(0)
    stride_b = B.stride(0)
    stride_c = C.stride(0)
    
    # Set block size - tuned for the problem size
    BLOCK_SIZE = 128
    
    # Grid: one block per row
    grid = (N,)
    
    # Launch kernel
    lower_triangular_matmul_kernel[grid](
        A, B, C, N,
        stride_a, stride_b, stride_c,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of the model that performs matrix multiplication of 
    lower triangular matrices using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_lower_triangular_matmul(A, B)