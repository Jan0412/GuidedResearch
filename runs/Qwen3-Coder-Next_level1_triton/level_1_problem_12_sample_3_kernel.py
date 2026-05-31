import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal vector (N,)
    B_ptr,  # Pointer to matrix B (N, M)
    C_ptr,  # Pointer to output matrix C (N, M)
    N,      # Number of rows
    M,      # Number of columns
    stride_B,  # Stride between rows in B
    stride_C,  # Stride between rows in C
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID represents the row index
    row_idx = tl.program_id(0)
    
    # Load the diagonal element for this row
    a_val = tl.load(A_ptr + row_idx)
    
    # Compute column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute pointer offsets for this row
    b_offset = row_idx * stride_B + col_offsets
    c_offset = row_idx * stride_C + col_offsets
    
    # Create mask to handle cases where M is not a multiple of BLOCK_SIZE
    mask = col_offsets < M
    
    # Load the row of B
    b_row = tl.load(B_ptr + b_offset, mask=mask, other=0.0)
    
    # Multiply by the diagonal element
    c_row = b_row * a_val
    
    # Store the result
    tl.store(C_ptr + c_offset, c_row, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs C = diag(A) @ B efficiently using a Triton kernel.
    
    Args:
        A: 1D tensor of shape (N,) representing the diagonal
        B: 2D tensor of shape (N, M)
        
    Returns:
        C: 2D tensor of shape (N, M) where C[i,j] = A[i] * B[i,j]
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    # Create output tensor
    C = torch.empty_like(B)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Grid: one block per row
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), C.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs C = diag(A) @ B using a custom Triton kernel.
    Instead of forming the diagonal matrix explicitly, computes row-wise scaling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication efficiently.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)