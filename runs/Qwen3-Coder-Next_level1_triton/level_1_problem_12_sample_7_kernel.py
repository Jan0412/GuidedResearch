import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal vector (N,)
    B_ptr,  # Pointer to matrix B (N, M)
    C_ptr,  # Pointer to output matrix C (N, M)
    N, M,   # Dimensions
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_SIZE_M: tl.constexpr
):
    # Each program handles one row of the matrix
    row_idx = tl.program_id(0)
    
    # Load the diagonal element for this row
    a_val = tl.load(A_ptr + row_idx)
    
    # Compute column offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    # Compute mask for columns (handle case when M is not divisible by BLOCK_SIZE_M)
    mask = col_offsets < M
    
    # Load the row from B
    b_row_ptr = B_ptr + row_idx * stride_B_row + col_offsets * stride_B_col
    b_row = tl.load(b_row_ptr, mask=mask, other=0.0)
    
    # Multiply by the diagonal element
    c_row = a_val * b_row
    
    # Store the result in C
    c_row_ptr = C_ptr + row_idx * stride_C_row + col_offsets * stride_C_col
    tl.store(c_row_ptr, c_row, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B using a Triton kernel.
    
    Args:
        A (torch.Tensor): 1D tensor of shape (N,)
        B (torch.Tensor): 2D tensor of shape (N, M)
        
    Returns:
        torch.Tensor: Result of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], f"A shape {A.shape} and B shape {B.shape} are incompatible"
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(B)
    
    N, M = B.shape
    
    # Determine block size for columns
    BLOCK_SIZE_M = 128
    
    # Grid is (N, 1, 1) - one block per row
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) @ B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)