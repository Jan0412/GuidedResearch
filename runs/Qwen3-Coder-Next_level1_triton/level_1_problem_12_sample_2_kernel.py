import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal vector (1D tensor of size N)
    B_ptr,  # Pointer to matrix B (2D tensor of shape N x M)
    C_ptr,  # Pointer to output matrix C (2D tensor of shape N x M)
    N, M,  # Dimensions
    stride_B_row, stride_B_col,  # Strides for B
    stride_C_row, stride_C_col,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID maps to row in the diagonal matrix (which row of B to process)
    row_idx = tl.program_id(0)
    
    # Load the diagonal element for this row
    a_val = tl.load(A_ptr + row_idx)
    
    # Start offset for the current row in B and C
    b_row_start = row_idx * stride_B_row
    c_row_start = row_idx * stride_C_row
    
    # Create column offsets
    cols = tl.arange(0, BLOCK_SIZE_M)
    mask = cols < M
    
    # Load the row from B
    b_row = tl.load(B_ptr + b_row_start + cols * stride_B_col, mask=mask, other=0.0)
    
    # Multiply by diagonal element
    c_row = a_val * b_row
    
    # Store result to C
    tl.store(C_ptr + c_row_start + cols * stride_C_col, c_row, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B using a custom Triton kernel.
    
    Args:
        A (torch.Tensor): 1D tensor of shape (N,) representing the diagonal
        B (torch.Tensor): 2D tensor of shape (N, M)
    
    Returns:
        torch.Tensor: Result of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    N, M = B.shape
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output
    C = torch.empty_like(B)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE_M = 256
    
    # Grid: one block per row of the diagonal matrix
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for diag(A) @ B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication C = diag(A) @ B using a Triton kernel.
        """
        return triton_diag_matmul(A, B)