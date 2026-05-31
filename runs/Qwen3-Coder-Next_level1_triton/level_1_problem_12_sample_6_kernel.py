import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal vector (N,)
    b_ptr,  # Pointer to matrix B (N, M)
    out_ptr,  # Pointer to output matrix (N, M)
    N,  # Number of rows
    M,  # Number of columns
    stride_b_row,  # Stride between rows in B
    stride_out_row,  # Stride between rows in output
    BLOCK_SIZE_M: tl.constexpr,
):
    # Get the row index this program instance handles
    row_idx = tl.program_id(0)
    
    # Load the diagonal element for this row
    a_val = tl.load(a_ptr + row_idx)
    
    # Compute column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    # Compute pointers for this row
    b_row_ptr = b_ptr + row_idx * stride_b_row
    out_row_ptr = out_ptr + row_idx * stride_out_row
    
    # Process columns in blocks
    for start_m in range(0, M, BLOCK_SIZE_M):
        cols = start_m + col_offsets
        mask = cols < M
        
        # Load B values for this row and column range
        b_vals = tl.load(b_row_ptr + cols, mask=mask, other=0.0)
        
        # Multiply by diagonal element and store
        out_vals = a_vals = a_val * b_vals
        tl.store(out_row_ptr + cols, out_vals, mask=mask)


def triton_diag_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs C = diag(A) @ B efficiently using Triton kernel.
    
    Args:
        a: 1D tensor of shape (N,) representing diagonal
        b: 2D tensor of shape (N, M)
    
    Returns:
        2D tensor of shape (N, M)
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 1, "A must be 1D"
    assert b.dim() == 2, "B must be 2D"
    assert a.shape[0] == b.shape[0], "A.shape[0] must equal B.shape[0]"
    
    a = a.contiguous()
    b = b.contiguous()
    
    N, M = b.shape
    out = torch.empty_like(b)
    
    # Block size for column dimension
    BLOCK_SIZE_M = 256
    
    # Grid: one block per row
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        a, b, out,
        N, M,
        b.stride(0), out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses custom Triton kernel for diag(A) @ B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs C = diag(A) @ B using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal. Shape: (N,).
            B (torch.Tensor): A 2D tensor. Shape: (N, M).
        
        Returns:
            torch.Tensor: Result of shape (N, M).
        """
        return triton_diag_matmul(A, B)