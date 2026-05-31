import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal vector A (N,)
    b_ptr,  # Pointer to matrix B (N, M)
    out_ptr,  # Pointer to output matrix (N, M)
    n_rows,  # N
    n_cols,  # M
    stride_b_row,  # Stride between rows in B
    stride_b_col,  # Stride between columns in B
    stride_out_row,  # Stride between rows in output
    stride_out_col,  # Stride between columns in output
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Get the row index this program instance handles
    row_idx = tl.program_id(0)
    
    # Load the diagonal element for this row
    a_val = tl.load(a_ptr + row_idx)
    
    # Compute column offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    # Calculate base pointers for this row
    b_row_ptr = b_ptr + row_idx * stride_b_row
    out_row_ptr = out_ptr + row_idx * stride_out_row
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE_M):
        # Create mask for valid column indices
        mask = (start_col + col_offsets) < n_cols
        
        # Load the row from B
        b_row = tl.load(
            b_row_ptr + (start_col + col_offsets) * stride_b_col,
            mask=mask,
            other=0.0
        )
        
        # Scale by the diagonal element and store
        result = a_val * b_row
        tl.store(
            out_row_ptr + (start_col + col_offsets) * stride_out_col,
            result,
            mask=mask
        )


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B efficiently using Triton.
    
    Args:
        A: 1D tensor of shape (N,) representing diagonal elements
        B: 2D tensor of shape (N, M)
    
    Returns:
        C: 2D tensor of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty_like(B)
    
    # Use reasonable block sizes
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 1  # Each program handles one row
    
    # Grid: one block per row
    grid = lambda meta: (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs C = diag(A) @ B using custom Triton kernel.
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