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
    Custom Triton kernel for upper triangular matrix multiplication.
    Only computes C[i,j] for i <= j, and only sums over relevant k indices.
    """
    # Get row and column indices for the output matrix
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Only compute for upper triangular part (i <= j)
    if row_idx > col_idx:
        return
    
    # Create pointers to the current row in A and relevant column in B
    a_row = A_ptr + row_idx * stride_a
    b_col = B_ptr + col_idx  # column pointer (stride 1 for column access)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute dot product for k from row_idx to col_idx (since A[row,k]=0 for k<row and B[k,col]=0 for k>col)
    # We process in blocks for better performance
    for k_start in range(row_idx, col_idx + 1, BLOCK_SIZE):
        k_end = tl.minimum(k_start + BLOCK_SIZE, col_idx + 1)
        k_range = k_start + tl.arange(0, BLOCK_SIZE)
        mask = (k_range < col_idx + 1) & (k_range >= row_idx)
        
        # Load A[row, k] for k in range
        a_val = tl.load(a_row + k_range * stride_a, mask=mask, other=0.0)
        
        # Load B[k, col] for k in range
        b_val = tl.load(b_col + k_range * stride_b, mask=mask, other=0.0)
        
        # Accumulate product
        acc += a_val * b_val
    
    # Sum the accumulator
    total = tl.sum(acc)
    
    # Store result to C[row, col]
    C_ptr_row = C_ptr + row_idx * stride_c
    tl.store(C_ptr_row + col_idx, total)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for upper triangular matrices using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.zeros_like(A)
    
    # Block size for tiling (tunable parameter)
    BLOCK_SIZE = 128
    
    # Grid: one block per output element in upper triangle
    grid = (N, N)
    
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
    using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.
        """
        return triton_triu_matmul(A, B)