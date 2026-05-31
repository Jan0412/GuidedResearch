import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for lower triangular matrix multiplication.
    Only computes C[i,j] for i >= j (lower triangular part).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate row and column indices
    row_start = pid_m * BLOCK_SIZE
    col_start = pid_n * BLOCK_SIZE
    
    # Create row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create broadcasted grids for row and column indices
    row_indices = row_offsets[:, None]
    col_indices = col_offsets[None, :]
    
    # Only compute for lower triangular part (i >= j)
    mask = row_indices >= col_indices
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over the K dimension (inner dimension of matrix multiply)
    for k in range(0, N, BLOCK_SIZE):
        # Create k offsets
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        
        # Load A tile: shape (BLOCK_SIZE, BLOCK_SIZE)
        # A is lower triangular, so we need to handle out-of-bounds properly
        a_row = row_offsets[:, None]
        a_col = k_offsets[None, :]
        a_mask = (a_row < N) & (a_col < N) & (a_row >= a_col)
        a = tl.load(A_ptr + a_row * stride_am + a_col * stride_an, 
                   mask=a_mask, other=0.0)
        
        # Load B tile: shape (BLOCK_SIZE, BLOCK_SIZE)
        # B is lower triangular
        b_row = k_offsets[:, None]
        b_col = col_offsets[None, :]
        b_mask = (b_row < N) & (b_col < N) & (b_row >= b_col)
        b = tl.load(B_ptr + b_row * stride_bm + b_col * stride_bn,
                   mask=b_mask, other=0.0)
        
        # Accumulate multiplication
        acc += tl.dot(a, b)
    
    # Store result only for lower triangular part
    c_row = row_offsets[:, None]
    c_col = col_offsets[None, :]
    c_mask = (c_row < N) & (c_col < N) & mask
    
    # Convert to float16 if needed for storage (but computation was in float32)
    c = acc.to(tl.float32)
    tl.store(C_ptr + c_row * stride_cm + c_col * stride_cn, c, mask=c_mask)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of lower triangular matrices A and B,
    returning only the lower triangular part of the result.
    
    Args:
        A: Lower triangular matrix of shape (N, N)
        B: Lower triangular matrix of shape (N, N)
    
    Returns:
        Lower triangular matrix C of shape (N, N) where C = tril(A @ B)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.zeros_like(A)
    
    # Define block size (tunable parameter)
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model for lower triangular matrix multiplication using Triton.
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
        return triton_tril_matmul(A, B)