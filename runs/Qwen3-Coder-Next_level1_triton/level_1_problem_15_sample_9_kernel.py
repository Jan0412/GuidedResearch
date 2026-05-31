import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row and column indices for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the row and column offsets
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over the k dimension
    # For lower triangular matrices, we only need to sum up to min(row, col) but 
    # since B is lower triangular, B[k, col] is zero when k > col
    # Since A is lower triangular, A[row, k] is zero when k > row
    # So we sum k from 0 to min(row, col)
    for k in range(0, N, BLOCK_SIZE):
        # Load A blocks - need to handle the triangular structure
        # A[row, k] is zero when k > row
        a_offsets = rows[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE))[None, :]
        a_mask = (rows[:, None] < N) & ((k + tl.arange(0, BLOCK_SIZE))[None, :] < N) & (rows[:, None] >= (k + tl.arange(0, BLOCK_SIZE))[None, :])
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B blocks - need to handle the triangular structure
        # B[k, col] is zero when k > col
        b_offsets = (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_bn + cols[None, :] * stride_bk
        b_mask = ((k + tl.arange(0, BLOCK_SIZE))[:, None] < N) & (cols[None, :] < N) & ((k + tl.arange(0, BLOCK_SIZE))[:, None] <= cols[None, :])
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate the product
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store the result - only store lower triangular part
    c_offsets = rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < N) & (cols[None, :] < N) & (rows[:, None] >= cols[None, :])
    
    # Convert to output type and store
    accumulator = accumulator.to(tl.float32)
    tl.store(C_ptr + c_offsets, accumulator, mask=c_mask)


def lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of lower triangular matrices A and B.
    
    Args:
        A (torch.Tensor): Lower triangular matrix of shape (N, N).
        B (torch.Tensor): Lower triangular matrix of shape (N, N).
    
    Returns:
        torch.Tensor: The result of matrix multiplication C of shape (N, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.zeros_like(A)
    
    # Configure the kernel
    BLOCK_SIZE = 128
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the kernel
    lower_triangular_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication of lower triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return lower_triangular_matmul(A, B)