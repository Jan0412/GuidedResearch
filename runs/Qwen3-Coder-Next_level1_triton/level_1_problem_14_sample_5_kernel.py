import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row and column indices for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for rows and columns
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Perform matrix multiplication for the upper triangular region
    # We only compute elements where row <= col in the output
    # But we need to compute the full block first and then mask
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE):
        # Load A block
        a_mask = (offs_m[:, None] < N) & ((k + tl.arange(0, BLOCK_SIZE)[None, :]) < N)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE)[None, :]) * stride_an,
                   mask=a_mask, other=0.0)
        
        # Load B block
        b_mask = ((k + tl.arange(0, BLOCK_SIZE)[:, None]) < N) & (offs_n[None, :] < N)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_bm + offs_n[None, :] * stride_bn,
                   mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
    
    # Store only upper triangular part
    c = acc.to(tl.float32)
    
    # Create mask for upper triangular part
    row_indices = offs_m[:, None]
    col_indices = offs_n[None, :]
    triu_mask = (row_indices <= col_indices) & (row_indices < N) & (col_indices < N)
    
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c, mask=triu_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = triu(A @ B) for upper triangular matrices A and B.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Block size for optimization
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    triu_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel that fuses the matmul and triu operations.
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
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)