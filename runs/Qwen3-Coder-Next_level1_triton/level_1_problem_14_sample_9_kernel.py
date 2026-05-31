import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triangular_matmul_kernel(
    A_ptr,  # Pointer to input matrix A
    B_ptr,  # Pointer to input matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Dimension of the square matrices
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr = False,
):
    # Each program instance computes a tile of the output matrix C
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute the starting indices for the tile
    rm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    rm_mask = rm < N
    cn_mask = cn < N
    rm = tl.where(rm_mask, rm, 0)
    cn = tl.where(cn_mask, cn, 0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, N):
        # Load A row: A[rm, k]
        a_k = tl.load(A_ptr + rm * stride_a + k, mask=rm_mask, other=0.0)
        
        # Load B column: B[k, cn]
        b_k = tl.load(B_ptr + k * stride_b + cn, mask=cn_mask, other=0.0)
        
        # Accumulate product: A[rm, k] * B[k, cn]
        acc += a_k[:, None] * b_k[None, :]
    
    # Store result only for upper triangular part
    # For upper triangular: rm <= cn
    c_tile = acc.to(tl.float32)
    
    # Apply upper triangular mask
    mask = rm[:, None] <= cn[None, :]
    c_tile = tl.where(mask, c_tile, 0.0)
    
    # Store the result
    tl.store(C_ptr + rm[:, None] * stride_c + cn[None, :], c_tile, mask=rm_mask[:, None] & cn_mask[None, :])


def triton_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication for upper triangular matrices using Triton kernel.
    
    Args:
        A (torch.Tensor): Upper triangular matrix of shape (N, N).
        B (torch.Tensor): Upper triangular matrix of shape (N, N).
        
    Returns:
        torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices A and B must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Set up strides
    stride_a = A.stride(0)
    stride_b = B.stride(0)
    stride_c = C.stride(0)
    
    # Grid configuration
    BLOCK_SIZE = 128
    num_pid_m = triton.cdiv(N, BLOCK_SIZE)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    triangular_matmul_kernel[grid](
        A, B, C, N,
        stride_a, stride_b, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using optimized Triton kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triangular_matmul(A, B)