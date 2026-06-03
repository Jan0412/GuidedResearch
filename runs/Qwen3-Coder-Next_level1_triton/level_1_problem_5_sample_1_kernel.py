import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mult_kernel(
    A_ptr,  # Pointer to input matrix A
    s,      # Scalar value (passed as a float)
    C_ptr,  # Pointer to output matrix C
    M, N,   # Dimensions of the matrix
    stride_am, stride_an,  # Strides for A (typically N for contiguous row-major)
    stride_cm, stride_cn,  # Strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Compute row and column indices for this program
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting offsets for rows and columns
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    mask_m = offset_m < M
    mask_n = offset_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load the block from A
    A_block = tl.load(
        A_ptr + offset_m[:, None] * stride_am + offset_n[None, :] * stride_an,
        mask=mask,
        other=0.0
    )
    
    # Multiply by scalar and store to C
    C_block = A_block * s
    tl.store(
        C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn,
        C_block,
        mask=mask
    )


def triton_scalar_mult(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Performs matrix-scalar multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, N)
        s: Scalar value
        
    Returns:
        C: Output tensor of shape (M, N)
    """
    assert A.is_cuda, "Input tensor must be on CUDA device."
    A = A.contiguous()
    
    M, N = A.shape
    C = torch.empty_like(A)
    
    # Set block sizes for reasonable occupancy
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # Launch kernel
    scalar_mult_kernel[grid](
        A, s, C,
        M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_scalar_mult(A, s)