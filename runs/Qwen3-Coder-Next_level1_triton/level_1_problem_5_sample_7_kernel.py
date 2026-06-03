import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    a_ptr,  # Pointer to input matrix A
    s,  # Scalar value
    out_ptr,  # Pointer to output matrix
    M,  # Number of rows
    N,  # Number of columns
    stride_am,  # Stride for row dimension of A
    stride_an,  # Stride for column dimension of A
    stride_om,  # Stride for row dimension of output
    stride_on,  # Stride for column dimension of output
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get the row and column indices for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges for rows and columns
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load data from A
    a_ptrs = a_ptr + rm[:, None] * stride_am + rn[None, :] * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    
    # Multiply by scalar
    out = a * s
    
    # Store result
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)


def triton_scalar_mul(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for scalar multiplication.
    It:
      1. Ensures the input tensor is contiguous on GPU.
      2. Creates output tensor.
      3. Calculates the grid (blocks) needed.
      4. Launches the Triton kernel.
    """
    assert A.is_cuda, "Input tensor must be on CUDA."
    A = A.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    M, N = A.shape
    
    # Set block sizes (tunable parameters)
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # Get strides
    stride_am, stride_an = A.stride()
    stride_om, stride_on = out.stride()
    
    # Launch the Triton kernel
    scalar_mul_kernel[grid](
        A, s, out, M, N, stride_am, stride_an, stride_om, stride_on,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for matrix-scalar multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_scalar_mul(A, s)