import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    a_ptr,  # Pointer to input matrix A
    s,      # Scalar value
    out_ptr,  # Pointer to output matrix
    M, N,     # Dimensions of the matrix
    stride_a, stride_out,  # Strides for the matrices
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Block index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting offsets for this block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load the block of A
    a_ptrs = a_ptr + rm[:, None] * stride_a + rn[None, :] * stride_out
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    
    # Multiply by scalar
    out = a * s
    
    # Store the result
    out_ptrs = out_ptr + rm[:, None] * stride_a + rn[None, :] * stride_out
    tl.store(out_ptrs, out, mask=mask)


def triton_scalar_mul(a: torch.Tensor, s: float) -> torch.Tensor:
    """
    Performs matrix-scalar multiplication using Triton kernel.
    
    Args:
        a: Input matrix of shape (M, N)
        s: Scalar value
        
    Returns:
        Resulting matrix of shape (M, N)
    """
    # Ensure input is contiguous and on CUDA
    a = a.contiguous()
    if not a.is_cuda:
        a = a.cuda()
    
    # Prepare output tensor
    out = torch.empty_like(a)
    
    # Get dimensions
    M, N = a.shape
    
    # Configure the kernel launch grid
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )
    
    # Launch the kernel
    scalar_mul_kernel[grid](
        a, s, out,
        M, N,
        a.stride(0), out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


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
        return triton_scalar_mul(A, s)