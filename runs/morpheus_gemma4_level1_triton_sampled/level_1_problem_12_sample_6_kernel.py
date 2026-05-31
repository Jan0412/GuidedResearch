import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    N, M,
    stride_bn, stride_bm,
    stride_on, stride_om,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Triton kernel to compute C = diag(A) @ B.
    This is equivalent to C[i, j] = A[i] * B[i, j].
    """
    # Program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute offsets for the current block
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Create masks for boundary conditions
    mask_n = rn < N
    mask_m = rm < M
    
    # Load the diagonal elements A[i]
    # A is a 1D tensor of shape (N,)
    a = tl.load(A_ptr + rn, mask=mask_n) # Shape: (BLOCK_SIZE_N,)
    
    # Load the block of B[i, j]
    # B is a 2D tensor of shape (N, M)
    b_offsets = rn[:, None] * stride_bn + rm[None, :] * stride_bm
    b = tl.load(B_ptr + b_offsets, mask=mask_n[:, None] & mask_m[None, :]) # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)
    
    # Perform element-wise multiplication with broadcasting
    # a[:, None] converts (BLOCK_SIZE_N,) to (BLOCK_SIZE_N, 1)
    out = a[:, None] * b
    
    # Store the result in Out_ptr
    out_offsets = rn[:, None] * stride_on + rm[None, :] * stride_om
    tl.store(Out_ptr + out_offsets, out, mask=mask_n[:, None] & mask_m[None, :])

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton diag_matmul_kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure A is contiguous for simple indexing in the kernel
    A = A.contiguous()
    
    N, M = B.shape
    out = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # Get strides for B and Out
    stride_bn, stride_bm = B.stride()
    stride_on, stride_om = out.stride()
    
    # Tunable block sizes
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 64
    
    # Define the grid (number of blocks in N and M dimensions)
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )
    
    # Launch the kernel
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        stride_bn, stride_bm,
        stride_on, stride_om,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B, implemented using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using the optimized Triton implementation.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)