import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    N, M,
    stride_bn, stride_bm,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program IDs to determine which block of the output matrix this instance handles
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute the range of rows and columns for this block
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Boundary masks to ensure we don't access memory out of bounds
    mask_n = rn < N
    mask_m = rm < M

    # Load the diagonal elements from A: shape (BLOCK_SIZE_N,)
    a = tl.load(A_ptr + rn, mask=mask_n)

    # Load the block of matrix B: shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # Indexing B using row and column offsets based on strides
    b_offsets = rn[:, None] * stride_bn + rm[None, :] * stride_bm
    b = tl.load(B_ptr + b_offsets, mask=mask_n[:, None] & mask_m[None, :])

    # Perform the multiplication C[i, j] = A[i] * B[i, j]
    # a[:, None] expands (BLOCK_SIZE_N,) to (BLOCK_SIZE_N, 1) for broadcasting
    out = a[:, None] * b

    # Store the result back to the output tensor
    tl.store(Out_ptr + b_offsets, out, mask=mask_n[:, None] & mask_m[None, :])

def triton_diag_matmul(A, B):
    """
    Wrapper for the Triton kernel to perform C = diag(A) @ B.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for efficient memory access
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty_like(B)
    
    # Get strides of B for indexing in the kernel
    stride_bn = B.stride(0)
    stride_bm = B.stride(1)
    
    # Block sizes for tiling
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 128
    
    # Grid dimensions: (number of blocks in N, number of blocks in M)
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )
    
    # Launch the Triton kernel
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        stride_bn, stride_bm,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B, implemented via a custom Triton kernel for speedup.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using the Triton implementation.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)