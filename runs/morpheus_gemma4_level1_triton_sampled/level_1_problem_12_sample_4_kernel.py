import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,      # Pointer to diagonal vector A
    B_ptr,      # Pointer to matrix B
    Out_ptr,    # Pointer to output matrix Out
    N,          # Number of rows
    M,          # Number of columns
    stride_bn,  # Stride of B along the row dimension
    stride_bm,  # Stride of B along the column dimension
    stride_on,  # Stride of Out along the row dimension
    stride_om,  # Stride of Out along the column dimension
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Calculate offsets for the current block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Create masks for boundary conditions
    mask_n = offs_n < N
    mask_m = offs_m < M
    mask = mask_n[:, None] & mask_m[None, :]

    # Load the diagonal elements of A for the current block of rows
    # A is a 1D tensor of shape (N,)
    a = tl.load(A_ptr + offs_n, mask=mask_n, other=0.0)

    # Load the block of matrix B
    # B is a 2D tensor of shape (N, M)
    # B_ptr + row_idx * stride_bn + col_idx * stride_bm
    b_ptr = B_ptr + offs_n[:, None] * stride_bn + offs_m[None, :] * stride_bm
    b = tl.load(b_ptr, mask=mask, other=0.0)

    # Perform the multiplication: C[i, j] = A[i] * B[i, j]
    # a is (BLOCK_SIZE_N,), b is (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # Broadcast a to (BLOCK_SIZE_N, 1) for element-wise multiplication
    out = a[:, None] * b

    # Store the result in the output matrix
    out_ptr = Out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    tl.store(out_ptr, out, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for the operation C = diag(A) @ B.
    This is mathematically equivalent to row-wise scaling of B by A.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for simpler indexing in the kernel
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty((N, M), device=B.device, dtype=B.dtype)

    # Tunable block sizes
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 64

    # Grid dimensions: (number of blocks for N, number of blocks for M)
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )

    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B, implemented via a custom Triton kernel for efficiency.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)