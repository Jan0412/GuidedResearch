import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,      # Pointer to diagonal vector A (N,)
    b_ptr,      # Pointer to matrix B (N, M)
    out_ptr,    # Pointer to output matrix C (N, M)
    N,          # Number of rows
    M,          # Number of columns
    stride_bn,  # Stride of B along rows
    stride_bm,  # Stride of B along columns
    stride_cn,  # Stride of C along rows
    stride_cm,  # Stride of C along columns
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program IDs for the 2D grid
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute offsets for the current block
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Masks to handle boundaries
    mask_n = offsets_n < N
    mask_m = offsets_m < M

    # Load the diagonal elements A[i] for the current block of rows
    # a is shape (BLOCK_SIZE_N,)
    a = tl.load(a_ptr + offsets_n, mask=mask_n, other=0.0)

    # Load the block of matrix B
    # b is shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    b_offsets = offsets_n[:, None] * stride_bn + offsets_m[None, :] * stride_bm
    b = tl.load(b_ptr + b_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0)

    # Perform the operation: C[i, j] = A[i] * B[i, j]
    # Broadcast a to (BLOCK_SIZE_N, 1) to multiply with b (BLOCK_SIZE_N, BLOCK_SIZE_M)
    out = a[:, None] * b

    # Store the result in matrix C
    out_offsets = offsets_n[:, None] * stride_cn + offsets_m[None, :] * stride_cm
    tl.store(out_ptr + out_offsets, out, mask=mask_n[:, None] & mask_m[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton implementation of C = diag(A) @ B, which is equivalent to row-wise scaling.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for predictable strides, though kernel handles strides
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty((N, M), device=B.device, dtype=B.dtype)

    # Tuning parameters
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 128

    # Grid: one block for every tile of the output matrix
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
    C = diag(A) * B
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
        # The operation torch.diag(A) @ B is mathematically equivalent to 
        # multiplying each row i of B by the scalar A[i].
        return triton_diag_matmul(A, B)