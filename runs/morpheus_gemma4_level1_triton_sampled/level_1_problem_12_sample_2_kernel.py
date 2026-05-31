import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,    # Pointer to diagonal vector A
    B_ptr,    # Pointer to matrix B
    C_ptr,    # Pointer to output matrix C
    N,        # Number of rows
    M,        # Number of columns
    stride_bn, # Stride of B along rows (usually M)
    stride_bm, # Stride of B along columns (usually 1)
    stride_cn, # Stride of C along rows (usually M)
    stride_cm, # Stride of C along columns (usually 1)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Calculate offsets for the current block
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Create masks to handle boundary conditions
    mask_n = offsets_n < N
    mask_m = offsets_m < M

    # Load the diagonal elements from A
    # A is 1D, so we just need the offset
    a = tl.load(A_ptr + offsets_n, mask=mask_n, other=0.0)

    # Load the block from B
    # B is 2D, index as [row, col]
    # We use broadcasting to create the 2D index grid
    b_offsets = offsets_n[:, None] * stride_bn + offsets_m[None, :] * stride_bm
    b = tl.load(B_ptr + b_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0)

    # Perform the multiplication: each row i of B is scaled by A[i]
    # a is (BLOCK_SIZE_N,), b is (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # We reshape a to (BLOCK_SIZE_N, 1) for broadcasting
    res = a[:, None] * b

    # Store the result in C
    c_offsets = offsets_n[:, None] * stride_cn + offsets_m[None, :] * stride_cm
    tl.store(C_ptr + c_offsets, res, mask=mask_n[:, None] & mask_m[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized diagonal matrix multiplication C = diag(A) @ B
    A: (N,)
    B: (N, M)
    C: (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous for the kernel
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    C = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # Tuning parameters
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32
    
    # Grid dimensions
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    Implemented using a custom Triton kernel for O(NM) complexity.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using Triton.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)