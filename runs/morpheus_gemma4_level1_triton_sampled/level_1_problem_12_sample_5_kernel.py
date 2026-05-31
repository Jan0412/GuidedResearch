import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    # Program IDs for rows and columns
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute offsets for the current block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Masks to handle boundary conditions
    mask_n = offs_n < N
    mask_m = offs_m < M

    # Load the diagonal elements from A
    # A is a 1D tensor of shape (N,)
    a = tl.load(A_ptr + offs_n, mask=mask_n) # Shape: (BLOCK_SIZE_N,)

    # Load the corresponding block from B
    # B is a 2D tensor of shape (N, M)
    b_ptr = B_ptr + (offs_n[:, None] * stride_B_row + offs_m[None, :] * stride_B_col)
    b = tl.load(b_ptr, mask=mask_n[:, None] & mask_m[None, :]) # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)

    # Perform the multiplication: C[i, j] = A[i] * B[i, j]
    # Broadcast A across the columns of B
    res = a[:, None] * b

    # Store the result in C
    c_ptr = C_ptr + (offs_n[:, None] * stride_C_row + offs_m[None, :] * stride_C_col)
    tl.store(c_ptr, res, mask=mask_n[:, None] & mask_m[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication of a diagonal matrix (represented by its diagonal A) 
    and a matrix B.
    C = diag(A) @ B
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for correct pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    
    # Prepare output tensor
    C = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # Strides
    stride_B_row, stride_B_col = B.stride()
    stride_C_row, stride_C_col = C.stride()
    
    # Block sizes
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32
    
    # Grid dimensions
    grid = ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        stride_B_row, stride_B_col,
        stride_C_row, stride_C_col,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix 
    with another matrix using a custom Triton kernel.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)