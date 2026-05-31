import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    A_ptr,      # Pointer to matrix A
    B_ptr,      # Pointer to vector B
    C_ptr,      # Pointer to output vector C
    M,          # Number of rows in A
    K,          # Number of columns in A / elements in B
    stride_am,  # Stride between rows of A
    stride_ak,  # Stride between columns of A
    stride_bk,  # Stride between elements of B
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row of A
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Pointers to the start of the current row in A and the start of B
    row_A_ptr = A_ptr + row_idx * stride_am
    
    # Accumulator for the dot product
    accumulator = 0.0

    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        offsets = k + tl.arange(0, BLOCK_K)
        mask = offsets < K
        
        # Load a block of A (row_idx, k:k+BLOCK_K) and a block of B (k:k+BLOCK_K)
        a = tl.load(row_A_ptr + offsets * stride_ak, mask=mask, other=0.0)
        b = tl.load(B_ptr + offsets * stride_bk, mask=mask, other=0.0)
        
        # Compute partial dot product
        accumulator += tl.sum(a * b)

    # Store the final result in C
    tl.store(C_ptr + row_idx, accumulator)


def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for Matrix-Vector multiplication C = A @ B.
    A: (M, K), B: (K, 1) -> C: (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    # B is (K, 1), we treat it as a vector of size K
    
    # Prepare output tensor
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)

    # Tuning parameter: block size for the K dimension
    BLOCK_K = 1024 

    # Grid: one program per row of A
    grid = (M,)

    matvec_kernel[grid](
        A, B, C, 
        M, K, 
        stride_am, stride_ak, stride_bk, 
        BLOCK_K=BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)