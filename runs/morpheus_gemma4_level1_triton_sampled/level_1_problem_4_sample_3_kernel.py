import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to matrix A
    b_ptr,  # Pointer to vector B
    c_ptr,  # Pointer to output vector C
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a block of M rows
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = row_offsets < M

    # Pointers to the current block of rows in A
    # row_offsets is (BLOCK_M,), stride_am is the distance between rows
    a_row_ptrs = a_ptr + row_offsets[:, None] * stride_am

    accumulator = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load a block of the vector B (1, BLOCK_K)
        b_vals = tl.load(b_ptr + offs_k * stride_bk, mask=mask_k, other=0.0)

        # Load a block of matrix A (BLOCK_M, BLOCK_K)
        # a_row_ptrs is (BLOCK_M, 1), offs_k is (BLOCK_K,)
        a_vals = tl.load(a_row_ptrs + offs_k[None, :] * stride_ak, mask=mask_k[None, :], other=0.0)

        # Compute partial dot product: (BLOCK_M, BLOCK_K) * (1, BLOCK_K) -> sum over axis 1 -> (BLOCK_M,)
        accumulator += tl.sum(a_vals * b_vals[None, :], axis=1)

    # Store the final results into C
    tl.store(c_ptr + row_offsets, accumulator, mask=mask_m)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for Matrix-Vector multiplication.
    A: (M, K)
    B: (K, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    # B is (K, 1), we treat it as a vector of size K
    
    # Prepare output tensor (M, 1)
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    C_vec = C.view(-1) # Flatten for easier indexing in kernel

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)

    # Tuning parameters
    BLOCK_M = 32
    BLOCK_K = 1024

    # Grid: Parallelize over the M dimension
    grid = (triton.cdiv(M, BLOCK_M),)

    matmul_kernel[grid](
        A, B, C_vec,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
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
        Performs matrix-vector multiplication using Triton.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)