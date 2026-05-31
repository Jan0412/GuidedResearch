import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for matrix-vector multiplication
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_bb,
    stride_cm, stride_cb,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID corresponds to the row of matrix A to compute
    pid_m = tl.program_id(0)
    
    # Create offset for rows of A and C
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < M
    
    # Initialize accumulator for dot product
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over columns of A (rows of B) in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        col_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        col_mask = col_offsets < K
        
        # Load block of A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_block = tl.load(
            A_ptr + row_offsets[:, None] * stride_am + col_offsets[None, :] * stride_ak,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0
        )
        
        # Load block of B: (BLOCK_SIZE_K, 1)
        b_block = tl.load(
            B_ptr + col_offsets[:, None] * stride_bk,
            mask=col_mask[:, None],
            other=0.0
        )
        
        # Compute partial dot product and accumulate
        acc += tl.sum(a_block * b_block[None, :], axis=1)
    
    # Store result
    output_offsets = row_offsets * stride_cm
    tl.store(
        C_ptr + output_offsets,
        acc,
        mask=row_mask
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K, 1)
        
    Returns:
        Output vector of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Matrix and vector dimensions must match for multiplication."
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, _ = B.shape  # B is (K, 1)
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 256
    
    # Calculate grid dimensions
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, 1)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)