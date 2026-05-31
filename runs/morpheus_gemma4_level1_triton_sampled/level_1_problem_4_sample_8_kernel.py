import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for Matrix-Vector multiplication (C = A * B).
    Each program handles one row of matrix A.
    """
    # Program ID corresponds to the row index of A
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Pointer to the start of the current row in A
    # A is (M, K), so we offset by row_idx * stride_am
    a_row_ptr = A_ptr + row_idx * stride_am
    
    # Accumulator for the dot product of row A[row_idx, :] and vector B[:, 0]
    acc = 0.0
    
    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask = offsets_k < K
        
        # Load a block of the row from A
        # A[row_idx, k : k + BLOCK_SIZE_K]
        a_val = tl.load(a_row_ptr + offsets_k * stride_ak, mask=mask, other=0.0)
        
        # Load a block of the vector B
        # B[k : k + BLOCK_SIZE_K, 0]
        b_val = tl.load(B_ptr + offsets_k * stride_bk, mask=mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(a_val * b_val)
    
    # Store the final result in C[row_idx, 0]
    # Since C is (M, 1), the offset is simply the row index
    tl.store(C_ptr + row_idx, acc)


def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matvec kernel.
    """
    # Ensure inputs are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    # B is expected to be (K, 1)
    
    # Prepare output tensor C of shape (M, 1)
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Get strides for memory indexing
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    
    # Tuning parameter: block size for the K dimension
    BLOCK_SIZE_K = 1024
    
    # Grid: one program per row of A
    grid = (M,)
    
    # Launch kernel
    matvec_kernel[grid](
        A, B, C,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for matrix-vector multiplication.
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
        return triton_matvec(A, B)