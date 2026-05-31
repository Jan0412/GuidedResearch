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
    stride_cm,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row of A
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Pointer to the start of the row in A
    a_ptr = A_ptr + row_idx * stride_am
    
    # Accumulator for the dot product
    acc = 0.0
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        offsets_k = k + tl.arange(0, BLOCK_K)
        mask = offsets_k < K
        
        # Load elements from A (row row_idx) and B (column 0)
        a = tl.load(a_ptr + offsets_k * stride_ak, mask=mask, other=0.0)
        b = tl.load(B_ptr + offsets_k * stride_bk, mask=mask, other=0.0)
        
        # Compute partial dot product and accumulate
        acc += tl.sum(a * b)
    
    # Store the final result in C
    tl.store(C_ptr + row_idx * stride_cm, acc)

def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for matrix-vector multiplication C = A * B.
    A: (M, K)
    B: (K, 1)
    C: (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk = B.stride(0)
    stride_cm = out.stride(0)
    
    # Grid: one block per row of A
    grid = (M,)
    
    # Block size for the K dimension reduction
    BLOCK_K = 1024
    
    matvec_kernel[grid](
        A, B, out,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        stride_cm,
        BLOCK_K=BLOCK_K
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using Triton.
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
        # Use the custom Triton kernel instead of torch.matmul
        return triton_matvec(A, B)