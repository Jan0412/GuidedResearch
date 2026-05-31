import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for Matrix-Vector multiplication.
    Each program instance computes one element of the output vector (one row of A dot B).
    """
    # Program ID corresponds to the row of A
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Pointer to the start of the current row in A
    a_row_ptr = A_ptr + row_idx * stride_am
    
    # Initialize accumulator for the dot product
    accumulator = 0.0
    
    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask = offsets < K
        
        # Load a block of the row from A and a block from vector B
        a_vals = tl.load(a_row_ptr + offsets * stride_ak, mask=mask, other=0.0)
        b_vals = tl.load(B_ptr + offsets * stride_bk, mask=mask, other=0.0)
        
        # Compute partial dot product and accumulate
        accumulator += tl.sum(a_vals * b_vals)

    # Store the final result in the output vector
    tl.store(Out_ptr + row_idx, accumulator)

def triton_gemv(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton GEMV kernel.
    A: (M, K)
    B: (K, 1)
    """
    # Ensure inputs are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    # B is expected to be (K, 1)
    
    # Output is a vector of size M
    out = torch.empty((M,), device=A.device, dtype=A.dtype)
    
    # Strides for pointer arithmetic
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    
    # Tuning parameter for the reduction block size
    BLOCK_SIZE_K = 1024
    
    # Grid is 1D: one program per row of A
    grid = (M,)
    
    gemv_kernel[grid](
        A, B, out,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Reshape output to (M, 1) to match torch.matmul behavior
    return out.view(M, 1)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using a custom Triton kernel.
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
        return triton_gemv(A, B)