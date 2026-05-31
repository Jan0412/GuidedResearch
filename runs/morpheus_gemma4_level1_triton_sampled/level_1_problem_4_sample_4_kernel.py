import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    A_ptr,      # Pointer to matrix A (M, K)
    B_ptr,      # Pointer to vector B (K, 1)
    Out_ptr,    # Pointer to output vector Out (M, 1)
    M,          # Number of rows in A
    K,          # Number of columns in A / elements in B
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID corresponds to the block of rows in M
    pid = tl.program_id(0)
    
    # Row offsets for this program
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column offsets for the inner loop
    rk = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over the K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Load a block of A: (BLOCK_M, BLOCK_K)
        # A is row-major: index = row * K + col
        a_offsets = rm[:, None] * K + (k + rk[None, :])
        a_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load a block of B: (BLOCK_K,)
        b_offsets = k + rk
        b_mask = (k + rk) < K
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute partial dot product for each row in the block
        # (BLOCK_M, BLOCK_K) * (1, BLOCK_K) -> sum along axis 1 -> (BLOCK_M,)
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store the final result in the output vector
    out_mask = rm < M
    tl.store(Out_ptr + rm, acc, mask=out_mask)

def triton_gemv(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for matrix-vector multiplication C = A @ B.
    """
    # Ensure inputs are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    # Output shape is (M, 1)
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Tuning parameters
    BLOCK_M = 32
    BLOCK_K = 1024
    
    # Grid: one program for every BLOCK_M rows
    grid = (triton.cdiv(M, BLOCK_M),)
    
    gemv_kernel[grid](
        A, B, out, 
        M, K, 
        BLOCK_M=BLOCK_M, 
        BLOCK_K=BLOCK_K
    )
    
    return out

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
        return triton_gemv(A, B)