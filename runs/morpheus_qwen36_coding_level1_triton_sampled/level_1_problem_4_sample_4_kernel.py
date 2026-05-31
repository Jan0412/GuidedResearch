import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for Matrix-Vector multiplication C = A * B.
    Each program handles one row of A.
    """
    pid = tl.program_id(0)
    row_idx = pid
    
    # Accumulator for the dot product
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over the K dimension in tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Offsets for the current tile
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Mask for boundary check (handles cases where K is not a multiple of BLOCK_SIZE_K)
        mask = offsets_k < K
        
        # Load A tile: A is (M, K), row-wise access
        # A_ptr + row_idx * K gives the start of the row
        A_tile = tl.load(A_ptr + row_idx * K + offsets_k, mask=mask, other=0.0)
        
        # Load B tile: B is (K, 1), contiguous access
        B_tile = tl.load(B_ptr + offsets_k, mask=mask, other=0.0)
        
        # Compute partial dot product and accumulate
        acc += tl.sum(A_tile * B_tile)
        
    # Store the result for this row
    tl.store(C_ptr + row_idx, acc)


def triton_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton GEMV kernel.
    """
    M, K = A.shape
    assert B.shape == (K, 1), f"Expected B shape ({K}, 1), got {B.shape}"
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Tuning parameters
    BLOCK_SIZE_K = 256
    num_warps = 4
    
    # Grid configuration: one block per row
    grid = (M,)
    
    # Launch kernel
    gemv_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C,
        M=M, K=K,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix-vector multiplication.
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