import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,  # Pointer to matrix A (M, K)
    B_ptr,  # Pointer to vector B (K,)
    out_ptr,  # Pointer to output (M,)
    M,  # Number of rows in A
    K,  # Number of columns in A
    stride_am,  # Stride for row in A
    stride_bk,  # Stride for element in B (should be 1 for contiguous)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID corresponds to the row index in M dimension
    pid = tl.program_id(0)
    
    # Load the row of A (this row spans the full K dimension)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M
    
    # Initialize accumulators for dot product
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load block of A: (BLOCK_M, BLOCK_K)
        a_block = tl.load(
            A_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load block of B: (BLOCK_K,)
        b_block = tl.load(
            B_ptr + k_offsets,
            mask=k_mask,
            other=0.0
        )
        
        # Compute partial dot products
        acc += tl.sum(a_block * b_block[None, :], axis=1)
    
    # Store results
    out_offsets = row_offsets
    out_mask = row_mask
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    A: (M, K), B: (K,) or (K, 1), output: (M,) or (M, 1)
    """
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    B_flat = B.view(-1)  # Flatten to (K,) for easier indexing
    
    # Prepare output tensor
    out = torch.empty(M, dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_M = 32  # Number of rows processed per block
    BLOCK_K = 256  # Block size for K dimension (tunable)
    
    # Grid: one block per row (or row group)
    grid = (triton.cdiv(M, BLOCK_M),)
    
    # Launch kernel
    matvec_kernel[grid](
        A, B_flat, out,
        M, K,
        A.stride(0), B_flat.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    
    # Reshape output to match expected (M, 1) if B was (K, 1)
    if B.dim() == 2 and B.size(1) == 1:
        return out.view(M, 1)
    else:
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)