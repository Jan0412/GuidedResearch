import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batch_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the block index
    blk_id = tl.program_id(1)
    
    # Compute the group ID
    group_id = blk_id // GROUP_M
    # Compute the block ID within the group
    blk_id_in_group = blk_id % GROUP_M
    
    # Compute the starting positions for this block
    pid_m = group_id * GROUP_M + blk_id_in_group
    pid_n = tl.program_id(2)
    
    # Compute the starting positions for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for valid indices
    mask_m = offs_m < m
    mask_n = offs_n < n
    mask_k = offs_k < k
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_idx in range(0, k, BLOCK_K):
        # Load A and B tiles
        a_ptrs = A_ptr + (
            batch_idx * stride_a_batch +
            tl.max(offs_m, 0)[:, None] * stride_a_m +
            (k_idx + offs_k[None, :]) * stride_a_k
        )
        b_ptrs = B_ptr + (
            batch_idx * stride_b_batch +
            (k_idx + offs_k[:, None]) * stride_b_k +
            tl.max(offs_n, 0)[None, :] * stride_b_n
        )
        
        # Load tiles with appropriate masking
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        
        # Perform matrix multiplication for this tile
        acc += tl.dot(a, b)
    
    # Compute output pointers
    c_ptrs = C_ptr + (
        batch_idx * stride_c_batch +
        tl.max(offs_m, 0)[:, None] * stride_c_m +
        tl.max(offs_n, 0)[None, :] * stride_c_n
    )
    
    # Store results
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))

def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match."
    assert A.shape[2] == B.shape[1], "Inner dimensions must match."
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Prepare output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device='cuda')
    
    # Define block sizes and group size
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),
        1
    )
    
    # Launch kernel
    batch_matmul_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
    )
    
    return C

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized with custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_batch_matmul(A, B)