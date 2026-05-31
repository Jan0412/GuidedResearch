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
    # Get the batch index for this program
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Get the block index for this program
    block_id = tl.program_id(1)
    
    # Compute the number of blocks needed for the M dimension
    m_blocks = (m + BLOCK_M - 1) // BLOCK_M
    n_blocks = (n + BLOCK_N - 1) // BLOCK_N
    
    # Compute the group index
    group_id = block_id // GROUP_M
    group_size = min(GROUP_M, n_blocks - group_id * GROUP_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_block in range(0, (k + BLOCK_K - 1) // BLOCK_K):
        # Load A and B tiles
        a_tile = tl.load(
            A_ptr + 
            batch_idx * stride_a_batch +
            tl.arange(0, BLOCK_M)[:, None] * stride_a_m +
            (k_block * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k,
            mask=(tl.arange(0, BLOCK_M)[:, None] < m) & 
                  (k_block * BLOCK_K + tl.arange(0, BLOCK_K)[None, :] < k),
            other=0.0
        )
        
        b_tile = tl.load(
            B_ptr + 
            batch_idx * stride_b_batch +
            (k_block * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]) * stride_b_k +
            tl.arange(0, BLOCK_N)[None, :] * stride_b_n,
            mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K)[:, None] < k) & 
                  (tl.arange(0, BLOCK_N)[None, :] < n),
            other=0.0
        )
        
        # Matrix multiply
        acc += tl.dot(a_tile, b_tile)
    
    # Write back result
    c_tile = acc.to(tl.float32)
    tl.store(
        C_ptr + 
        batch_idx * stride_c_batch +
        tl.arange(0, BLOCK_M)[:, None] * stride_c_m +
        tl.arange(0, BLOCK_N)[None, :] * stride_c_n,
        c_tile,
        mask=(tl.arange(0, BLOCK_M)[:, None] < m) & 
              (tl.arange(0, BLOCK_N)[None, :] < n)
    )

def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D"
    assert A.shape[0] == B.shape[0], "Batch sizes must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Prepare output tensor
    C = torch.empty(batch_size, m, n, device=A.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate strides
    stride_a_batch, stride_a_m, stride_a_k = A.stride()
    stride_b_batch, stride_b_k, stride_b_n = B.stride()
    stride_c_batch, stride_c_m, stride_c_n = C.stride()
    
    # Grid configuration
    grid = (
        batch_size,
        (n + BLOCK_N - 1) // BLOCK_N,
    )
    
    # Launch kernel
    batch_matmul_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_b_batch, stride_b_k, stride_b_n,
        stride_c_batch, stride_c_m, stride_c_n,
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M
    )
    
    return C

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized using custom Triton kernels.
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