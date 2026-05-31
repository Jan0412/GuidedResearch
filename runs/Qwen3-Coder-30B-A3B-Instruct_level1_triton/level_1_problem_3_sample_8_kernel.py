import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, m, n, k,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Get the block index
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k_block in range(0, tl.cdiv(k, BLOCK_K)):
        # Load tiles from global memory
        a_tile = tl.load(
            a_ptr + 
            batch_idx * stride_a_batch +
            m_block * BLOCK_M * stride_a_m +
            k_block * BLOCK_K * stride_a_k +
            tl.arange(0, BLOCK_M)[:, None] * stride_a_m +
            tl.arange(0, BLOCK_K)[None, :] * stride_a_k,
            mask=(tl.arange(0, BLOCK_M)[:, None] < m - m_block * BLOCK_M) &
                  (tl.arange(0, BLOCK_K)[None, :] < k - k_block * BLOCK_K),
            other=0.0
        )
        
        b_tile = tl.load(
            b_ptr + 
            batch_idx * stride_b_batch +
            k_block * BLOCK_K * stride_b_k +
            n_block * BLOCK_N * stride_b_n +
            tl.arange(0, BLOCK_K)[:, None] * stride_b_k +
            tl.arange(0, BLOCK_N)[None, :] * stride_b_n,
            mask=(tl.arange(0, BLOCK_K)[:, None] < k - k_block * BLOCK_K) &
                  (tl.arange(0, BLOCK_N)[None, :] < n - n_block * BLOCK_N),
            other=0.0
        )
        
        # Matrix multiplication
        acc += tl.dot(a_tile, b_tile)
    
    # Write back to global memory
    c_tile = acc.to(tl.float32)
    tl.store(
        c_ptr + 
        batch_idx * stride_c_batch +
        m_block * BLOCK_M * stride_c_m +
        n_block * BLOCK_N * stride_c_n +
        tl.arange(0, BLOCK_M)[:, None] * stride_c_m +
        tl.arange(0, BLOCK_N)[None, :] * stride_c_n,
        c_tile,
        mask=(tl.arange(0, BLOCK_M)[:, None] < m - m_block * BLOCK_M) &
              (tl.arange(0, BLOCK_N)[None, :] < n - n_block * BLOCK_N)
    )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Triton implementation of batched matrix multiplication
    """
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D"
    assert A.shape[0] == B.shape[0], "Batch sizes must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate strides
    stride_a_batch, stride_a_m, stride_a_k = A.stride()
    stride_b_batch, stride_b_k, stride_b_n = B.stride()
    stride_c_batch, stride_c_m, stride_c_n = C.stride()
    
    # Grid dimensions
    grid = (
        batch_size,
        tl.cdiv(m, BLOCK_M),
        tl.cdiv(n, BLOCK_N)
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_b_batch, stride_b_k, stride_b_n,
        stride_c_batch, stride_c_m, stride_c_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return C

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized with custom Triton kernels.
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
        return triton_bmm(A, B)