import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Get the row and column indices for this thread block
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Calculate starting positions for this tile
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k_start in range(0, k, BLOCK_K):
        # Load tiles from memory
        a_tile = tl.load(
            A_ptr +
            batch_idx * stride_a_batch +
            m_start * stride_a_m +
            k_start * stride_a_k,
            mask=(tl.arange(0, BLOCK_M)[:, None] < m - m_start) &
                  (tl.arange(0, BLOCK_K)[None, :] < k - k_start),
            other=0.0
        )
        
        b_tile = tl.load(
            B_ptr +
            batch_idx * stride_b_batch +
            k_start * stride_b_k +
            n_start * stride_b_n,
            mask=(tl.arange(0, BLOCK_K)[:, None] < k - k_start) &
                  (tl.arange(0, BLOCK_N)[None, :] < n - n_start),
            other=0.0
        )
        
        # Perform matrix multiplication for this tile
        acc += tl.dot(a_tile, b_tile)
    
    # Write back the result
    c_tile = acc.to(tl.float32)
    tl.store(
        C_ptr +
        batch_idx * stride_c_batch +
        m_start * stride_c_m +
        n_start * stride_c_n,
        c_tile,
        mask=(tl.arange(0, BLOCK_M)[:, None] < m - m_start) &
              (tl.arange(0, BLOCK_N)[None, :] < n - n_start)
    )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D"
    assert A.size(0) == B.size(0), "Batch dimensions must match"
    assert A.size(2) == B.size(1), "Inner dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Prepare output tensor
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
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N)
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_b_batch, stride_b_k, stride_b_n,
        stride_c_batch, stride_c_m, stride_c_n,
        BLOCK_M, BLOCK_N, BLOCK_K
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