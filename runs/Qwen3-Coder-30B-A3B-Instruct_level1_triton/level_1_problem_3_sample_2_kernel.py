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
    
    # Create accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k_start in range(0, k, BLOCK_K):
        # Load tiles from memory
        a_tile = tl.load(
            A_ptr + 
            batch_idx * stride_a_batch + 
            m_start * stride_a_m + 
            k_start * stride_a_k,
            mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < m) & 
                  (k_start + tl.arange(0, BLOCK_K)[None, :] < k),
            other=0.0
        )
        
        b_tile = tl.load(
            B_ptr + 
            batch_idx * stride_b_batch + 
            k_start * stride_b_k + 
            n_start * stride_b_n,
            mask=(k_start + tl.arange(0, BLOCK_K)[:, None] < k) & 
                  (n_start + tl.arange(0, BLOCK_N)[None, :] < n),
            other=0.0
        )
        
        # Perform matrix multiplication for this tile
        acc += tl.dot(a_tile, b_tile)
    
    # Write back the result
    c_tile = acc
    tl.store(
        C_ptr + 
        batch_idx * stride_c_batch + 
        m_start * stride_c_m + 
        n_start * stride_c_n,
        c_tile,
        mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < m) & 
              (n_start + tl.arange(0, BLOCK_N)[None, :] < n)
    )

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using Triton kernel.
        """
        batch_size, m, k = A.shape
        _, _, n = B.shape
        
        # Ensure inputs are contiguous
        A = A.contiguous()
        B = B.contiguous()
        
        # Create output tensor
        C = torch.empty(batch_size, m, n, device=A.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        # Calculate strides
        stride_a_batch, stride_a_m, stride_a_k = A.stride()
        stride_b_batch, stride_b_k, stride_b_n = B.stride()
        stride_c_batch, stride_c_m, stride_c_n = C.stride()
        
        # Launch kernel
        grid = (
            batch_size,
            triton.cdiv(m, BLOCK_M),
            triton.cdiv(n, BLOCK_N)
        )
        
        bmm_kernel[grid](
            A, B, C,
            batch_size, m, n, k,
            stride_a_batch, stride_a_m, stride_a_k,
            stride_b_batch, stride_b_k, stride_b_n,
            stride_c_batch, stride_c_m, stride_c_n,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        
        return C