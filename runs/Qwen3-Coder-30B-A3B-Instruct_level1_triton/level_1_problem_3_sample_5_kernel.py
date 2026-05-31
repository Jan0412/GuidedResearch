import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, m, n, k,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the block index for m and n dimensions
    bm = tl.program_id(1)
    bn = tl.program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k_iter in range(0, tl.cdiv(k, BLOCK_K)):
        # Load A and B tiles
        a_tile = tl.load(
            a_ptr + 
            batch_idx * stride_a_batch +
            bm * BLOCK_M * stride_a_m + 
            k_iter * BLOCK_K * stride_a_k,
            mask=(k_iter * BLOCK_K + tl.arange(0, BLOCK_K) < k)[None, :],
            other=0.0
        )
        
        b_tile = tl.load(
            b_ptr + 
            batch_idx * stride_b_batch +
            k_iter * BLOCK_K * stride_b_k + 
            bn * BLOCK_N * stride_b_n,
            mask=(k_iter * BLOCK_K + tl.arange(0, BLOCK_K) < k)[:, None],
            other=0.0
        )
        
        # Perform matrix multiplication for this tile
        acc += tl.dot(a_tile, b_tile)
    
    # Store the result
    c_tile = acc.to(tl.float32)
    tl.store(
        c_ptr + 
        batch_idx * stride_c_batch +
        bm * BLOCK_M * stride_c_m + 
        bn * BLOCK_N * stride_c_n,
        c_tile
    )

def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D."
    assert A.size(0) == B.size(0), "Batch sizes must match."
    assert A.size(2) == B.size(1), "Inner dimensions must match."
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Prepare output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device=A.device)
    
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
        tl.cdiv(m, BLOCK_M),
        tl.cdiv(n, BLOCK_N)
    )
    
    # Launch the kernel
    batch_matmul_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_b_batch, stride_b_k, stride_b_n,
        stride_c_batch, stride_c_m, stride_c_n,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
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
        return triton_batch_matmul(A, B)