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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Get the tile index for this program
    tile_idx = tl.program_id(1)
    
    # Calculate the starting positions for this tile
    m_start = (tile_idx // (n // BLOCK_N)) * BLOCK_M
    n_start = (tile_idx % (n // BLOCK_N)) * BLOCK_N
    
    # Create pointers for matrices A and B
    a_batch_ptr = a_ptr + batch_idx * stride_a_batch
    b_batch_ptr = b_ptr + batch_idx * stride_b_batch
    
    # Create pointers for the output matrix C
    c_batch_ptr = c_ptr + batch_idx * stride_c_batch
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, k, BLOCK_K):
        # Load tiles from memory
        a_tile = tl.load(tl.make_block_ptr(
            a_batch_ptr, 
            shape=(m, k), 
            strides=(stride_a_m, stride_a_k),
            offsets=(m_start, k_start),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        ))
        
        b_tile = tl.load(tl.make_block_ptr(
            b_batch_ptr, 
            shape=(k, n), 
            strides=(stride_b_k, stride_b_n),
            offsets=(k_start, n_start),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0)
        ))
        
        # Perform matrix multiplication
        acc += tl.dot(a_tile, b_tile)
    
    # Write back result
    c_tile = tl.make_block_ptr(
        c_batch_ptr,
        shape=(m, n),
        strides=(stride_c_m, stride_c_n),
        offsets=(m_start, n_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    tl.store(c_tile, acc)

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Batched matrix multiplication using Triton kernel
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty(batch_size, m, n, device=A.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (m + BLOCK_M - 1) // BLOCK_M * (n + BLOCK_N - 1) // BLOCK_N
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernel for batched matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)