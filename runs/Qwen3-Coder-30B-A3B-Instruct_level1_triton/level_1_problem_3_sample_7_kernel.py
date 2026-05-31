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
    # Get the tile index for this program within the batch
    tile_idx = tl.program_id(1)
    
    # Compute the starting positions for this tile
    pid_m = tile_idx // (n // BLOCK_N)
    pid_n = tile_idx % (n // BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_iter in range(0, k, BLOCK_K):
        # Load A and B tiles
        a_ptrs = A_ptr + batch_idx * stride_a_batch + pid_m * BLOCK_M * stride_a_m + k_iter * stride_a_k
        b_ptrs = B_ptr + batch_idx * stride_b_batch + k_iter * stride_b_k + pid_n * BLOCK_N * stride_b_n
        
        # Create masks for boundary conditions
        a_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < m) & \
                 (k_iter + tl.arange(0, BLOCK_K)[None, :] < k)
        b_mask = (k_iter + tl.arange(0, BLOCK_K)[:, None] < k) & \
                 (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < n)
        
        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply
        acc += tl.dot(a, b)
    
    # Compute output pointer
    c_ptr = C_ptr + batch_idx * stride_c_batch + pid_m * BLOCK_M * stride_c_m + pid_n * BLOCK_N * stride_c_n
    
    # Store result
    c_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < m) & \
             (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < n)
    tl.store(c_ptr, acc, mask=c_mask)

def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
        
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.dim() == 3 and B.dim() == 3, "A and B must be 3D tensors"
    assert A.size(0) == B.size(0), "Batch sizes must match"
    assert A.size(2) == B.size(1), "Inner dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    # Allocate output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device='cuda')
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate strides
    stride_a_batch, stride_a_m, stride_a_k = A.stride()
    stride_b_batch, stride_b_k, stride_b_n = B.stride()
    stride_c_batch, stride_c_m, stride_c_n = C.stride()
    
    # Grid dimensions
    grid = (
        batch_size,
        (m // BLOCK_M) * (n // BLOCK_N)
    )
    
    # Launch kernel
    batch_matmul_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_b_batch, stride_b_k, stride_b_n,
        stride_c_batch, stride_c_m, stride_c_n,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
        num_warps=4
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