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
    # Get the batch index for this program
    batch_idx = tl.program_id(0)
    # Get the group index for this program
    group_idx = tl.program_id(1)
    
    # Each group processes a subset of rows
    # Calculate starting row for this group
    start_m = group_idx * BLOCK_M
    
    # Load A and B tiles
    offs_am = start_m + tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for the first blocks of A and B
    a_ptrs = a_ptr + (
        batch_idx * stride_a_batch +
        offs_am[:, None] * stride_a_m +
        offs_k[None, :] * stride_a_k
    )
    b_ptrs = b_ptr + (
        batch_idx * stride_b_batch +
        offs_k[:, None] * stride_b_k +
        offs_bn[None, :] * stride_b_n
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, k, BLOCK_K):
        # Load tiles
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (offs_k[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k) & (offs_bn[None, :] < n), other=0.0)
        
        # Matrix multiply
        acc += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k
    
    # Write result
    offs_cm = start_m + tl.arange(0, BLOCK_M)
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (
        batch_idx * stride_c_batch +
        offs_cm[:, None] * stride_c_m +
        offs_cn[None, :] * stride_c_n
    )
    
    # Store result
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < m) & (offs_cn[None, :] < n))

def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton implementation of batched matrix multiplication.
    """
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
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
    
    # Calculate grid
    grid = (
        batch_size,  # Batch dimension
        (m + BLOCK_M - 1) // BLOCK_M,  # Group dimension
    )
    
    # Launch kernel
    batch_matmul_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_batch_matmul(A, B)