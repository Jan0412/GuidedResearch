import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the program ID and determine which tile to compute
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute the start indices for this tile
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid elements
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_k = offs_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from global memory
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                    mask=(mask_am[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                    mask=(mask_k[:, None] & mask_bn[None, :]), other=0.0)
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
    
    # Write back the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c, acc, mask=mask_c)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using Triton kernel.
    """
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.size(1) == b.size(0), "Matrix dimensions incompatible for multiplication"
    
    # Ensure tensors are contiguous and on CUDA
    a = a.contiguous().cuda()
    b = b.contiguous().cuda()
    
    # Prepare output tensor
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_warps = 4
    num_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    grid = lambda meta: (num_blocks,)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        num_warps=num_warps
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)