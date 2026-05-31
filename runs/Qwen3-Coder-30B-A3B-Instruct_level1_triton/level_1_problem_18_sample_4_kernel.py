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
    # Get the program ID and group information
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers to the start of each tile
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from global memory
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Create masks for valid indices
        mask_k = offs_k < K - k
        
        # Load tiles with proper masking
        a_tile = tl.load(a_ptrs, mask=(mask_am[:, None] & mask_k[None, :]), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_bn[None, :]), other=0.0)
        
        # Perform matrix multiplication for this tile
        acc += tl.dot(a_tile, b_tile)
    
    # Write back to global memory
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_am[:, None] & mask_bn[None, :]))

def triton_matmul_torch(a: torch.Tensor, b: torch.Tensor):
    """
    Triton-based matrix multiplication for A^T @ B^T
    """
    # Transpose inputs for the operation A^T @ B^T = (B @ A)^T
    a = a.T.contiguous()  # Now shape (M, K)
    b = b.T.contiguous()  # Now shape (N, K)
    
    # Ensure inputs are on CUDA and contiguous
    a = a.cuda().contiguous()
    b = b.cuda().contiguous()
    
    # Calculate output dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"
    
    # Allocate output tensor
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.
        Equivalent to torch.matmul(A.T, B.T)
        """
        return triton_matmul_torch(A, B)