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
    # Get the program ID and group ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offset pointers for batch dimension
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + 
            offs_k[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k[:, None] * stride_bk + 
            offs_bn[None, :] * stride_bn
        )
        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute partial matmul
        acc += tl.dot(a, b)
    
    # Write back result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + 
        offs_cn[None, :] * stride_cn
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a, b):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    """
    assert a.dim() == 3 and b.dim() == 2, "Expected 3D tensor A and 2D tensor B"
    assert a.shape[2] == b.shape[0], "Matrix dimensions must be compatible"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Prepare output tensor
    N, M, K = a.shape
    L = b.shape[1]
    c = torch.empty((N, M, L), dtype=torch.float32, device=a.device)
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(L, META["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        N * M, L, K,
        a.stride(0), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    
    return c

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using optimized Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul(A, B)