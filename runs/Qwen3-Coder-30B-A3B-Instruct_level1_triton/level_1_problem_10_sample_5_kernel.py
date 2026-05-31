import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix
    b_ptr,  # Pointer to second input matrix
    c_ptr,  # Pointer to output matrix
    M,      # Number of rows in A and C
    N,      # Number of columns in B and C
    K,      # Number of columns in A and rows in B
    stride_am, stride_ak,  # Stride of matrix A
    stride_bk, stride_bn,  # Stride of matrix B
    stride_cm, stride_cn,  # Stride of matrix C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Identify the block ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offset pointers for batched computation
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load A and B tiles
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute partial product
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton implementation of 3D tensor-matrix multiplication.
    """
    assert a.dim() == 3 and b.dim() == 2, "Expected A to be 3D and B to be 2D"
    assert a.shape[2] == b.shape[0], "Matrix dimensions incompatible for multiplication"
    
    # Ensure inputs are contiguous and on CUDA
    a = a.contiguous().cuda()
    b = b.contiguous().cuda()
    
    # Prepare output tensor
    M, K = a.shape[1], a.shape[2]
    L = b.shape[1]
    c = torch.empty(a.shape[0], M, L, dtype=torch.float32, device=a.device)
    
    # Define kernel parameters
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_M = 8
    
    # Calculate grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(L, meta["BLOCK_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        a.shape[0], M, L, K,  # M, N, K
        a.stride(0), a.stride(2),  # stride_am, stride_ak
        b.stride(0), b.stride(1),  # stride_bk, stride_bn
        c.stride(0), c.stride(2),  # stride_cm, stride_cn
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul(A, B)