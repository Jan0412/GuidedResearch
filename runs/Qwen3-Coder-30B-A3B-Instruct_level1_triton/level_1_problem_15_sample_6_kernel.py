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
    
    # Offset pointers for batching
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
    mask_b = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        
    # Compute output indices
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Apply lower triangular mask
    mask_triangular = offs_cm[:, None] >= offs_cn[None, :]
    mask_c = mask_c & mask_triangular
    
    # Store result
    c = accumulator.to(tl.float32)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=mask_c)

def triton_matmul_tril(a: torch.Tensor, b: torch.Tensor):
    """
    Matrix multiplication of two lower triangular matrices with tril applied
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.size(1) == b.size(0), "Matrix dimensions incompatible for multiplication"
    
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
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
    Optimized model using Triton kernels for matrix multiplication of lower triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        Uses a fused Triton kernel for both matmul and tril operations.
        
        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_matmul_tril(A, B)