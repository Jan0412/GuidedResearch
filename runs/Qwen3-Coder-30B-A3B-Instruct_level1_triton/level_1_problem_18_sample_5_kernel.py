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
    ACTIVATION: tl.constexpr,
):
    # Map program ID to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B tiles.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Compute dot product.
        accumulator += tl.dot(a, b)
        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function if specified.
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # Write back result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_torch(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using Triton kernel.
    """
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.shape[1] == b.shape[0], "Matrix dimensions incompatible for multiplication"
    
    # Transpose inputs to match expected format (we're computing A^T @ B^T)
    # So we actually compute (B^T)^T @ (A^T)^T = B @ A
    # But we want A^T @ B^T which means we compute the transpose of this
    # Let's directly handle the transposed operation properly
    
    # For A.T @ B.T where A is (K,M) and B is (N,K)
    # This becomes (M,K) @ (K,N) = (M,N)
    # We'll compute it as (K,M)^T @ (K,N)^T = (M,K) @ (N,K) 
    # Which equals (M,N) but with different indexing
    
    # Actually, let's compute it correctly: 
    # If A is (K,M) and B is (N,K) then A^T is (M,K) and B^T is (K,N)
    # So A^T @ B^T = (M,K) @ (K,N) = (M,N)
    
    # Our kernel expects row-major matrices, so we work with:
    # A^T (M,K) and B^T (K,N) and compute (M,K) @ (K,N)
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Determine grid size
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
        ACTIVATION=None
    )
    
    return c

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Note: This computes A^T @ B^T, which is equivalent to (B @ A)^T
        # But since A is (K,M) and B is (N,K), we compute the transpose manually
        # We can also do it directly with proper indexing
        return triton_matmul_torch(A.T, B.T)