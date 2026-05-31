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
    # Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix_multiplication.py
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    remaining = pid % num_pid_in_group
    target_m = first_pid_m + (remaining % GROUP_SIZE_M)
    target_n = (remaining // GROUP_SIZE_M) * BLOCK_SIZE_N
    
    # Offset pointers for batched matrix multiplication
    offs_am = target_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = target_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_k = offs_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B tiles
        a = tl.load(
            a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(mask_am[:, None] & mask_k[None, :]),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn),
            mask=(mask_k[:, None] & mask_bn[None, :]),
            other=0.0
        )
        
        # Matrix multiply
        acc += tl.dot(a, b)
    
    # Apply activation if specified
    if ACTIVATION == "leaky_relu":
        acc = tl.leaky_relu(acc)
    
    # Write results
    offs_cm = target_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = target_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = acc.to(tl.float32)
    c_ptr += (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr, c, mask=mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.shape[1] == b.shape[0], f"Matrix shapes {a.shape} and {b.shape} are incompatible for multiplication"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Calculate output shape
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
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
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M,
        "leaky_relu"  # Not actually used in this case but kept for compatibility
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
        Performs matrix-vector multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)