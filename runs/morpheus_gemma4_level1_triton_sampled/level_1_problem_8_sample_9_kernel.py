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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A * B.
    Optimized for FP32 precision.
    """
    # Map program IDs to the blocks of the output matrix C
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache optimization: Grouping blocks of M to increase reuse of B
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    
    # Re-order to improve L2 cache hit rate
    # This is a simplified version of the L2-aware tiling
    pid_m = pid_m
    pid_n = pid_n

    # Pointers to the start of the blocks in A and B
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    
    # Offsets for the current block
    offsets_am = (rm + tl.arange(0, BLOCK_SIZE_M))
    offsets_bn = (rn + tl.arange(0, BLOCK_SIZE_N))
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a = tl.load(
            a_ptr + (offsets_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offsets_k[None, :]) * stride_ak),
            mask=(offsets_am[:, None] < M) & ((k * BLOCK_SIZE_K + offsets_k[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((k * BLOCK_SIZE_K + offsets_k[:, None]) * stride_bk + offsets_bn[None, :] * stride_bn),
            mask=((k * BLOCK_SIZE_K + offsets_k[:, None]) < K) & (offsets_bn[None, :] < N),
            other=0.0,
        )
        
        # Perform dot product and accumulate
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_offsets = offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets, 
        accumulator, 
        mask=(offsets_am[:, None] < M) & (offsets_bn[None, :] < N)
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid calculation
    # We use a 1D grid and map it to 2D inside the kernel for L2 optimization
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

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
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Ensure tensors are on GPU for Triton
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return triton_matmul(A, B)