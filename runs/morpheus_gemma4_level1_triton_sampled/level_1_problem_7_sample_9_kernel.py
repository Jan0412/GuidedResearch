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
):
    # Map program IDs to the block of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the blocks
    offs_am = (pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (K, N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # We shift the pointers by k * BLOCK_SIZE_K * stride
        a = tl.load(a_ptrs + k * BLOCK_SIZE_K * stride_ak, 
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_SIZE_K < K), 
                    other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_SIZE_K * stride_bk, 
                    mask=(offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_bn[None, :] < N), 
                    other=0.0)
        
        # Perform matrix multiplication for the block
        accumulator += tl.dot(a, b)

    # Store the result in C
    offs_cm = offs_am[:, None]
    offs_cn = offs_bn[None, :]
    c_ptrs = c_ptr + (offs_cm * stride_cm + offs_cn * stride_cn)
    mask = (offs_cm < M) & (offs_cn < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on CUDA
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    # Since K=64 is small, we can process it in few blocks.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32 # Must be power of 2 and >= 16 for tl.dot

    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
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
        Performs matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # The original model uses torch.matmul(A, B).
        # We replace it with our custom triton_matmul implementation.
        return triton_matmul(A, B)