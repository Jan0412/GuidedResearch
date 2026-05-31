import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Map program IDs to the output matrix blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_ak = tl.arange(0, BLOCK_K)

    # Pointers to the first block of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_ak[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks of A and B
        # Masking is used to handle cases where M, N, or K are not multiples of block sizes
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & ((k * BLOCK_K + offs_ak[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k * BLOCK_K + offs_ak[:, None]) < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication for the block
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block in K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store the result in the output matrix
    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    tl.store(out_ptrs, accumulator, mask=(offs_om[:, None] < M) & (offs_on[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)