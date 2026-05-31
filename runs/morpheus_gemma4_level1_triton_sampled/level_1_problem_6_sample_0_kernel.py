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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        
        # Load a block of A: shape (BLOCK_M, BLOCK_K)
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + (k_offset + rk[None, :]) * stride_ak),
            mask=(rm[:, None] < M) & ((k_offset + rk[None, :]) < K),
            other=0.0
        )
        # Load a block of B: shape (BLOCK_K, BLOCK_N)
        b = tl.load(
            b_ptr + ((k_offset + rk[:, None]) * stride_bk + rn[None, :] * stride_bn),
            mask=((k_offset + rk[:, None]) < K) & (rn[None, :] < N),
            other=0.0
        )
        
        # Perform matrix multiplication for the block
        acc += tl.dot(a, b)

    # Store the final result block to C
    tl.store(
        c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton wrapper for matrix multiplication.
    Optimized for FP32 and handles large K dimensions.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous to simplify indexing
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions of matrices must match"
    
    # Prepare the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # Tiling parameters
    # Small M, N (256) and very large K allows for flexible block sizes.
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Grid consists of blocks covering the M and N dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch the Triton kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B)
    using a custom Triton kernel for speedups on large K dimensions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)