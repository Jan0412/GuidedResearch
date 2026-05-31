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
    # Map program IDs to the corresponding block of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the blocks
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < N

    # Initialize pointers to the start of the blocks for A and B
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator for the dot product result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask for the K dimension to handle cases where K is not a multiple of BLOCK_K
        mask_k = rk < (K - k * BLOCK_K)
        
        # Load tiles from A and B
        a = tl.load(a_ptr, mask=mask_m & mask_k[None, :], other=0.0)
        b = tl.load(b_ptr, mask=mask_k[:, None] & mask_n, other=0.0)
        
        # Perform matrix multiplication of tiles
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next tiles along the K dimension
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    # Store the final result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr, accumulator, mask=mask_m & mask_n)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are on CUDA and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Get dimensions
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Hyperparameters for tiling
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel for speedups, specifically optimized for 
    large K dimensions and FP32 precision.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        # Ensure tensors are contiguous for the Triton kernel
        A = A.contiguous()
        B = B.contiguous()
        return triton_matmul(A, B)