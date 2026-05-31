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
    # Map program IDs to the corresponding block of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the current block of A and B
    # A is (M, K), B is (K, N)
    a_offsets = rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_offsets = rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # Initialize accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        
        # Load blocks of A and B with boundary masking
        a = tl.load(a_ptr + a_offsets + k_offset * stride_ak, 
                    mask=(rm[:, None] < M) & (rk[None, :] + k_offset < K), 
                    other=0.0)
        b = tl.load(b_ptr + b_offsets + k_offset * stride_bk, 
                    mask=(rk[:, None] + k_offset < K) & (rn[None, :] < N), 
                    other=0.0)
        
        # Perform matrix multiplication for the current block
        accumulator += tl.dot(a, b)

    # Store the final result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Prepare output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
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
    using a custom Triton kernel for speedups on large K dimensions.
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
        return triton_matmul(A, B)