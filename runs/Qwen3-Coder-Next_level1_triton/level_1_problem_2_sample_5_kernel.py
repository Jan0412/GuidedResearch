import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID is 2D: one for M dimension and one for N dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask to handle boundary cases
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in chunks
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute current K offsets
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=mask_m[:, None],
            other=0.0
        )
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=mask_n[None, :],
            other=0.0
        )
        
        # Accumulate the matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to float16 if needed, or keep as float32
    c = accumulator.to(tl.float32)
    
    # Store result
    tl.store(
        C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based matrix multiplication kernel wrapper.
    Performs C = A @ B where A is (M, K) and B is (K, N).
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    
    # Check dimension compatibility
    assert K == K_b, f"Inner dimensions must match: A has K={K}, B has K={K_b}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes for the kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    
    # Set grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)