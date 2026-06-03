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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Number of programs in the M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group M programs for better cache locality
    if GROUP_SIZE_M > 1:
        # Compute group ID and local IDs
        pid_m = pid_m % GROUP_SIZE_M
        pid_n = pid_n * GROUP_SIZE_M + (pid_m % GROUP_SIZE_M)
    
    # Create tile offsets for M and N dimensions
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offset_m < M
    mask_n = offset_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute K offset
        offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offset_k < K
        
        # Load A tile: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = A_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        
        # Load B tile: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
    
    # Cast accumulator to float16 for storage if needed, but keep as float32 for this task
    c = accumulator.to(tl.float32)
    
    # Store result
    c_ptrs = C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure tensors are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tuned for FP32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
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
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)