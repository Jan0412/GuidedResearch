import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs = num_programs_m * num_programs_n
    
    # Group programs for better cache usage
    group_size = min(GROUP_SIZE_M, num_programs_m)
    group_id = pid // group_size
    group_size = min(num_programs_m - group_id * group_size, group_size)
    pid_m = group_id * group_size + (pid % group_size)
    pid_n = (pid % num_programs) // group_size
    
    # Offsets for blocks
    offsets_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    a_mask = offsets_am[:, None] < M
    b_mask = offsets_bn[None, :] < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Get current K offsets
        k_offsets = k * BLOCK_SIZE_K + offsets_k
        k_mask = k_offsets < K
        
        # Load A block
        a_ptrs = A + (offsets_am[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        a_block = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0)
        
        # Load B block
        b_ptrs = B + (k_offsets[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
        b_block = tl.load(b_ptrs, mask=k_mask[:, None] & b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    # Convert to output type and store
    c_block = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + (offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn)
    c_mask = a_mask & b_mask
    tl.store(c_ptrs, c_block, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (K, N).
        
    Returns:
        Output tensor of shape (M, N).
    """
    # Ensure tensors are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    
    # Check dimension compatibility
    assert K == K_b, f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
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
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)