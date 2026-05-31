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
    # Program ID
    pid = tl.program_id(0)
    
    # Number of programs in each group
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and local program ID
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for rows and columns
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    masks_m = offsets_m < M
    masks_n = offsets_n < N
    masks_k = offsets_k < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            (offsets_m[:, None] * stride_am + 
             (k + offsets_k[None, :]) * stride_ak)
        )
        a = tl.load(A_ptr + a_offsets, mask=masks_m[:, None] & masks_k[None, :], other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (
            ((k + offsets_k[:, None]) * stride_bk + 
             offsets_n[None, :] * stride_bn)
        )
        b = tl.load(B_ptr + b_offsets, mask=masks_k[:, None] & masks_n[None, :], other=0.0)
        
        # Accumulate matrix multiply
        accumulator += tl.dot(a, b)
    
    # Cast accumulator to float16 if needed (but for FP32, keep as is)
    c = accumulator.to(tl.float32)
    
    # Store result
    c_offsets = (
        offsets_m[:, None] * stride_cm + 
        offsets_n[None, :] * stride_cn
    )
    tl.store(C_ptr + c_offsets, c, mask=masks_m[:, None] & masks_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B.T using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K)  # Note: we want B.T, so effectively (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_groups = triton.cdiv(num_pid_m * num_pid_n, num_pid_in_group)
    grid = (num_groups * num_pid_in_group,)
    
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
    Optimized model using Triton kernel for matrix multiplication.
    Performs C = A * B.T using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)