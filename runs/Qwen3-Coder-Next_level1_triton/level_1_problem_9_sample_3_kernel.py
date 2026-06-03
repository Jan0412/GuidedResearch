import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Number of programs in the M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Grouped implementation for better cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_m
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = (pid_n % num_pid_m) * GROUP_SIZE_M + (pid_m % GROUP_SIZE_M)
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure offsets are within bounds
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate offset in K dimension
        k_offset = k * BLOCK_SIZE_K
        
        # Create offsets for K dimension
        offsets_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
        # Load blocks from A and B
        # A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_offsets = offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & (offsets_k[None, :] < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_offsets = offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b_mask = (offsets_k[:, None] < K) & mask_n[None, :]
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Perform matrix multiplication
        acc = tl.dot(a, b, acc)
    
    # Convert accumulator to float16 if needed
    acc = acc.to(tl.float32)
    
    # Store result
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_matmul(A, B):
    """
    Triton-based matrix multiplication kernel wrapper.
    Supports tall and skinny matrices with optimized block sizes.
    """
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_B, N = B.shape
    
    # Verify dimension compatibility
    assert K == K_B, f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    # Use smaller block sizes for tall and skinny matrices
    # For M >> N: use more blocks in M dimension, fewer in N
    # For N >> M: use more blocks in N dimension, fewer in M
    if M > N:
        # Tall matrix case: optimize for M dimension
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    else:
        # Skinny matrix case: optimize for N dimension
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 16
    
    # Calculate grid dimensions
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
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    Optimized for tall and skinny matrices with appropriate block sizes.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.
            
        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return triton_matmul(A, B)