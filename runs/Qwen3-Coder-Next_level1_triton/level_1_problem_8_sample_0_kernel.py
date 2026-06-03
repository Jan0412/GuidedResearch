import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride parameters
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and relative position in group
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    am_mask = offsets_am[:, None] < M
    bn_mask = offsets_bn[None, :] < N
    bk_mask = offsets_k[None, :] < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A block
        a_offset = offsets_am[:, None] * stride_am + (offsets_k[None, :] + k * BLOCK_SIZE_K) * stride_ak
        a = tl.load(A_ptr + a_offset, mask=am_mask & (offsets_k[None, :] + k * BLOCK_SIZE_K < K)[None, :], other=0.0)
        
        # Load B block
        b_offset = (offsets_k[None, :] + k * BLOCK_SIZE_K) * stride_bk + offsets_bn[None, :] * stride_bn
        b = tl.load(B_ptr + b_offset, mask=bk_mask & (offsets_k[None, :] + k * BLOCK_SIZE_K < K)[:, None] & bn_mask, other=0.0)
        
        # Matrix multiply
        acc = tl.dot(a, b, acc)
    
    # Store result
    c = acc.to(tl.float32)
    c_offset = offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn
    c_mask = (offsets_am[:, None] < M) & (offsets_bn[None, :] < N)
    tl.store(C_ptr + c_offset, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor with shape (M, K)
        B: Input tensor with shape (K, N)
    
    Returns:
        C: Output tensor with shape (M, N)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_, N = B.shape
    assert K == K_, "Incompatible matrix dimensions"
    
    # Allocate output
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tuned for FP32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_groups = triton.cdiv(num_pid_m * num_pid_n, GROUP_SIZE_M * num_pid_n)
    
    # Launch kernel
    matmul_kernel[
        num_pid_m * num_pid_n,
    ](
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
        Performs matrix multiplication of A and B using Triton kernel.
        
        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).
        
        Returns:
            C: Output tensor with shape (M, N).
        """
        return triton_matmul(A, B)