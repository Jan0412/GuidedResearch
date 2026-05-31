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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offset ranges for blocks
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    am_mask = offsets_am[:, None] < M
    bn_mask = offsets_bn[None, :] < N
    bk_mask = offsets_k[None, :] < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A tile: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offset = (offsets_am[:, None] * stride_am + 
                   (k * BLOCK_SIZE_K + offsets_k)[None, :] * stride_ak)
        a = tl.load(A_ptr + a_offset, mask=am_mask & (offsets_k[None, :] < K), other=0.0)
        
        # Load B tile: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offset = ((k * BLOCK_SIZE_K + offsets_k)[:, None] * stride_bk + 
                   offsets_bn[None, :] * stride_bn)
        b = tl.load(B_ptr + b_offset, mask=bk_mask[:, None] & bn_mask, other=0.0)
        
        # Matrix multiply
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert to output dtype and store result
    c = accumulator.to(C_ptr.dtype.element_ty)
    
    # Store the result
    c_offset = (offsets_am[:, None] * stride_cm + 
               offsets_bn[None, :] * stride_cn)
    c_mask = am_mask & bn_mask
    tl.store(C_ptr + c_offset, c, mask=c_mask)


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A.T @ B.T using Triton kernel.
    This is equivalent to torch.matmul(A.T, B.T)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Dimensions
    K, M = A.shape
    N, K2 = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication C = A.T @ B.T using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transposed(A, B)