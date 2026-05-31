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
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    am_mask = off_m[:, None] < M
    bn_mask = off_n[None, :] < N
    bk_mask = off_k[None, :] < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A block
        a_offset = off_m[:, None] * stride_am + (off_k[None, :] + k * BLOCK_SIZE_K) * stride_ak
        a = tl.load(A + a_offset, mask=am_mask & ((off_k[None, :] + k * BLOCK_SIZE_K) < K)[None, :], other=0.0)
        
        # Load B block (note: B is transposed in the original, so we use B directly)
        b_offset = (off_k[:, None] + k * BLOCK_SIZE_K) * stride_bk + off_n[None, :] * stride_bn
        b = tl.load(B + b_offset, mask=bk_mask & ((off_k[:, None] + k * BLOCK_SIZE_K) < K)[:, None] & bn_mask, other=0.0)
        
        # Accumulate
        acc = tl.dot(a, b, acc)
    
    # Cast to output type and store
    acc = acc.to(C.dtype.element_ty)
    c_offset = off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    c_mask = am_mask & bn_mask
    tl.store(C + c_offset, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B.T using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K) (note: will be used as B.T internally)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[1], "A and B must have same K dimension"
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, K_b = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tuned for FP32 on modern GPUs)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = math.ceil(M / BLOCK_SIZE_M)
    num_pid_n = math.ceil(N / BLOCK_SIZE_N)
    grid = (num_pid_m * num_pid_n,)
    
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
        Performs matrix multiplication using Triton kernel: C = A @ B.T
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)