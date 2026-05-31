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

    # Block offsets
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N

    # Create pointers for the blocks of A and B
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = offset_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = offset_n + tl.arange(0, BLOCK_SIZE_N)
    
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(N, K),
        strides=(stride_bk, stride_bn),
        offsets=(offset_n, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0)
    )
    
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b.T)
        
        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_SIZE_K))
    
    # Convert to output type and store
    C_block = accumulator.to(C_ptr.dtype.element_ty)
    tl.store(C_block_ptr, C_block)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B^T using Triton.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K) - note B.T will be used for computation
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported."
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, _ = B.shape
    
    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B^T) using Triton kernel.
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