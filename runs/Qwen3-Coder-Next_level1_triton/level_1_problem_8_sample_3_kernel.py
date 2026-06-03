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
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped program mapping
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    group_id = pid // num_programs_in_group
    first_program_m = group_id * GROUP_SIZE_M
    program_id_m = min(first_program_m + GROUP_SIZE_M, num_programs_m)
    program_id_n = (pid % num_programs_in_group) % num_programs_n
    
    # Block offsets
    block_m = program_id_m * BLOCK_SIZE_M
    block_n = program_id_n * BLOCK_SIZE_N
    
    # Create pointer arrays for the blocks
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_n),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1)
    )
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(block_m, block_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        A_block = tl.load(A_block_ptr)
        B_block = tl.load(B_block_ptr)
        
        # Matrix multiplication
        accumulator += tl.dot(A_block, B_block)
        
        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))
    
    # Store result
    tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor with shape (M, K).
        B: Input tensor with shape (K, N).
    
    Returns:
        C: Output tensor with shape (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions: A.shape[1] != B.shape[0]"
    
    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Set block sizes (tuned for FP32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
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
    Optimized model that performs a single matrix multiplication (C = A * B) using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized Triton kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return triton_matmul(A, B)