import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Number of programs in the M dimension
    num_program_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs in the N dimension
    num_program_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group size for better cache utilization
    num_program_in_group = GROUP_SIZE_M * num_program_n
    group_id = pid // num_program_in_group
    first_program_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_program_m - first_program_m, GROUP_SIZE_M)
    program_id_m = first_program_m + (pid % group_size_m)
    program_id_n = (pid % num_program_in_group) // group_size_m
    
    # Block offsets
    block_start_m = program_id_m * BLOCK_SIZE_M
    block_start_n = program_id_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    masks_m = offsets_m < M
    masks_n = offsets_n < N
    masks_k = offsets_k < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_offsets = (
            (offsets_m[:, None] * stride_am + (k + offsets_k[None, :]) * stride_ak) *
            masks_m[:, None] * masks_k[None, :]
        )
        a = tl.load(A + a_offsets, mask=masks_m[:, None] * masks_k[None, :], other=0.0)
        
        # Load B block (note: B is transposed in the original operation, so we access B.T)
        b_offsets = (
            ((k + offsets_k[:, None]) * stride_bk + offsets_n[None, :] * stride_bn) *
            masks_k[:, None] * masks_n[None, :]
        )
        b = tl.load(B + b_offsets, mask=masks_k[:, None] * masks_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Cast accumulator to C's dtype and store result
    C_block_start_m = block_start_m
    C_block_start_n = block_start_n
    
    c_offsets = (
        (C_block_start_m + offsets_m)[:, None] * stride_cm +
        (C_block_start_n + offsets_n)[None, :] * stride_cn
    )
    masks_c = (offsets_m[:, None] < M) * (offsets_n[None, :] < N)
    
    tl.store(C + c_offsets, accumulator.to(C.dtype.element_ty), mask=masks_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B.T using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (N, K) (will be used as B.T in computation).
        
    Returns:
        Output tensor of shape (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, _ = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
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
    Optimized model that performs matrix multiplication using Triton kernel.
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