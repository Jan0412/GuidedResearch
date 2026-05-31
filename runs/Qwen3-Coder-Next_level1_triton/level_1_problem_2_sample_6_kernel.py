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
    GROUP_SIZE_M: tl.constexpr
):
    # Matrix multiplication kernel for FP32 tensors
    # Program ID identifies the block of output matrix C
    pid = tl.program_id(0)
    
    # Number of programs in group
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    
    # Group ID and program ID within group
    group_id = pid // num_programs_in_group
    first_program_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_programs_m - first_program_m, GROUP_SIZE_M)
    pid_m = first_program_m + (pid % group_size_m)
    pid_n = (pid % num_programs_in_group) // group_size_m
    
    # Create offsets for M and N dimensions
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
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_offsets = (
            (offsets_am[:, None] * stride_am + 
             (k + offsets_k[None, :]) * stride_ak)
        )
        a = tl.load(A + a_offsets, mask=am_mask & (k + offsets_k[None, :] < K), other=0.0)
        
        # Load B block
        b_offsets = (
            ((k + offsets_k[:, None]) * stride_bk + 
             offsets_bn[None, :] * stride_bn)
        )
        b = tl.load(B + b_offsets, mask=bk_mask & (k + offsets_k[:, None] < K), other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to float32 (it's already float32 but ensure correct type)
    c = accumulator.to(tl.float32)
    
    # Store result
    c_offsets = (
        offsets_am[:, None] * stride_cm + 
        offsets_bn[None, :] * stride_cn
    )
    c_mask = am_mask & bn_mask
    tl.store(C + c_offsets, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication C = A * B
    """
    # Ensure tensors are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: A is {M}x{K}, B is {K2}x{N}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters for optimization)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
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
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel
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