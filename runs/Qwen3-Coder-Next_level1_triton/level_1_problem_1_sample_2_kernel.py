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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program IDs
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    
    # Group ID and program IDs within group
    group_id = pid // num_programs_in_group
    first_program_m = group_id * GROUP_SIZE_M
    program_ids_m = tl.arange(0, GROUP_SIZE_M)
    program_ids_n = tl.arange(0, num_programs_n)
    
    # Compute actual program IDs
    program_ids_m = tl.where(program_ids_m < num_programs_m - first_program_m, program_ids_m + first_program_m, num_programs_m)
    pid_m = tl.max(program_ids_m)
    pid_n = tl.max(program_ids_n)
    
    # Create block offsets
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    rm_mask = rm < M
    rn_mask = rn < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A block
        a_ptrs = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        a_mask = (rm_mask[:, None] & (rk < K)[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block
        b_ptrs = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_mask = ((rk < K)[:, None] & rn_mask[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiply
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to output type and store
    c = accumulator.to(tl.float32)
    
    c_ptrs = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm_mask[:, None] & rn_mask[None, :])
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    Assumes both inputs are square matrices of same size (N, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have same shape."
    assert A.shape[0] == A.shape[1], "Only square matrices supported."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, N = A.shape
    K = A.shape[1]  # Should equal N for square matrix
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Define block sizes for FP32 (tuned for modern GPUs)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 16  # Use smaller block size for better occupancy with FP32
    
    # Calculate grid dimensions
    num_programs_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = triton.cdiv(N, BLOCK_SIZE_N)
    GROUP_SIZE_M = 8  # Group size for better cache utilization
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
    Optimized version of the model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return matmul_triton(A, B)