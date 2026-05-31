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
    
    # Number of programs in each dimension
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Reorder programs to improve cache locality (Grouped matrix multiplication)
    # Each program handles one tile of C, but we reorder them to process contiguous blocks
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute program indices
    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n
    
    # Create range for row tiles
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Create range for column tiles
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    rm = tl.where(rm < M, rm, M - 1)
    rn = tl.where(rn < N, rn, N - 1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offsets for A matrix (M x K)
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        rk = tl.where(rk < K, rk, K - 1)
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc = acc + tl.dot(a, b)
    
    # Convert accumulator to output type and store result
    acc = acc.to(tl.float32)
    
    # Store result to C matrix
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    Assumes square matrices for simplicity, but can handle rectangular ones too.
    """
    # Ensure tensors are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get matrix dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: A is {A.shape}, B is {B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel launch parameters
    # These are optimized values for FP32 on modern GPUs
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m * grid_n,)
    
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
    Optimized model that performs matrix multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
        
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)