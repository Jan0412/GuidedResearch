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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    amask = offs_m[:, None] < M
    bnmask = offs_n[None, :] < N
    bkmask = offs_k[None, :] < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak),
            mask=amask & (k + offs_k)[None, :] < K,
            other=0.0
        )
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=bkmask & (k + offs_k)[:, None] < K,
            other=0.0
        )
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to output type if needed
    c = accumulator.to(C_ptr.dtype.element_ty)
    
    # Store result
    tl.store(
        C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        c,
        mask=amask & bnmask
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B.T using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K) - note that B.T is used in computation
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"
    
    # Allocate output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
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