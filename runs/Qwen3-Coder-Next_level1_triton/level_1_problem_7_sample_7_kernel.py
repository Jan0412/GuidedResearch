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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    # Program IDs for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Split K dimension for larger K
    pid_k = tl.program_id(2)
    
    # Compute starting offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Create masks
    amask = offs_m[:, None] < M
    bnmask = offs_n[None, :] < N
    bkmask = offs_k[None, :] < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in chunks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute actual k offset
        actual_k = k * BLOCK_K + pid_k * BLOCK_K
        # Load A and B tiles
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (actual_k % K) * stride_ak, 
                   mask=amask & (actual_k % K < K)[None, :], other=0.0)
        b = tl.load(B_ptr + (actual_k % K) * stride_bk + offs_n[None, :] * stride_bn,
                   mask=bkmask & (actual_k % K < K)[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
    
    # Scale by split K factor if needed
    if SPLIT_K > 1:
        accumulator = accumulator / SPLIT_K
    
    # Write result
    c = accumulator.to(tl.float32)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c, mask=amask & bnmask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton kernel.
    Supports FP32 tensors.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix multiplication dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    SPLIT_K = 4  # Split K dimension for better parallelism
    
    # Compute grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_k = SPLIT_K
    
    # Launch kernel
    matmul_kernel[grid_m, grid_n, grid_k](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K
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
        Performs optimized matrix multiplication using Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)