import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from A and B
        # A is (M, K), B is (N, K)
        a_offsets = offs_am[:, None] * K + (k + offs_k)[None, :]
        b_offsets = offs_bn[None, :] * K + (k + offs_k)[:, None]
        
        mask_a = (offs_am[:, None] < M) & ((k + offs_k)[None, :] < K)
        mask_b = (offs_bn[None, :] < N) & ((k + offs_k)[:, None] < K)
        
        a_tile = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        b_tile = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # Perform dot product
        accumulator = tl.dot(a_tile, b_tile)

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offsets = offs_cm[:, None] * N + offs_cn[None, :]
    
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptr + c_offsets, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton matmul kernel.
    Assumes A and B are already transposed and contiguous:
      A shape: (M, K)
      B shape: (N, K)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "K dimensions must match."
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Determine block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B using a custom Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose and make contiguous for efficient kernel execution
        # A.T becomes (M, K), B.T becomes (N, K)
        A_T = A.T.contiguous()
        B_T = B.T.contiguous()
        
        # Compute C = A_T @ B_T.T which is equivalent to A.T @ B
        return triton_matmul(A_T, B_T)