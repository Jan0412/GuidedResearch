import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A @ B.
    Uses tiled approach with tl.dot for efficient computation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Base pointers for the current tile
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator for the tile
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B with masking for boundary conditions
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute partial dot product using tl.dot (optimized for tensor cores)
        accumulator += tl.dot(a, b)
        
        # Advance pointers for the next block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    # Store the result to C
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []