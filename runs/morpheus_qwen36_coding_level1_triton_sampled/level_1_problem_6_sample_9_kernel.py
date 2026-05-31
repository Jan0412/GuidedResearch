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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for M and N dimensions
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    
    # Base pointers for A and B tiles
    A_ptrs = A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    B_ptrs = B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    
    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles from shared memory (Triton handles caching automatically)
        A_tile = tl.load(A_ptrs)
        B_tile = tl.load(B_ptrs)
        
        # Perform matrix multiplication and accumulate
        accumulator = tl.dot(A_tile, B_tile, accumulator)
        
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    # Store the result
    C_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(C_ptrs, accumulator)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    
    assert K == K2, "Inner dimensions must match."
    
    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Tunable block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128
    
    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Strides for row-major tensors
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


def get_inputs():
    M = 256
    N = 256
    K = 131072 * 4
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]


def get_init_inputs():
    return []