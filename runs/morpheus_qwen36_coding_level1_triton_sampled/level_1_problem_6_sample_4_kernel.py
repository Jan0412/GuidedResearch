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
):
    # Program ID (single block covers the entire output due to small M, N)
    pid = tl.program_id(0)
    
    # Offset arrays for M and N dimensions
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Masks to handle boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator matrix
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension in tiles
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # Create pointers for current tiles
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # Load tiles with masking to avoid out-of-bounds access
        A_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        B_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Perform matrix multiplication and accumulate
        acc = tl.dot(A_tile, B_tile, acc)
        
    # Store the final result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match."
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Tune block sizes based on the problem dimensions
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    
    grid = (1,)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)