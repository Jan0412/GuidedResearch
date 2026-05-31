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
    """
    Triton kernel for matrix multiplication C = A @ B.
    Uses blocked approach with tl.dot for efficient computation.
    """
    # Program IDs for block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create row and column offsets for the output block
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    
    # Masks to handle boundary conditions
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    
    # Accumulator for partial dot products
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load block from A
        # A is (M, K), so we compute linear offsets for the block
        a_ptrs = A_ptr + offsets_m[:, None] * stride_am + (k + offsets_k)[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load block from B
        # B is (K, N)
        b_ptrs = B_ptr + (k + offsets_k)[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Compute dot product and accumulate
        acc += tl.dot(a, b)
        
    # Store the result block to C
    c_ptrs = C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Shape mismatch: A is {A.shape}, B is {B.shape}"
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Block sizes (tunable parameters)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Calculate grid dimensions
    num_block_m = (M + BLOCK_M - 1) // BLOCK_M
    num_block_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Strides for row-major tensors
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1
    
    # Launch kernel
    matmul_kernel[(num_block_m, num_block_n)](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        """
        return triton_matmul(A, B)