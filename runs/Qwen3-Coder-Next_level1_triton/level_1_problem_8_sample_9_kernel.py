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
    # Matrix multiplication kernel optimized for FP32
    # Each program handles one block of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the row and column of the output tile
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    am_mask = rm < M
    bn_mask = rn < N
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute the k offset
        k_offset = k * BLOCK_SIZE_K
        rk = k_offset + tl.arange(0, BLOCK_SIZE_K)
        rk_mask = rk < K
        
        # Load tiles from A and B
        a_tile = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
                         mask=rk_mask[None, :] & am_mask[:, None], other=0.0)
        b_tile = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn, 
                         mask=rk_mask[:, None] & bn_mask[None, :], other=0.0)
        
        # Perform matrix multiplication and accumulate
        acc = tl.dot(a_tile, b_tile, acc)
    
    # Store the result to C
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, 
             mask=am_mask[:, None] & bn_mask[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using Triton kernel optimized for FP32 precision.
    
    Args:
        A: Input tensor with shape (M, K)
        B: Input tensor with shape (K, N)
        
    Returns:
        C: Output tensor with shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes - tuned for FP32 performance on modern GPUs
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1
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
    Optimized version of the Model class that uses a custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a custom Triton kernel.
        
        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).
            
        Returns:
            C: Output tensor with shape (M, N).
        """
        return triton_matmul(A, B)