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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for rows (M) and columns (N)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_offsets = (offsets_m[:, None] * stride_am + 
                    (k + offsets_k)[None, :] * stride_ak)
        a_mask = (mask_m[:, None] & 
                 ((k + offsets_k)[None, :] < K))
        a = tl.load(A + a_offsets, mask=a_mask, other=0.0)
        
        # Load B block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_offsets = ((k + offsets_k)[:, None] * stride_bk + 
                    offsets_n[None, :] * stride_bn)
        b_mask = (((k + offsets_k)[:, None] < K) & 
                 mask_n[None, :])
        b = tl.load(B + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Cast accumulator to float16 if needed and store result
    c = accumulator.to(C.dtype.element_ty)
    
    # Store result
    c_offsets = (offsets_m[:, None] * stride_cm + 
                offsets_n[None, :] * stride_cn)
    c_mask = (mask_m[:, None] & mask_n[None, :])
    tl.store(C + c_offsets, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton kernel.
    Handles tall-and-skinny cases efficiently with tiling.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"
    
    # Allocate output
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes - optimized for tall-and-skinny case
    # Since M >> N, we use larger BLOCK_SIZE_M to process more rows at once
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
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
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    Specialized for tall-and-skinny matrix multiplications.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return triton_matmul(A, B)