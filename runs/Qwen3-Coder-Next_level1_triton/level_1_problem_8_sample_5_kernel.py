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
    GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication kernel using tiled approach with blocked GEMM
    # Program IDs for blocks in M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute starting offsets for tiles
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    masks_am = offsets_am < M
    masks_bn = offsets_bn < N
    
    # Initialize accumulator for the tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute current K offset
        k_offset = k * BLOCK_SIZE_K
        
        # Load tile from A: shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_offsets = (
            (offsets_am[:, None] * stride_am + 
             (k_offset + offsets_k)[None, :] * stride_ak)
        )
        a_mask = (masks_am[:, None] & 
                  ((k_offset + offsets_k)[None, :] < K))
        a = tl.load(A + a_offsets, mask=a_mask, other=0.0)
        
        # Load tile from B: shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_offsets = (
            ((k_offset + offsets_k)[:, None] * stride_bk + 
             offsets_bn[None, :] * stride_bn)
        )
        b_mask = (((k_offset + offsets_k)[:, None] < K) & 
                  masks_bn[None, :])
        b = tl.load(B + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication for this tile
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to output type and store result
    c = accumulator.to(C.dtype.element_ty)
    
    # Store the result tile
    c_offsets = (
        offsets_am[:, None] * stride_cm + 
        offsets_bn[None, :] * stride_cn
    )
    c_mask = masks_am[:, None] & masks_bn[None, :]
    tl.store(C + c_offsets, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor with shape (M, K).
        B: Input tensor with shape (K, N).
        
    Returns:
        C: Output tensor with shape (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    
    # Extract dimensions
    M, K = A.shape
    K_b, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Configure block sizes (tuned for FP32)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
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
    Optimized model that performs matrix multiplication using Triton kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized Triton kernel.
        
        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).
            
        Returns:
            C: Output tensor with shape (M, N).
        """
        return triton_matmul(A, B)