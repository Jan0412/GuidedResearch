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
    GROUP_SIZE_M: tl.constexpr
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the row and column of the output matrix
    # Each block computes a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of C
    # The tiles are computed in a way that maximizes data reuse of A and B
    
    # Create offset arrays for rows (M) and columns (N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to ensure indices are within bounds
    rm = tl.max_contiguous(tl.multiple_of(rm, 1), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, 1), BLOCK_SIZE_N)
    rm_mask = (rm < M)[:, None]
    rn_mask = (rn < N)[None, :]
    
    # Create a pointer offset for A and B to point to the correct tile
    # A is accessed as [M, K] and B as [K, N]
    # For each tile in C, we need to compute the dot product over K dimension
    # We loop over K in chunks of BLOCK_SIZE_K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offset for A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        ak = k + tl.arange(0, BLOCK_SIZE_K)
        ak = tl.max_contiguous(tl.multiple_of(ak, 1), BLOCK_SIZE_K)
        a_ptr = A_ptr + (rm[:, None] * stride_am + ak[None, :] * stride_ak)
        a = tl.load(a_ptr, mask=rm_mask, other=0.0)
        
        # Compute offset for B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        bk = k + tl.arange(0, BLOCK_SIZE_K)
        bk = tl.max_contiguous(tl.multiple_of(bk, 1), BLOCK_SIZE_K)
        b_ptr = B_ptr + (bk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b = tl.load(b_ptr, mask=rn_mask, other=0.0)
        
        # Accumulate the matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store the result to C
    c = accumulator.to(tl.float32)  # Ensure output is float32
    c_ptr = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = rm_mask & rn_mask
    tl.store(c_ptr, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (K, N).
    
    Returns:
        Output tensor of shape (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    
    M, K = A.shape
    K, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Set block sizes for the kernel (tunable parameters for performance)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1
    )
    
    # Launch the kernel
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
    Optimized model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)