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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the start of the blocks for M and N dimensions
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for better performance
    if GROUP_SIZE_M > 1:
        # Grouping logic
        group_id = pid_m // GROUP_SIZE_M
        group_size = min(num_blocks_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
        pid_m = group_id * GROUP_SIZE_M + (pid_m % group_size)
    
    # Create block offsets for M and N
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offsets for K dimension
        offset_k = k * BLOCK_SIZE_K
        offsets_k = offset_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < K
        
        # Load blocks from A and B
        a_ptrs = A_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        
        b_ptrs = B_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        
        # Perform matrix multiplication for this block
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to the appropriate output type if needed
    # For FP32, we can store directly
    c = accumulator.to(tl.float32)
    
    # Store result
    c_ptrs = C_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 is supported."
    
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions must match for multiplication."
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 256
    
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
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)
        
        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)