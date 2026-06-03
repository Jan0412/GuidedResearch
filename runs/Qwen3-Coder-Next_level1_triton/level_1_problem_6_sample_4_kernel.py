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
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block coordinates
    # Grouping M blocks to improve cache utilization
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group M blocks together for better cache locality
    num_groups = num_blocks_m // GROUP_SIZE_M
    group_id = pid_m // GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid_m % group_size_m)
    pid_n = pid_n
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Calculate current K offset
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offset_k < K
        
        # Load tiles from A and B
        # A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_offsets = offsets_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        a = tl.load(A + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_offsets = offset_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b = tl.load(B + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result to C
    c = accumulator.to(tl.float32)  # Ensure output is float32
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(C + c_offsets, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication C = A * B
    Optimized for large K dimensions
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
    
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    
    # Ensure K dimensions match
    assert K == K_b, f"Dimension mismatch: A.shape[1]={K}, B.shape[0]={K_b}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes for optimal performance with large K
    # These are tuned for large K dimensions and FP32
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 256  # Larger K blocks help with memory bandwidth utilization
    
    # Group size for better cache utilization
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
    Optimized version of the Model class using Triton kernels for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication of A and B using Triton kernel.
        """
        return triton_matmul(A, B)