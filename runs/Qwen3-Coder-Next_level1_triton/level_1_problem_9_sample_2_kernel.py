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
    
    # Compute which tile row this program should handle
    # Grouping rows to improve cache locality
    num_program_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_program_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_program_in_group = GROUP_SIZE_M * num_program_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_program_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = tl.program_id(1)
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offset ranges for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure offsets are within bounds
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load tile from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            offsets_m[:, None] * stride_am +
            offsets_k[None, :] * stride_ak
        )
        a = tl.load(A_ptr + a_offsets, mask=mask_m[:, None], other=0.0)
        
        # Load tile from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (
            offsets_k[:, None] * stride_bk +
            offsets_n[None, :] * stride_bn
        )
        b = tl.load(B_ptr + b_offsets, mask=mask_n[None, :], other=0.0)
        
        # Accumulate the block matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Cast to output dtype and store result
    c = accumulator.to(tl.float32)
    
    c_offsets = (
        offsets_m[:, None] * stride_cm +
        offsets_n[None, :] * stride_cn
    )
    tl.store(C_ptr + c_offsets, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    Supports A of shape (M, K) and B of shape (K, N) -> C of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported"
    assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} @ {B.shape}"
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tuning parameters)
    # For M much larger than N, we want more blocks in M direction
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Calculate strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,  # Commonly used value for grouping
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)