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
    
    # Compute start indices for M and N blocks
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped layout for better cache utilization
    if GROUP_SIZE_M > 1:
        group_id = pid_m // GROUP_SIZE_M
        group_size_m = min(num_blocks_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
        pid_m = group_id * GROUP_SIZE_M + (pid_m % group_size_m)
    
    # Compute block start indices
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create index ranges
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute current K offset
        k_offset = k * BLOCK_SIZE_K
        k_mask = k_offset + offsets_k < K
        
        # Load tile from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            (offsets_m[:, None] * stride_am + 
             (k_offset + offsets_k[None, :]) * stride_ak)
        )
        a = tl.load(A_ptr + a_offsets, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        
        # Load tile from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (
            ((k_offset + offsets_k[:, None]) * stride_bk + 
             offsets_n[None, :] * stride_bn)
        )
        b = tl.load(B_ptr + b_offsets, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiplication with accumulation
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
    
    # Convert accumulator to output type and store
    c = accumulator.to(tl.float32)
    
    # Compute output offsets
    c_offsets = (
        offsets_m[:, None] * stride_cm + 
        offsets_n[None, :] * stride_cn
    )
    tl.store(C_ptr + c_offsets, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    # Launch kernel
    matmul_kernel[grid_m, grid_n](
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
    Optimized version of the model using Triton matrix multiplication kernel.
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