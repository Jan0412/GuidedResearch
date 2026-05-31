import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_bh,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    
    # Group rows for better cache utilization
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = 1  # Only one column in output
    
    # Grouping logic
    pid_m = pid % num_pid_m
    pid_n = 0
    
    # Create row offsets for this block
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < M
    
    # Create column offset (always 0 since output is Mx1)
    col_offset = 0
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load A block: M x K_block
        a_offsets = row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(A_ptr + a_offsets, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)
        
        # Load B block: K_block x 1
        b_offsets = k_offsets[:, None] * stride_bk + col_offset * stride_bh
        b = tl.load(B_ptr + b_offsets, mask=(k_mask[:, None]), other=0.0)
        
        # Matrix-vector multiplication (sum over K dimension)
        accumulator += tl.sum(a * b, axis=1)
    
    # Convert accumulator to float32 and store
    c = accumulator.to(tl.float32)
    
    # Store result
    c_offsets = row_offsets * stride_cm + col_offset * stride_cn
    tl.store(C_ptr + c_offsets, c, mask=row_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K, 1)
        
    Returns:
        Output vector of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 2, "A must be 2D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"
    assert B.shape[1] == 1, "B must be a column vector"
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, _ = B.shape
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 256
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid = (num_pid_m,)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)