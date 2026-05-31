import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,  # Pointer to matrix A (M, K)
    B_ptr,  # Pointer to vector B (K, 1)
    C_ptr,  # Pointer to output vector C (M, 1)
    M,      # Number of rows in A
    K,      # Number of columns in A
    stride_am,  # Stride for A
    stride_bk,  # Stride for B
    stride_cm,  # Stride for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    
    # Group rows to improve cache locality
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = 1  # Only one column in output
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    
    # Create row offsets for this block
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M
    
    # Create column offsets for this block
    col_start = 0  # Only one column in B and C
    cols = col_start + tl.arange(0, 1)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load row block from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = rows[:, None] * stride_am + k_offsets[None, :] * 1  # A is row-major
        a_block = tl.load(A_ptr + a_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load column block from B: shape (BLOCK_SIZE_K, 1)
        b_offsets = k_offsets[:, None] * stride_bk + cols[None, :] * 1  # B is column vector
        b_block = tl.load(B_ptr + b_offsets, mask=k_mask[:, None], other=0.0)
        
        # Compute partial dot product and accumulate
        accumulator += tl.sum(a_block * b_block, axis=1).to(tl.float32)
    
    # Store result
    c_offsets = rows * stride_cm + cols[None, :]
    tl.store(C_ptr + c_offsets, accumulator[:, None], mask=row_mask[:, None])


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K, 1)
    
    Returns:
        Output vector of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Matrix and vector dimensions must match"
    assert N == 1, "B must be a column vector"
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 256
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = 1
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    matvec_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), B.stride(0), C.stride(0),
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
        
        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).
        
        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)