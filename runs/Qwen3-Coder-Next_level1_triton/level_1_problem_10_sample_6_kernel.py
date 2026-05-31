import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to input 3D tensor A of shape (N, M, K)
    B_ptr,  # Pointer to input matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor C of shape (N, M, L)
    N, M, K, L,
    stride_A0, stride_A1, stride_A2,  # Strides for A: (M*K, K, 1)
    stride_B0, stride_B1,  # Strides for B: (L, 1)
    stride_C0, stride_C1, stride_C2,  # Strides for C: (M*L, L, 1)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    # Program IDs for the batch dimension (N) and output rows (M)
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Skip if batch index is out of bounds
    if pid_batch >= N:
        return

    # Offset for the batch in A and C
    A_batch_offset = pid_batch * stride_A0
    C_batch_offset = pid_batch * stride_C0
    
    # Compute row start index for C
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M
    
    # Compute column start index for C
    col_start = pid_n * BLOCK_SIZE_L
    cols = col_start + tl.arange(0, BLOCK_SIZE_L)
    col_mask = cols < L
    
    # Combine row and column masks
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Accumulator for the matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load block of A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = A_batch_offset + rows[:, None] * stride_A1 + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_A2
        a = tl.load(A_ptr + a_offsets, mask=row_mask[:, None] & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
        
        # Load block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_offsets = (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_B0 + cols[None, :] * stride_B1
        b = tl.load(B_ptr + b_offsets, mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & col_mask[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b)
    
    # Convert accumulator to output type and store result
    c = acc.to(tl.float32)  # Ensure output is float32
    c_offsets = C_batch_offset + rows[:, None] * stride_C1 + cols[None, :] * stride_C2
    tl.store(C_ptr + c_offsets, c, mask=mask)


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.

    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).

    Returns:
        torch.Tensor: Output tensor of shape (N, M, L).
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, "Inner dimensions must match for matrix multiplication"
    
    # Prepare output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 64
    
    # Calculate grid dimensions
    grid = lambda meta: (
        N,
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(L, meta["BLOCK_SIZE_L"]),
    )
    
    # Launch kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        return triton_matmul_3d(A, B)