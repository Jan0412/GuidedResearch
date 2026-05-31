import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr,  # Pointer to input A (original shape K x M), but we'll treat it as M x K due to transpose
    B_ptr,  # Pointer to input B (original shape N x K), but we'll treat it as K x N due to transpose
    C_ptr,  # Pointer to output C (shape M x N)
    M, N, K,  # Dimensions after transpose: A^T is M x K, B^T is K x N, C = A^T * B^T is M x N
    stride_am, stride_ak,  # Strides for A^T (which is A transposed, so original A has strides (K, 1))
    stride_bk, stride_bn,  # Strides for B^T (which is B transposed, so original B has strides (1, N))
    stride_cm, stride_cn,  # Strides for output C
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    amask = offs_m < M
    bnmask = offs_n < N
    bkmask = offs_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Get current K offset
        k_offset = k * BLOCK_SIZE_K
        k_mask = k_offset + offs_k < K
        
        # Load A^T block: A^T is M x K, so row is m, column is k
        # Original A is K x M, so A^T[i,j] = A[j,i]
        # For A^T[offs_m, k_offset + offs_k], we need A[k_offset + offs_k, offs_m]
        # Since A has strides (K, 1), A[ki, mi] is at A_ptr[ki * K_stride + mi * 1]
        # But K_stride for original A is M (since A is K x M: stride for row is M, stride for col is 1)
        # So A[ki, mi] is at A_ptr[ki * M + mi]
        # For A^T, we need A[k_offset + offs_k, offs_m], so indices are [k_offset + offs_k, offs_m]
        a_ptr = A_ptr + (k_offset + offs_k[:, None]) * M + offs_m[None, :]
        a = tl.load(a_ptr, mask=k_mask[:, None] & amask[None, :], other=0.0)
        
        # Load B^T block: B^T is K x N, so row is k, column is n
        # Original B is N x K, so B^T[i,j] = B[j,i]
        # For B^T[k_offset + offs_k, offs_n], we need B[offs_n, k_offset + offs_k]
        # Since B is N x K: stride for row is K, stride for col is 1
        # So B[ni, ki] is at B_ptr[ni * K + ki]
        # For B^T, we need B[offs_n, k_offset + offs_k], so indices are [offs_n, k_offset + offs_k]
        b_ptr = B_ptr + (offs_n[:, None]) * K + (k_offset + offs_k[None, :])
        b = tl.load(b_ptr, mask=bnmask[:, None] & k_mask[None, :], other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Cast to output dtype and store
    acc = acc.to(tl.float32)
    c_ptr = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = amask[:, None] & bnmask[None, :]
    tl.store(c_ptr, acc, mask=c_mask)


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T * B^T using custom Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (N, K)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Extract dimensions
    K, M = A.shape
    N, _ = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        M, 1,  # strides for A^T (A is K x M, so A^T strides are M, 1)
        1, N,  # strides for B^T (B is N x K, so B^T strides are 1, N)
        N, 1,  # strides for C (M x N)
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    Computes C = A^T * B^T where A has shape (K, M) and B has shape (N, K)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose(A, B)