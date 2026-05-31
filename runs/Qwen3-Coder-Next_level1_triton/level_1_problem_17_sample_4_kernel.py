import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for optimized matrix multiplication with transpose (C = A * B^T)
@triton.jit
def matmul_transpose_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and local IDs
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_am = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = block_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    a_mask = offsets_am[:, None] < M
    b_mask = offsets_bn[:, None] < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block (M x K)
        offsets_ak = k + offsets_k
        a = tl.load(A + offsets_am[:, None] * stride_am + offsets_ak[None, :] * stride_ak, mask=a_mask, other=0.0)
        
        # Load B block (N x K), but since we need B^T, we transpose the indices
        # B^T has shape (N, K), so element [i,j] of B^T is B[j,i]
        # We want to access B^T[offsets_bn, offsets_ak] = B[offsets_ak, offsets_bn]
        offsets_bk = k + offsets_k  # This corresponds to K dimension of original B
        b = tl.load(B + offsets_bk[:, None] * stride_bk + offsets_bn[None, :] * stride_bn, mask=b_mask, other=0.0)
        
        # Matrix multiplication: accumulator += A * B^T
        # Note: tl.dot handles the multiplication and accumulation
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to output type and store
    c = accumulator.to(C.dtype.element_ty)
    
    # Write back result
    offsets_cm = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = block_n + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(C + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn, c, mask=c_mask)


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B^T using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (N, K) - note: we use B^T internally
    
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel launch parameters
    # Use heuristics for block sizes based on problem size
    BLOCK_SIZE_M = 128 if M >= 512 else 64
    BLOCK_SIZE_N = 128 if N >= 512 else 64
    BLOCK_SIZE_K = 32 if K >= 2048 else 16
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    matmul_transpose_kernel[grid](
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
    Optimized model that performs a single matrix multiplication (C = A * B^T) using Triton kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A * B^T using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose(A, B)