import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (M x K) - stored as contiguous A transposed
    B_ptr,    # Pointer to B (K x N)
    C_ptr,    # Pointer to C (M x N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create tile offsets for M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping strategy to improve cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = (pid_m % num_pid_n) + (pid_m // num_pid_n) * num_pid_n
    
    # Offset for this block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create ranges for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A^T block: A^T is (M x K), so we need A^T[m, k]
        # A^T has strides [stride_am, stride_ak] where stride_am is for M dimension
        a_offsets = offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a = tl.load(A_T_ptr + a_offsets, mask=mask_m[:, None], other=0.0)
        
        # Load B block: B is (K x N), so we need B[k, n]
        b_offsets = offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b = tl.load(B_ptr + b_offsets, mask=mask_n[None, :], other=0.0)
        
        # Matrix multiplication accumulate
        acc = tl.dot(a, b, acc)
    
    # Convert accumulator to float16 if needed, but for FP32 keep as float32
    acc = acc.to(tl.float32)
    
    # Store result
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(C_ptr + c_offsets, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T * B where A is (K x M), B is (K x N), result is (M x N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Calculate strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Define block sizes (tuned for FP32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    grid = (num_pid_m, num_pid_n)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication C = A^T * B using Triton kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T * B using optimized Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)