import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (transposed A, now shape M x K)
    B_ptr,    # Pointer to B (shape K x N)
    C_ptr,    # Pointer to output C (shape M x N)
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
    
    # Create tile offsets for M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for better cache utilization (similar to CUTLASS)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = (pid_m % num_pid_n) + (pid_n * group_size_m) % num_pid_n
    
    # Create tile offsets for M and N dimensions
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid indices
    mask_m = offset_m < M
    mask_n = offset_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offset_k < K
        
        # Load tile from A^T (shape M x K)
        # A^T is stored as A_T_ptr with stride_am (M) and stride_ak (K)
        a = tl.load(
            A_T_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load tile from B (shape K x N)
        # B has stride_bk (K) and stride_bn (N)
        b = tl.load(
            B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Matrix multiplication: acc += a @ b
        acc = tl.dot(a, b, acc)
    
    # Convert accumulator to output type and store
    acc = acc.to(tl.float32)
    tl.store(
        C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication C = A^T * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    K, M = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Inner dimensions must match: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    # Launch kernel
    matmul_kernel[
        (num_pid_m, num_pid_n),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    ](
        A.T.data_ptr(),  # A^T pointer (shape M x K)
        B.data_ptr(),    # B pointer (shape K x N)
        C.data_ptr(),    # C pointer (shape M x N)
        M, N, K,
        A.T.stride(0), A.T.stride(1),  # stride_am, stride_ak
        B.stride(0), B.stride(1),      # stride_bk, stride_bn
        C.stride(0), C.stride(1),      # stride_cm, stride_cn
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    Computes C = A^T * B where A has shape (K, M) and B has shape (K, N).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose(A, B)