import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (transposed A, originally A with shape (K, M) -> A^T has shape (M, K))
    B_ptr,    # Pointer to B with shape (K, N)
    C_ptr,    # Pointer to output C with shape (M, N)
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    off_m_mask = off_m < M
    off_n_mask = off_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)
        off_k_mask = off_k < K
        
        # Load tiles from A^T (shape MxK) - note A_T_ptr points to A^T
        # A^T[off_m, off_k]
        a_tile = tl.load(
            A_T_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak,
            mask=off_m_mask[:, None] & off_k_mask[None, :],
            other=0.0
        )
        
        # Load tiles from B (shape KxN) - B[off_k, off_n]
        b_tile = tl.load(
            B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn,
            mask=off_k_mask[:, None] & off_n_mask[None, :],
            other=0.0
        )
        
        # Matrix multiplication
        accumulator = tl.dot(a_tile, b_tile, accumulator)
    
    # Convert to float16 if needed, but for FP32 we keep as is
    c_result = accumulator.to(tl.float32)
    
    # Store result
    tl.store(
        C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn,
        c_result,
        mask=off_m_mask[:, None] & off_n_mask[None, :]
    )


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A^T @ B using Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
        
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    _, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        A.T.contiguous(),  # A^T has shape (M, K) and is contiguous after .T
        B,
        C,
        M, N, K,
        A.T.contiguous().stride(0), A.T.contiguous().stride(1),
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
    Optimized model that performs matrix multiplication C = A^T * B using Triton kernel
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
        return triton_matmul_transpose(A, B)