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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Grouped program ID
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for rows and columns
    offsets_am = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = block_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    a_mask = (offsets_am[:, None] < M) & (offsets_k[None, :] < K)
    b_mask = (offsets_k[:, None] < K) & (offsets_bn[None, :] < N)
    
    # Initialize accumulator for C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        a = tl.load(A_ptr + (offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak), mask=a_mask, other=0.0)
        b = tl.load(B_ptr + (offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn), mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Cast to output type and store result
    c = accumulator.to(tl.float32)
    
    # Write back to C
    offsets_cm = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = block_n + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(C_ptr + (offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn), c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
        
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for matrix multiplication"
    
    # Prepare output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    matmul_kernel[grid](
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
    Optimized model that performs a single matrix multiplication (C = A * B)
    using custom Triton kernel.
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