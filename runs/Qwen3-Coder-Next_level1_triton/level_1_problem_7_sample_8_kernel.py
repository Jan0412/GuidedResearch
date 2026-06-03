import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID for the M dimension
    pid = tl.program_id(0)
    # Number of programs in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs in N dimension
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Number of programs in GROUP_SIZE_M groups
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Group ID
    group_id = pid // num_pid_in_group
    # First program ID in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # Remaining program IDs in the group
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # Program ID within the group
    pid_m = first_pid_m + (pid % group_size_m)
    # Program ID for N dimension
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offsets for blocks of M and N
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    am_mask = offsets_am[:, None] < M
    bn_mask = offsets_bn[None, :] < N
    bk_mask = offsets_k[None, :] < K
    
    # Initialize accumulator for the block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offset for K dimension
        k_offset = k * BLOCK_SIZE_K
        
        # Load tile from A matrix
        a = tl.load(
            A_ptr + offsets_am[:, None] * stride_am + (k_offset + offsets_k)[None, :] * stride_ak,
            mask=am_mask & (offsets_k[None, :] < K),
            other=0.0
        )
        
        # Load tile from B matrix
        b = tl.load(
            B_ptr + (k_offset + offsets_k)[:, None] * stride_bk + offsets_bn[None, :] * stride_bn,
            mask=bk_mask[:, None] & bn_mask,
            other=0.0
        )
        
        # Accumulate the matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to output type
    c = accumulator.to(tl.float32)
    
    # Store result to C matrix
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(
        C_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn,
        c,
        mask=c_mask
    )


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
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    assert A.dtype == B.dtype, "Input tensors must have the same dtype"
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
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
    Optimized version of Model that uses Triton kernel for matrix multiplication.
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