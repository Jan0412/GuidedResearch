import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
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
    off_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = off_m[:, None] < M
    b_mask = off_n[None, :] < N
    ak_mask = off_k < K
    bk_mask = off_k < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = A + (off_m[:, None] * stride_am + (k + off_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=a_mask & ak_mask[None, :], other=0.0)
        
        # Load B block
        b_ptrs = B + ((k + off_k[:, None]) * stride_bk + off_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=bk_mask[:, None] & b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc = tl.dot(a, b, acc)

    # Store result
    c_ptrs = C + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    c_mask = a_mask & b_mask
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized Triton kernel for matrix multiplication.
    A: (M, K), B: (K, N) -> C: (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"

    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Define block sizes (tuned for large K dimension)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 256  # Large block size for large K dimension
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    GROUP_SIZE_M = 8
    num_groups = ((num_pid_m + GROUP_SIZE_M - 1) // GROUP_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_pid = num_groups * num_pid_in_group
    
    # Calculate strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    matmul_kernel[lambda meta: (total_pid,)](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)