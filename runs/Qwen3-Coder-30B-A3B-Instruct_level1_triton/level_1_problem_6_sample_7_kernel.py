import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix A
    b_ptr,  # Pointer to second input matrix B
    c_ptr,  # Pointer to output matrix C
    M,      # Number of rows in A and C
    N,      # Number of columns in B and C
    K,      # Number of columns in A and rows in B
    stride_am, stride_ak,  # Stride of matrix A
    stride_bk, stride_bn,  # Stride of matrix B
    stride_cm, stride_cn,  # Stride of matrix C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Identify the block ID for this program
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # Compute partial accumulator
        acc += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Perform matrix multiplication using Triton kernel.
    """
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.size(1) == b.size(0), "Matrix dimensions incompatible for multiplication"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only FP32 supported"

    M, K = a.shape
    K, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8

    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)