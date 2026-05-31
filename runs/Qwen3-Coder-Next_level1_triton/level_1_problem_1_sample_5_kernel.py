import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for matrix multiplication (C = A * B)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program ID is the index of the block in the grid
    pid = tl.program_id(0)
    
    # Compute the row and column of the matrix C that this program should compute
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping to improve cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create offsets for rows and columns
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to handle out-of-bounds accesses
    a_mask = (offset_am[:, None] < M) & (offset_k[None, :] < K)
    b_mask = (offset_k[:, None] < K) & (offset_bn[None, :] < N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a block of A
        a = tl.load(A_ptr + offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak, mask=a_mask, other=0.0)
        # Load a block of B
        b = tl.load(B_ptr + offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn, mask=b_mask, other=0.0)
        # Accumulate the product
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to float16 if needed (but we want FP32 as per requirement)
    c = accumulator.to(tl.float32)
    
    # Write back to C
    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(C_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check matrix dimensions
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Allocate output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Launch the kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)