import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map program ID to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zero.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute matrix multiplication.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation if specified.
    if ACTIVATION == "relu":
        acc = tl.where(acc > 0, acc, 0.0)

    # Store the result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

@triton.jit
def triu_kernel(
    input_ptr, output_ptr,
    N,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * N
    
    # Calculate row and column indices
    row = offsets // N
    col = offsets % N
    
    # Load input value
    input_val = tl.load(input_ptr + row * stride_input_row + col * stride_input_col, mask=mask, other=0.0)
    
    # Keep only upper triangular part
    output_val = tl.where(row <= col, input_val, 0.0)
    
    # Store result
    tl.store(output_ptr + row * stride_output_row + col * stride_output_col, output_val, mask=mask)

def triton_matmul_triangular(A, B):
    """Custom Triton kernel for matrix multiplication followed by upper triangular masking."""
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Matrix dimensions must match for multiplication"
    
    M, K = A.shape
    K, N = B.shape
    
    # Ensure matrices are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output tensor
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    # Launch matmul kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        "none"
    )
    
    # Apply upper triangular mask
    output = torch.empty_like(C)
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((M * N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    triu_kernel[grid](
        C, output,
        M,
        C.stride(0), C.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for matrix multiplication and upper triangular masking.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using optimized Triton kernels.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_matmul_triangular(A, B)