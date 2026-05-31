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

    # Initialize accumulator with zeros.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load a and b tiles.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Compute dot product.
        accumulator += tl.dot(a, b)
        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation.
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    elif ACTIVATION == "gelu":
        accumulator = tl.gelu(accumulator)

    # Write back to global memory.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def tril_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map program ID to the block of output it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for this block.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    input_ptrs = input_ptr + (offs_am[:, None] * stride_input_row + offs_bn[None, :] * stride_input_col)
    output_ptrs = output_ptr + (offs_am[:, None] * stride_output_row + offs_bn[None, :] * stride_output_col)

    # Load input tile.
    input_tile = tl.load(input_ptrs, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N), other=0.0)

    # Apply lower triangular mask.
    mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_N)[None, :]
    result = tl.where(mask, input_tile, 0.0)

    # Store result.
    tl.store(output_ptrs, result, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def triton_matmul(A, B):
    """Custom Triton implementation of matrix multiplication"""
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"

    M, K = A.shape
    K, N = B.shape

    # Allocate output tensor
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)

    # Define parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M,
        "none"
    )
    return C

def triton_tril(C):
    """Custom Triton implementation of lower triangular masking"""
    assert C.is_cuda, "Tensor must be on CUDA"
    
    M, N = C.shape
    
    # Allocate output tensor
    D = torch.empty(M, N, device=C.device, dtype=torch.float32)

    # Define parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    tril_kernel[grid](
        C, D,
        M, N,
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return D

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Use Triton kernels for both operations
        C = triton_matmul(A, B)
        D = triton_tril(C)
        return D