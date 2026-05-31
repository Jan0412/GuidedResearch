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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A and B.
    a_ptrs = a_ptr + (offs_am * stride_am + offs_k * stride_ak)
    b_ptrs = b_ptr + (offs_k * stride_bk + offs_bn * stride_bn)
    
    # Ensure bounds
    a_mask = (offs_am < M) & (offs_k < K)
    b_mask = (offs_k < K) & (offs_bn < N)
    
    # Dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_mask = (offs_am < M) & (offs_k < K)
        b_mask = (offs_k < K) & (offs_bn < N)
    
    # Store result
    c_ptrs = c_ptr + (offs_am * stride_cm + offs_bn * stride_cn)
    c_mask = (offs_am < M) & (offs_bn < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def tril_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    
    # Flatten indices
    row = offsets // N
    col = offsets % N
    
    # Check if element is below or on diagonal
    condition = row >= col
    
    # Load input value
    input_val = tl.load(input_ptr + row * stride_input_row + col * stride_input_col, mask=mask, other=0.0)
    
    # Conditionally store
    output_val = tl.where(condition, input_val, 0.0)
    tl.store(output_ptr + row * stride_output_row + col * stride_output_col, output_val, mask=mask)

def triton_matmul_tril(a, b):
    """
    Performs matrix multiplication followed by lower triangular masking using Triton kernels.
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA."
    assert a.shape[1] == b.shape[0], "Matrix dimensions incompatible for multiplication"
    
    M, K = a.shape
    K, N = b.shape
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Create output tensor
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch matmul kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    # Apply lower triangular mask
    result = torch.empty_like(c)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((M * N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    tril_kernel[grid](
        c, result,
        M, N,
        c.stride(0), c.stride(1),
        result.stride(0), result.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_tril(A, B)