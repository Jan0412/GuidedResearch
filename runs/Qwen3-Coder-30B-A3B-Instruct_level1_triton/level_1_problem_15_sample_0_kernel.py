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
    # Get the program ID and determine which tile to compute
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute partial dot product
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Create pointer for output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Store result
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.jit
def lower_triangular_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid rows and columns
    row_mask = row[:, None] < M
    col_mask = col[None, :] < N
    
    # Create full mask for triangular matrix (lower triangular including diagonal)
    triangular_mask = col[None, :] >= row[:, None]
    
    # Combine all masks
    mask = row_mask & col_mask & triangular_mask
    
    # Load input
    input_data = tl.load(input_ptr + row[:, None] * stride_input_row + col[None, :] * stride_input_col, mask=mask, other=0.0)
    
    # Store output
    tl.store(output_ptr + row[:, None] * stride_output_row + col[None, :] * stride_output_col, input_data, mask=mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """Custom Triton kernel for matrix multiplication of lower triangular matrices"""
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    
    M, K = A.shape
    K, N = B.shape
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output tensor
    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid
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
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    
    return C

def triton_lower_triangular(C: torch.Tensor):
    """Custom Triton kernel to enforce lower triangular structure"""
    assert C.is_cuda, "Tensor must be on CUDA"
    
    M, N = C.shape
    C = C.contiguous()
    output = torch.empty_like(C, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 128
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE)
    )
    
    lower_triangular_kernel[grid](
        C, output,
        M, N,
        C.stride(0), C.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Perform matrix multiplication using custom Triton kernel
        C = triton_matmul(A, B)
        # Apply lower triangular mask using custom Triton kernel
        return triton_lower_triangular(C)