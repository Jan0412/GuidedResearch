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
    
    # Offset pointers for batching
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator with zero
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Dot product loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation if specified
    if ACTIVATION == "leaky_relu":
        acc = tl.leaky_relu(acc)
    
    # Compute upper triangular mask
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = off_m[:, None] <= off_n[None, :]
    
    # Store result
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, acc, mask=mask)

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
    
    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute row and column indices
    row = offsets // N
    col = offsets % N
    
    # Apply upper triangular mask
    masked_elements = tl.where(row <= col, input_elements, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, masked_elements, mask=mask)

def triton_matmul_triu(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton implementation for matrix multiplication followed by upper triangular masking.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
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
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        "none",
    )
    
    return c

def triton_triu(input_tensor: torch.Tensor):
    """
    Custom Triton implementation for applying upper triangular mask.
    """
    assert input_tensor.is_cuda, "Tensor must be on CUDA"
    
    N = input_tensor.shape[0]
    assert input_tensor.shape == (N, N), "Input must be square matrix"
    
    # Ensure tensor is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid
    grid = lambda meta: ((N * N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    triu_kernel[grid](
        input_tensor, output,
        N,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE,
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
        # Perform matrix multiplication using Triton kernel
        matmul_result = triton_matmul_triu(A, B)
        
        # Apply upper triangular mask using Triton kernel
        result = triton_triu(matmul_result)
        
        return result