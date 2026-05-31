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
    # Map program id to the block of C it should compute.
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

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute matmul
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation if specified
    if ACTIVATION == "relu":
        acc = tl.where(acc > 0, acc, 0.0)
    elif ACTIVATION == "gelu":
        acc = tl.gelu(acc)

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

@triton.jit
def triu_kernel(
    input_ptr, output_ptr,
    size,
    stride_input,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid out-of-bounds access
    mask = idx < size
    
    # Load input values
    input_vals = tl.load(input_ptr + idx * stride_input, mask=mask, other=0.0)
    
    # Set lower triangular part to zero
    # We use a simple approach where we check if the index corresponds to a position 
    # in the lower triangle of the matrix
    row = idx // size
    col = idx % size
    mask_tri = col >= row
    
    # Apply triangular mask
    output_val = tl.where(mask_tri, input_vals, 0.0)
    
    # Store result
    tl.store(output_ptr + idx * stride_output, output_val, mask=mask)

def triton_matmul_triu(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication followed by upper triangular masking.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.shape[1] == b.shape[0], "Matrix dimensions incompatible for multiplication"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Get dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Prepare output tensor
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Determine the number of blocks needed
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch the Triton kernel
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
        ACTIVATION="none"
    )
    
    # Apply upper triangular mask
    c_out = torch.empty_like(c)
    
    # Flatten the tensors for kernel processing
    flat_c = c.view(-1)
    flat_c_out = c_out.view(-1)
    
    # Launch triangular kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ((flat_c.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triu_kernel[grid](
        flat_c, flat_c_out,
        flat_c.numel(),
        1, 1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return c_out.view(M, N)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using optimized Triton kernels.
        """
        return triton_matmul_triu(A, B)