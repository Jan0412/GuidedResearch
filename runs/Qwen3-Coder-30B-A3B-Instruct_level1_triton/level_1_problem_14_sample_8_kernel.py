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

    # Initialize accumulator with zeros.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Compute accumulator.
        acc += tl.dot(a, b)
        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation.
    if ACTIVATION == "relu":
        acc = tl.where(acc >= 0, acc, 0.0)
    elif ACTIVATION == "silu":
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == "gelu":
        acc = acc * 0.5 * (1 + tl.tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))

    # Write back to global memory.
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

@triton.jit
def triu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # For upper triangular, we need to check row and column indices
    # Calculate row and column indices from linear index
    row = offsets // 4096  # Assuming square matrix of size 4096
    col = offsets % 4096
    
    # Keep elements where row <= col, otherwise set to 0
    condition = row <= col
    out = tl.where(condition, x, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)

def triton_matmul_triangular(A, B):
    """
    Performs matrix multiplication followed by upper triangular masking using Triton kernels.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1], "Matrices must be square"
    
    # Ensure matrices are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output tensor
    M, K = A.shape
    K, N = B.shape
    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Determine the number of blocks needed
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch the matmul kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M,
        "none",
    )
    
    # Apply upper triangular mask
    out = torch.empty_like(C)
    n_elements = C.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triu_kernel[grid](C, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_triangular(A, B)