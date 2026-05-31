import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for matrix A (rows)
    offsets_am = block_m + tl.arange(0, BLOCK_SIZE_M)
    # Create offsets for matrix B (columns)
    offsets_bn = block_n + tl.arange(0, BLOCK_SIZE_N)
    # Create offsets for reduction dimension
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers for A and B
    A_ptrs = A_ptr + (offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(A_ptrs, mask=offsets_k[None, :] < N - k, other=0.0)
        b = tl.load(B_ptrs, mask=offsets_k[:, None] < N - k, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to float16 if needed (but keeping as float32 for FP32 requirement)
    C_block = accumulator.to(tl.float32)
    
    # Store result
    offsets_cm = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = block_n + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + (offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn)
    mask = (offsets_cm[:, None] < N) & (offsets_cn[None, :] < N)
    tl.store(C_ptrs, C_block, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel wrapper for matrix multiplication C = A * B
    Optimized for FP32 precision with tiled computation and grouped scheduling.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    N = A.size(0)
    
    # Create output tensor
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes - tuned for FP32 on modern GPUs
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    num_pid_m = triton.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_programs = num_pid_m * num_pid_n
    
    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    matmul_kernel[(num_programs,)](
        A, B, C,
        N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
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
        Performs the matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)