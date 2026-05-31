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
    
    # Compute the starting positions for this block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers to the start of each tile
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation if specified
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # Compute output pointers
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Store result
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using Triton kernel.
    """
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D"
    assert a.shape[1] == b.shape[1], "Inner dimensions must match for matrix multiplication"
    
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=None,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        Note: B is transposed internally in the kernel to match the expected operation.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K) - will be transposed internally.

        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose B to get (K, N) shape for proper matrix multiplication
        B_t = B.transpose(0, 1)
        return triton_matmul(A, B_t)