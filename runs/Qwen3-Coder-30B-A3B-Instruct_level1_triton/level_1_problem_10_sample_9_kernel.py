import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix
    b_ptr,  # Pointer to second input matrix
    c_ptr,  # Pointer to output matrix
    M,      # Number of rows in A and C
    N,      # Number of columns in B and C
    K,      # Number of columns in A and rows in B
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # Map program ID to the block of C it should compute
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Which block group should this program handle?
    group_id = pid // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % grid_n)
    
    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator with appropriate type
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute accumulator
        acc += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write back result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using Triton kernel
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA"
    assert a.dim() == 3 and b.dim() == 2, "A must be 3D and B must be 2D"
    assert a.size(-1) == b.size(0), "Inner dimensions must match"
    
    # Get shapes
    N, M, K = a.shape
    K2, L = b.shape
    assert K == K2, "Matrix dimensions incompatible"
    
    # Create output tensor
    c = torch.empty((N, M, L), device=a.device, dtype=torch.float32)
    
    # Define parameters
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_M = 8
    
    # Calculate grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(L, meta["BLOCK_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        N * M, L, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        ACC_TYPE=tl.float32
    )
    
    return c

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using Triton optimization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul(A, B)