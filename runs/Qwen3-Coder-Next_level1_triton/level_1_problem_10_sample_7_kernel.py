import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to 3D tensor A of shape (N, M, K)
    B_ptr,  # Pointer to matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor C of shape (N, M, L)
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,  # Strides for A: (M*K, K, 1)
    stride_b0, stride_b1,            # Strides for B: (L, 1)
    stride_c0, stride_c1, stride_c2, # Strides for C: (M*L, L, 1)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Grouped gemm scheduling for better cache utilization
    # Create a grid of blocks over N and M dimensions
    # Group size determines how many blocks are processed together
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Compute group ID and local IDs
    group_id = pid_n // GROUP_SIZE_M
    first_m_block = group_id * GROUP_SIZE_M
    group_size_m = min(num_blocks_m - first_m_block, GROUP_SIZE_M)
    
    # Update pid_m based on group
    pid_m = pid_m + first_m_block
    group_size_m = tl.where(pid_m < num_blocks_m, group_size_m, 0)
    
    # Create ranges for blocks
    block_n = pid_n % GROUP_SIZE_M
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    l_offsets = tl.arange(0, BLOCK_SIZE_L)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute k offsets with mask
        k_block = k * BLOCK_SIZE_K
        k_mask = k_offsets < (K - k_block)
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A has shape (N, M, K), we're accessing A[n, m_block, k_block]
        # We need to handle multiple N indices, but for now assume we process one N at a time
        # Actually, we'll process one (n, m_block) at a time, so we need to select the correct N
        a_ptrs = A_ptr + (pid_n * M * K + m_offsets[:, None] * K + k_block + k_offsets[None, :]) * stride_a0
        a_mask = (m_offsets[:, None] < M) & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_ptrs = B_ptr + (k_block + k_offsets[:, None]) * stride_b0 + l_offsets[None, :] * stride_b1
        b_mask = k_mask[:, None] & (l_offsets[None, :] < L)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Convert to output dtype if needed (FP32 in this case)
    acc = acc.to(tl.float32)
    
    # Store result
    c_ptrs = C_ptr + (pid_n * M * L + m_offsets[:, None] * L + l_offsets[None, :]) * stride_c0
    c_mask = (m_offsets[:, None] < M) & (l_offsets[None, :] < L)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).
        
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"
    
    # Prepare output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        N,  # Number of blocks over N dimension (1 block per N in this implementation)
        tl.cdiv(M, BLOCK_SIZE_M),  # Number of blocks over M dimension
    )
    
    # Launch kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_a0=M * K, stride_a1=K, stride_a2=1,
        stride_b0=L, stride_b1=1,
        stride_c0=M * L, stride_c1=L, stride_c2=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul_3d(A, B)