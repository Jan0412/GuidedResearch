import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A,  # Pointer to A matrix (N, M, K)
    B,  # Pointer to B matrix (K, L)
    C,  # Pointer to C matrix (N, M, L)
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,  # Strides for A: (N, M, K)
    stride_b0, stride_b1,  # Strides for B: (K, L)
    stride_c0, stride_c1, stride_c2,  # Strides for C: (N, M, L)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    # Grouping for better performance
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(N * M, BLOCK_SIZE_M) // num_pid_n
    
    # Compute batch and position within batch
    batch_id = pid // (num_pid_m * num_pid_n)
    pid_m = (pid % (num_pid_m * num_pid_n)) // num_pid_n
    pid_n = pid % num_pid_n
    
    # For simplicity, we'll flatten N and M into a single dimension
    # and handle the 3D structure by computing offsets appropriately
    # Actually, let's handle this more systematically
    
    # Compute which (batch_n, block_m) this program handles
    batch_n = pid // (tl.cdiv(M, BLOCK_SIZE_N) * tl.cdiv(L, BLOCK_SIZE_L))
    block_m = (pid // tl.cdiv(L, BLOCK_SIZE_L)) % tl.cdiv(M, BLOCK_SIZE_N)
    block_l = pid % tl.cdiv(L, BLOCK_SIZE_L)
    
    # But since we have N*M batches, let's simplify:
    # Each program handles one (batch_idx, block_m, block_l) where batch_idx in [0, N*M)
    batch_idx = pid // (tl.cdiv(M, BLOCK_SIZE_N) * tl.cdiv(L, BLOCK_SIZE_L))
    block_m = (pid // tl.cdiv(L, BLOCK_SIZE_L)) % tl.cdiv(M, BLOCK_SIZE_N)
    block_l = pid % tl.cdiv(L, BLOCK_SIZE_L)
    
    # Compute actual n and m indices
    n_idx = batch_idx // M
    m_idx = batch_idx % M
    
    # Create offset arrays for M and L dimensions
    offsets_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_l = block_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Create mask for valid indices
    mask_m = offsets_m < M
    mask_l = offsets_l < L
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A has shape (N, M, K), so we need to get A[n_idx, offsets_m, k_offset]
        a_offsets = (
            n_idx * stride_a0 + 
            offsets_m[:, None] * stride_a1 + 
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_a2
        )
        a_mask = (
            (offsets_m[:, None] < M) & 
            ((k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < K)
        )
        a = tl.load(A + a_offsets, mask=a_mask, other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_offsets = (
            (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_b0 + 
            offsets_l[None, :] * stride_b1
        )
        b_mask = (
            ((k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < K) &
            (offsets_l[None, :] < L)
        )
        b = tl.load(B + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result: C has shape (N, M, L)
    c_offsets = (
        n_idx * stride_c0 + 
        offsets_m[:, None] * stride_c1 + 
        offsets_l[None, :] * stride_c2
    )
    c_mask = (mask_m[:, None] & mask_l[None, :])
    
    # Convert to output dtype
    acc = acc.to(C.dtype.element_ty)
    tl.store(C + c_offsets, acc, mask=c_mask)


def triton_matmul_3d(A, B):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K)
        B (torch.Tensor): Input matrix of shape (K, L)
    
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2, "Incompatible dimensions"
    
    # Create output tensor
    C = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    # Set up block sizes for performance
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_L = 128
    
    # Compute grid size
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_l = (L + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    num_programs = N * num_blocks_m * num_blocks_l
    
    # Compute strides
    stride_a0, stride_a1, stride_a2 = A.stride()
    stride_b0, stride_b1 = B.stride()
    stride_c0, stride_c1, stride_c2 = C.stride()
    
    # Launch kernel
    matmul_3d_kernel[
        num_programs,
    ](
        A, B, C,
        N, M, K, L,
        stride_a0, stride_a1, stride_a2,
        stride_b0, stride_b1,
        stride_c0, stride_c1, stride_c2,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        GROUP_SIZE_M=8,
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

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        return triton_matmul_3d(A, B)