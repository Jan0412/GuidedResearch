import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, k, n,
    stride_ab, stride_ak, stride_am,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Get the batch index this program instance handles
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Split-K dimension: we'll accumulate partial results across multiple kernels
    pid_k = tl.program_id(3)
    
    # Compute the starting points for A and B
    # A[pid_b, pid_m, :] and B[pid_b, :, pid_n]
    # We need to compute C[pid_b, pid_m, pid_n] = sum_{l=0}^{k-1} A[pid_b, pid_m, l] * B[pid_b, l, pid_n]
    
    # Calculate the block offsets for A (m x k)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Calculate the block offsets for B (k x n)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for the rows
    amask = rm < m
    bnmask = rn < n
    kmask = rk < k
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension in chunks of BLOCK_SIZE_K
    # Since we're using split-K, each program handles a slice of the K dimension
    for k_idx in range(0, k, BLOCK_SIZE_K):
        # Adjust k_idx for split-K (we only want to process one slice per kernel)
        # But in this implementation, we use SPLIT_K to control how many kernels handle each K-block
        # So we'll use pid_k to determine which K-block we're processing
        
        # Calculate actual k offset
        actual_k = k_idx + pid_k * BLOCK_SIZE_K
        
        # Check if this k offset is within bounds
        if actual_k >= k:
            continue
            
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = A_ptr + pid_b * stride_ab + rm[:, None] * stride_am + (actual_k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = B_ptr + pid_b * stride_bb + (actual_k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=kmask[:, None], other=0.0)
        
        # Matrix multiply
        acc += tl.dot(a, b)
    
    # Accumulate across split-K blocks if needed (but here we're doing a single pass per kernel)
    # For simplicity, we assume SPLIT_K=1 in this basic implementation
    
    # Store result
    c_ptrs = C_ptr + pid_b * stride_cb + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = amask[:, None] & bnmask[None, :]
    
    # Convert to output type
    acc = acc.to(C_ptr.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for batched matrix multiplication.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
        
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    # Strides for tensor layouts
    stride_ab, stride_ak, stride_am = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()
    
    # Define block sizes for the GEMM kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    SPLIT_K = 1  # For simplicity, no split-K in this implementation
    
    # Grid dimensions: (batch_size, num_blocks_m, num_blocks_n, SPLIT_K)
    grid = (
        batch_size,
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
        SPLIT_K,
    )
    
    # Launch the kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        stride_ab, stride_ak, stride_am,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using a custom Triton kernel.
        
        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).
            
        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)