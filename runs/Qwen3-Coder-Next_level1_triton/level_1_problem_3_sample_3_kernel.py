import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Adjust strides for batch index
    A_ptr += batch_idx * stride_ab
    B_ptr += batch_idx * stride_bb
    C_ptr += batch_idx * stride_cb
    
    # Compute offsets for M dimension (rows of C)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m_adjusted = pid_m % num_blocks_m
    
    # Compute offsets for N dimension (cols of C)
    pid_n_adjusted = pid_n % tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped GEMM: better cache utilization by having adjacent threads work on same row of C
    # See Triton tutorial for more details
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Grouping logic
    group_size_m = min(GROUP_SIZE_M, num_blocks_m)
    pid_m_group = pid_m // group_size_m
    pid_m_in_group = pid_m % group_size_m
    pid_m = pid_m_group * group_size_m + pid_m_in_group
    
    # Compute starting offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    km_mask = tl.arange(0, BLOCK_SIZE_K) < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_idx in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_idx + tl.arange(0, BLOCK_SIZE_K)
        
        # Load tile from A: shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=m_mask[:, None] & km_mask[None, :],
            other=0.0
        )
        
        # Load tile from B: shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_tile = tl.load(
            B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
            mask=km_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiply
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
    
    # Convert to output type and store result
    c_tile = acc.to(tl.float32)
    
    # Store result
    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        c_tile,
        mask=m_mask[:, None] & n_mask[None, :]
    )


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
        
    Returns:
        Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=torch.float32, device=A.device)
    
    # Strides for batched operations
    stride_ab = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)
    
    stride_bb = B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)
    
    stride_cb = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)
    
    # Define block sizes for performance optimization
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid dimensions
    grid = (
        tl.cdiv(m, BLOCK_SIZE_M),
        tl.cdiv(n, BLOCK_SIZE_N),
        batch_size
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.
        """
        return triton_bmm(A, B)