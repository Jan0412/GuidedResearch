import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,  # Pointer to input matrix A (batch_size, m, k)
    B_ptr,  # Pointer to input matrix B (batch_size, k, n)
    C_ptr,  # Pointer to output matrix C (batch_size, m, n)
    batch_size,
    m, n, k,
    stride_ab, stride_am, stride_ak,  # Strides for A
    stride_bb, stride_bk, stride_bn,  # Strides for B
    stride_cb, stride_cm, stride_cn,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(2)
    # Compute batch offsets
    A_ptr_batch = A_ptr + batch_idx * stride_ab
    B_ptr_batch = B_ptr + batch_idx * stride_bb
    C_ptr_batch = C_ptr + batch_idx * stride_cb

    # Get block start indices for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute number of blocks in M dimension
    num_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_n = tl.cdiv(n, BLOCK_SIZE_N)

    # Grouping logic to improve cache usage
    # Group size determines how many blocks in M dimension each thread block handles
    num_blocks_in_group = GROUP_SIZE_M * num_n
    group_id = pid_m // GROUP_SIZE_M
    first_m_block = group_id * GROUP_SIZE_M
    group_size_m = min(num_m - first_m_block, GROUP_SIZE_M)
    pid_m = first_m_block + (pid_m % group_size_m)

    # Compute offsets for M and N dimensions
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create masks for bounds checking
    m_mask = m_offsets < m
    n_mask = n_offsets < n
    mn_mask = m_mask[:, None] & n_mask[None, :]

    # Initialize accumulator for C
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_offset in range(0, k, BLOCK_SIZE_K):
        # Compute K offsets
        k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < k

        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(A_ptr_batch + a_offsets, mask=k_mask[None, :], other=0.0)

        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b = tl.load(B_ptr_batch + b_offsets, mask=k_mask[:, None], other=0.0)

        # Accumulate matrix multiplication
        acc = acc + tl.dot(a, b, out_dtype=tl.float32)

    # Convert accumulator to FP32 (if needed)
    c = acc.to(tl.float32)

    # Store result C
    c_offsets = m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptr_batch + c_offsets, c, mask=mn_mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
    
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
        batch_size,
    )
    
    # Calculate strides
    stride_ab = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)
    stride_bb = B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)
    stride_cb = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)
    
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
    Optimized version of the batched matrix multiplication model using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)