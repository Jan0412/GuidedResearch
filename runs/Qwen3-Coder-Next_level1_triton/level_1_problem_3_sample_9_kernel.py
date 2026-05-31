import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    batch_size, m, k, n,
    # Strides
    stride_az, stride_am, stride_ak,
    stride_bz, stride_bk, stride_bn,
    stride_cz, stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for batched matrix multiplication (C = A * B).
    Each batch is processed independently.
    """
    # Get program IDs
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Set the starting positions for this program
    # Note: We're using a grid of (num_blocks, batch_size), so pid = block_id within a batch
    num_blocks_m = tl.cdiv(m, BLOCK_SIZE_M)
    block_id = pid
    pid_m = block_id // num_blocks_m
    pid_n = block_id % num_blocks_m
    
    # Reassign to ensure proper tiling
    # We use a grid where each block covers a (BLOCK_SIZE_M x BLOCK_SIZE_N) tile of C
    num_blocks_n = tl.cdiv(n, BLOCK_SIZE_N)
    block_id = pid
    pid_m = (block_id // num_blocks_n) // num_blocks_m
    pid_n = (block_id // num_blocks_n) % num_blocks_n
    
    # Compute the starting offsets for the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure we don't go out of bounds
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), m)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), n)
    
    # Create masks for bounds checking
    rm_mask = rm < m
    rn_mask = rn < n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, k, BLOCK_SIZE_K):
        # Compute k offsets
        rk = k_start + tl.arange(0, BLOCK_SIZE_K)
        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), k)
        rk_mask = rk < k
        
        # Compute indices for A: [batch_id, rm, rk]
        a_ptrs = A_ptr + batch_id * stride_az + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rm_mask[:, None] & rk_mask[None, :], other=0.0)
        
        # Compute indices for B: [batch_id, rk, rn]
        b_ptrs = B_ptr + batch_id * stride_bz + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=rk_mask[:, None] & rn_mask[None, :], other=0.0)
        
        # Accumulate the dot product
        acc += tl.dot(a, b)
    
    # Convert accumulator to float16 if needed, but we're keeping it in fp32 for precision
    # Store the result to C: [batch_id, rm, rn]
    c_ptrs = C_ptr + batch_id * stride_cz + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = rm_mask[:, None] & rn_mask[None, :]
    
    # Store result (convert back to input dtype)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based batched matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match."
    assert A.shape[2] == B.shape[1], "Inner dimensions must match for matrix multiplication."
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    # Calculate strides
    stride_az = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)
    stride_bz = B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)
    stride_cz = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Grid: (num_blocks_per_batch, batch_size)
    # num_blocks_per_batch = ceil(m / BLOCK_SIZE_M) * ceil(n / BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(n, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n, batch_size)
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        stride_az, stride_am, stride_ak,
        stride_bz, stride_bk, stride_bn,
        stride_cz, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using custom Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)