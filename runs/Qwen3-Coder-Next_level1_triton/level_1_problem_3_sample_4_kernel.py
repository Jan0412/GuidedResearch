import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr = 4, num_warps: tl.constexpr = 8
):
    # Batch index
    batch_idx = tl.program_id(2)
    
    # Pointers for this batch
    A_ptr += batch_idx * stride_ab
    B_ptr += batch_idx * stride_bb
    C_ptr += batch_idx * stride_cb
    
    # Create program indices for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for M and N blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_idx in range(0, k, BLOCK_K):
        # Load block of A
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + 
                         (k_idx + offs_k)[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Load block of B
        b_ptrs = B_ptr + ((k_idx + offs_k)[:, None] * stride_bk + 
                         offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + 
                     offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Convert to output type and store
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
    
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match."
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    # Configure block sizes (tuned for FP32 on modern GPUs)
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(m, BLOCK_M),   # Number of blocks in M dimension
        triton.cdiv(n, BLOCK_N),   # Number of blocks in N dimension
        batch_size                 # Batch dimension
    )
    
    # Launch kernel
    batched_gemm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=4, num_warps=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton batched GEMM kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.
        """
        return triton_bmm(A, B)