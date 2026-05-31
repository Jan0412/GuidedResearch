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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr = 3, USE_TMA: tl.constexpr = False
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute starting offsets for A, B, and C matrices
    # A: (batch, m, k) -> offset = pid_batch * stride_ab + pid_m * stride_am
    # B: (batch, k, n) -> offset = pid_batch * stride_bb + pid_n * stride_bn
    # C: (batch, m, n) -> offset = pid_batch * stride_cb + pid_m * stride_cm + pid_n * stride_cn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute pointers for A and B
    a_offset = pid_batch * stride_ab + pid_m * stride_am
    b_offset = pid_batch * stride_bb + pid_n * stride_bn
    
    # Iterate over K dimension in blocks
    for k_start in range(0, k, BLOCK_K):
        # Compute offsets for A and B in K dimension
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        # Load A block: shape (BLOCK_M, BLOCK_K)
        a_mask = (pid_m * stride_am + tl.arange(0, BLOCK_M)[:, None] < m * stride_am) & \
                 (k_offsets[None, :] < k)
        a_block = tl.load(
            A_ptr + a_offset + tl.arange(0, BLOCK_M)[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=a_mask,
            other=0.0
        )
        
        # Load B block: shape (BLOCK_K, BLOCK_N)
        b_mask = (k_offsets[:, None] < k) & \
                 (pid_n * stride_bn + tl.arange(0, BLOCK_N)[None, :] < n * stride_bn)
        b_block = tl.load(
            B_ptr + b_offset + k_offsets[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn,
            mask=b_mask,
            other=0.0
        )
        
        # Accumulate matrix multiply
        accumulator = tl.dot(a_block, b_block, accumulator, input_precision="tf32" if A_ptr.dtype == tl.float32 else "ieee")
    
    # Store result
    c_offset = pid_batch * stride_cb + pid_m * stride_cm + pid_n * stride_cn
    c_mask = (pid_m * stride_cm + tl.arange(0, BLOCK_M)[:, None] < m * stride_cm) & \
             (pid_n * stride_bn + tl.arange(0, BLOCK_N)[None, :] < n * stride_bn)
    
    # Cast accumulator to output type if needed
    if C_ptr.dtype == tl.float32:
        c_block = accumulator.to(tl.float32)
    else:
        c_block = accumulator.to(C_ptr.dtype)
    
    tl.store(
        C_ptr + c_offset + tl.arange(0, BLOCK_M)[:, None] * stride_cm + tl.arange(0, BLOCK_N)[None, :] * stride_cn,
        c_block,
        mask=c_mask
    )


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for batched matrix multiplication.
    Assumes A: (batch_size, m, k), B: (batch_size, k, n)
    Returns C: (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match."
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    # Set block sizes for the kernel
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Define grid dimensions
    grid = (batch_size, triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        A.stride(0), A.stride(2), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(2), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using custom Triton kernel.
        """
        return triton_bmm(A, B)