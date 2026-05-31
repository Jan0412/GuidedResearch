import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A [b, i, j, l]
    B_ptr,  # Pointer to matrix B [l, k]
    C_ptr,  # Pointer to output tensor C [b, i, j, k]
    B_stride_l, B_stride_k,  # Strides for B tensor
    C_stride_b, C_stride_i, C_stride_j, C_stride_k,  # Strides for C tensor
    # Dimensions
    b, i, j, l, k,
    # Block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,  # Block size for k dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for l dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for l dimension (same as above)
    # Number of warps
    num_warps: tl.constexpr,
):
    # Program IDs for batch, i, j dimensions (all are sequential)
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    
    # Offset to the current [b, i, j] slice in A
    A_offset = pid_b * (i * j * l) + pid_i * (j * l) + pid_j * l
    C_offset = pid_b * (i * j * k * C_stride_b) + pid_i * (j * k * C_stride_i) + pid_j * (k * C_stride_j)
    
    # Initialize the accumulator block
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over l dimension in blocks
    for start_l in range(0, l, BLOCK_SIZE_K):
        # Compute l offsets
        l_offsets = start_l + tl.arange(0, BLOCK_SIZE_K)
        l_mask = l_offsets < l
        
        # Load a block of A: shape [BLOCK_SIZE_K]
        a_ptrs = A_ptr + A_offset + l_offsets
        a = tl.load(a_ptrs, mask=l_mask, other=0.0)
        
        # Load a block of B: shape [BLOCK_SIZE_K, BLOCK_SIZE_M]
        k_offsets = tl.arange(0, BLOCK_SIZE_M)
        b_ptrs = B_ptr + l_offsets[:, None] * B_stride_l + k_offsets[None, :] * B_stride_k
        b_block = tl.load(b_ptrs, mask=(l_mask[:, None] & (k_offsets < k)[None, :]), other=0.0)
        
        # Compute partial matmul: acc += a * b
        acc += tl.sum(a[:, None] * b_block, axis=0)
    
    # Store the result to C
    c_offsets = tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_offsets < k
    tl.store(C_ptr + C_offset + c_offsets, acc.to(tl.float32), mask=c_mask)


def triton_matmul_4d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    Optimized with Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 4 and B.dim() == 2, "A must be 4D and B must be 2D."
    assert A.shape[3] == B.shape[0], "Last dim of A must match first dim of B."
    
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Create output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tuned for FP32)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32  # Actually same as BLOCK_SIZE_K
    
    # Determine grid dimensions
    grid = (b, i, j)
    
    # Launch kernel
    matmul_4d_kernel[grid](
        A, B, C,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        b, i, j, l, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 4D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d(A, B)