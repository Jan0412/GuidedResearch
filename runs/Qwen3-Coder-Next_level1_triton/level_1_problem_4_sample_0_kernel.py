import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, output_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_bb,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    
    # Create offsets for the output matrix C (M x 1)
    # Since N=1 in our case, we only need to handle M dimension
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M
    
    # Initialize accumulator for dot product
    output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load block of A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        a_mask = (mask_m[:, None] & (k_offsets < K)[None, :])
        A_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0)
        )
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Load block of B: (BLOCK_SIZE_K, 1)
        b_mask = (k_offsets < K) & (tl.arange(0, 1) < 1)
        B_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, 1),
            strides=(stride_bk, stride_bb),
            offsets=(k, 0),
            block_shape=(BLOCK_SIZE_K, 1),
            order=(1, 0)
        )
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute partial dot product
        # a has shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # b has shape (BLOCK_SIZE_K, 1)
        # Result should be (BLOCK_SIZE_M, 1)
        output += tl.sum(a * b, axis=1)  # Sum over K dimension
    
    # Store result
    output_ptr_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, 1),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, 1),
        order=(1, 0)
    )
    tl.store(output_ptr_block, output[:, None], boundary_check=(0, 1))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K, 1)
        
    Returns:
        Output vector of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 2 and B.dim() == 2, "A must be 2D matrix and B must be 2D vector"
    assert A.shape[1] == B.shape[0], "A.shape[1] must equal B.shape[0]"
    
    M, K = A.shape
    K_b, N = B.shape
    assert N == 1, "B must be a column vector"
    
    # Ensure contiguous memory
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    output = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_K = 256
    BLOCK_SIZE_N = 1
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, output,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)