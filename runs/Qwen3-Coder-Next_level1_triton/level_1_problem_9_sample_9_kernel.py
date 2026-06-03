import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Matrix multiplication kernel for FP32 tensors
    # Program ID represents the block in the M dimension
    pid = tl.program_id(0)
    # Number of programs in the M dimension
    num_program_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs in the N dimension
    num_program_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute the group of programs that should be processed together
    # Grouped by M to improve cache reuse
    num_groups = num_program_m // GROUP_SIZE_M
    group_id = pid // GROUP_SIZE_M
    if group_id < num_groups:
        first_m = group_id * GROUP_SIZE_M
        last_m = (group_id + 1) * GROUP_SIZE_M
    else:
        first_m = num_groups * GROUP_SIZE_M
        last_m = num_program_m
    
    # Program ID within the group
    pid_m = first_m + (pid % GROUP_SIZE_M)
    pid_n = (pid % num_program_n)
    
    # Compute starting offsets for blocks
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundary conditions
    mask_m = off_m < M
    mask_n = off_n < N
    
    # Initialize accumulator for C matrix
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = off_k < K
        
        # Load block from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load block from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Convert accumulator to float16 if needed, but for now keep as float32
    c = acc.to(tl.float32)
    
    # Store result to C
    tl.store(
        C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based matrix multiplication optimized for tall and skinny matrices.
    Handles both (M, K) @ (K, N) -> (M, N) and cases where M >> N or N >> M.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions must match for multiplication"
    
    # Allocate output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes optimized for tall and skinny matrices
    # For M >> N case, we want larger blocks in M dimension
    # For N >> M case, we want larger blocks in N dimension
    # Default configuration optimized for M=32768, N=32 case
    BLOCK_SIZE_M = 128 if M > N else 64
    BLOCK_SIZE_N = 64 if N > M else 128
    BLOCK_SIZE_K = 32
    
    # Ensure block sizes are valid
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, M)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, N)
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, K)
    
    # Grid dimensions: number of blocks in M and N dimensions
    grid = ((M // BLOCK_SIZE_M + 1) * (N // BLOCK_SIZE_N + 1),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    Specialized for tall and skinny matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication using Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.
        
        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return triton_matmul(A, B)