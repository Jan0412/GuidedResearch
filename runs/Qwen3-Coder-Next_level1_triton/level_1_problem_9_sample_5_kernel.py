import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    am_mask = offsets_am[:, None] < M
    bn_mask = offsets_bn[None, :] < N
    bk_mask = offsets_k[None, :] < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A block
        a_offset = (offsets_am[:, None] * stride_am + 
                   (k * BLOCK_SIZE_K + offsets_k[None, :]) * stride_ak)
        a = tl.load(A + a_offset, mask=am_mask & (offsets_k[None, :] < K), other=0.0)
        
        # Load B block
        b_offset = ((k * BLOCK_SIZE_K + offsets_k[:, None]) * stride_bk + 
                   offsets_bn[None, :] * stride_bn)
        b = tl.load(B + b_offset, mask=bk_mask & (offsets_k[:, None] < K), other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Convert to output type and store
    C_block = accumulator.to(tl.float32)
    
    # Store result
    c_offset = (offsets_am[:, None] * stride_cm + 
               offsets_bn[None, :] * stride_cn)
    c_mask = am_mask & bn_mask
    tl.store(C + c_offset, C_block, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    Optimized for tall/skinny matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions mismatch: A.shape={A.shape}, B.shape={B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure block sizes for tall/skinny matrices
    # For M >> N case (tall matrix): use larger BLOCK_SIZE_M
    # For N >> M case (skinny matrix): use larger BLOCK_SIZE_N
    
    if M > N:
        # Tall matrix case: M >> N
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    else:
        # Skinny matrix case: N >> M
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 16
    
    # Ensure BLOCK_SIZE_K divides K
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, K)
    if K % BLOCK_SIZE_K != 0:
        # Adjust to nearest divisor
        for candidate in [8, 16, 32]:
            if K % candidate == 0 and candidate <= BLOCK_SIZE_K:
                BLOCK_SIZE_K = candidate
                break
    
    # Group size for better occupancy
    GROUP_SIZE_M = 4
    
    # Calculate grid dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (num_pid_m * num_pid_n,)
    
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
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    Optimized for tall and skinny matrix cases.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return triton_matmul(A, B)