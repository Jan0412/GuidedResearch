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
    # Matrix multiplication kernel using tiling
    # Program ID represents block in M x N grid
    pid = tl.program_id(axis=0)
    
    # Number of programs in M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped scheduling for better cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets for M and N dimensions
    offsets_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointer arithmetic for A and B matrices
    A_ptrs = A + (offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    B_ptrs = B + (offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
    
    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tile from A and B
        a = tl.load(A_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(B_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to appropriate type and store result
    C_block_out = accumulator.to(tl.float32)
    
    # Store result tile
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C + (offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn)
    mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(C_ptrs, C_block_out, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using a custom Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
        
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure tensors are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    
    M, K = A.shape
    K, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    # Set block sizes - tuned for large K dimension
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 256
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
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
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)
            
        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)