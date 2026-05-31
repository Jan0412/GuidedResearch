import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix A
    b_ptr,  # Pointer to second input matrix B
    c_ptr,  # Pointer to output matrix C
    M,      # Number of rows in A and C
    N,      # Number of columns in B and C
    K,      # Number of columns in A and rows in B
    stride_am, stride_ak,  # Stride of matrix A
    stride_bk, stride_bn,  # Stride of matrix B
    stride_cm, stride_cn,  # Stride of matrix C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    remaining = pid % num_pid_in_group
    tile_id = remaining % num_pid_n
    if tile_id < GROUP_M:
        pid_m = first_pid_m + tile_id
    else:
        pid_m = first_pid_m + (tile_id - GROUP_M) // (num_pid_n - GROUP_M)
    pid_n = remaining // num_pid_n
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a acc type (float32) to avoid overflow.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        # Load the next block of A and B, moving them to this GPU if needed.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # Compute the partial product.
        acc += tl.dot(a, b, allow_tf32=True)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with the accumulator.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using a Triton kernel.
    """
    # Ensure inputs are contiguous and on the same device
    a = a.contiguous()
    b = b.contiguous()
    
    # Check dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Matrix dimensions mismatch: {M}x{K} and {K_}x{N}"
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Configure kernel parameters
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8
    
    # Set the accumulator type to float32 for better numerical stability
    ACC_TYPE = tl.float32
    
    # Calculate grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        ACC_TYPE=ACC_TYPE
    )
    
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)
            
        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)