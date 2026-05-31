import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map program IDs to block indices
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # L2 cache optimization: group blocks in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    # Swizzle pid_m to improve L2 cache hit rate
    group_id = pid_n
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid // num_pid_m) % group_size_m

    # Compute offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks for the current batch
    a_ptr = A_ptr + pid_batch * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = B_ptr + pid_batch * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a = tl.load(a_ptr, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptr, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        
        # Matrix multiply-accumulate
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result
    c_ptr = C_ptr + pid_batch * stride_cb + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=mask)

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, M, K = A.shape
    _, K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match"
    
    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid: (M_blocks * N_blocks, N_blocks, Batch)
    # We use a 1D pid for M and N combined for the first dimension as per the swizzling logic
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    grid = (num_pid_m * num_pid_n, num_pid_n, batch_size)
    
    bmm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)