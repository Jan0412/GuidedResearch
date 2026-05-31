import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    
    # L2 cache optimization: Grouping blocks in M dimension
    num_pid_in_group = GROUP_SIZE_M
    group_id = pid // (num_pid_n * num_pid_in_group)
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_in_group, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid % (num_pid_n * num_pid_in_group)) // num_pid_n
    pid_n = (pid % (num_pid_n * num_pid_in_group)) % num_pid_n

    # Batch ID is the 3rd dimension of the grid
    pid_batch = tl.program_id(1)

    # Pointers to the start of the current batch's matrices
    a_ptr += pid_batch * stride_ab
    b_ptr += pid_batch * stride_bb
    c_ptr += pid_batch * stride_cb

    # Offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A and B
    # A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k_offset in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (k_offset * BLOCK_SIZE_K + offs_k[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offset * BLOCK_SIZE_K + offs_k[:, None] < k) & (offs_bn[None, :] < n), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Pointers for C
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    # Store the result
    tl.store(c_ptrs, accumulator, mask=(offs_am[:, None] < m) & (offs_bn[None, :] < n) )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # Ensure tensors are contiguous and on CUDA
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, k_b, n = B.shape
    assert k == k_b, "K dimensions must match"

    # Output tensor
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)

    # Strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid: (M_blocks * N_blocks, batch_size)
    # We flatten the M and N blocks into the first dimension for L2 optimization
    grid = (
        tl.cdiv(m, BLOCK_SIZE_M) * tl.cdiv(n, BLOCK_SIZE_N),
        batch_size
    )

    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return C

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)