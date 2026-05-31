import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to the block of C it should compute
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # a_ptr is offset to the start of the current M-block
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance the pointers to the next block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_3d(A, B):
    # A: (N, M, K), B: (K, L) -> Out: (N, M, L)
    batch_size, M, K = A.shape
    K_B, L = B.shape
    assert K == K_B, "K dimensions must match"

    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Output tensor
    C = torch.empty((batch_size, M, L), device=A.device, dtype=A.dtype)

    # Strides
    stride_am = K
    stride_ak = 1
    stride_bk = L
    stride_bn = 1
    stride_cm = L
    stride_cn = 1

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid for the kernel: (Batch, M_blocks, L_blocks)
    # We iterate through the batch dimension in a Python loop for simplicity 
    # or we could incorporate it into the kernel. Let's incorporate it into the kernel 
    # by adjusting pointers to handle the 3D structure.
    
    # To make the kernel truly 3D, we add a batch dimension to the grid and pointers.
    # However, we can reuse the 2D kernel by iterating over the batch or modifying the kernel.
    # Let's modify the kernel to handle the batch dimension.
    
    return _triton_matmul_3d_batched(A, B, C, batch_size, M, L, K, 
                                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

@triton.jit
def matmul_kernel_batched(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Offset pointers by batch index
    a_ptr_batch = a_ptr + pid_b * stride_ab
    c_ptr_batch = c_ptr + pid_b * stride_cb

    a_ptrs = a_ptr_batch + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr_batch + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def _triton_matmul_3d_batched(A, B, C, batch_size, M, L, K, 
                              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    stride_ab = A.stride(0)
    stride_cb = C.stride(0)
    
    grid = (batch_size, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(L, BLOCK_SIZE_N))
    
    matmul_kernel_batched[grid](
        A, B, C,
        M, L, K,
        stride_ab, stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        # Ensure inputs are on GPU and FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_matmul_3d(A, B)