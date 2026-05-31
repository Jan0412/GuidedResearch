import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_T_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets for A and B_T
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Masks for M and N dimensions
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K blocks
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_blocks):
        k_start = k * BLOCK_K
        # Mask for K dimension: valid only if k_start + offs_k < K
        mask_k = offs_k < (K - k_start)
        
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        # A is (M, K), row-major. stride_am = K, stride_ak = 1.
        a_offsets = offs_am[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        mask_a = mask_am[:, None] & mask_k[None, :]
        a = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        
        # Load B_T tile: shape (BLOCK_K, BLOCK_N)
        # B_T is (N, K), row-major. We want B_T[n, k] -> B[k, n].
        # We compute offsets to load B_T as (BLOCK_K, BLOCK_N).
        # B_T offset = n * K + k.
        # We need output shape (BLOCK_K, BLOCK_N), so row index is k, col index is n.
        # b_offsets[k, n] = n * K + k.
        b_offsets = offs_bn[None, :] * K + (k_start + offs_k)[:, None]
        mask_b = mask_k[:, None] & mask_bn[None, :]
        b = tl.load(B_T_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b)
        
    # Store C tile: shape (BLOCK_M, BLOCK_N)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = mask_am & mask_bn
    c_offsets = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(C_ptr + c_offsets, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_B, N = B.shape
    
    assert K == K_B, "Incompatible dimensions"
    
    # Transpose B to B_T for coalesced memory access
    # B_T shape: (N, K)
    B_T = B.T.contiguous()
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B_T.stride(0)
    stride_bn = B_T.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Grid calculation
    num_block_M = (M + BLOCK_M - 1) // BLOCK_M
    num_block_N = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_block_M * num_block_N,)
    
    matmul_kernel[grid](
        A, B_T, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


def get_inputs():
    M = 16384 * 2
    N = 16384 * 2
    K = 32 * 2
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []