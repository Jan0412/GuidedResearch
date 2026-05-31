import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triangular_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Map pid to block coordinates (i, j)
    pid_i = pid // num_pid_n
    pid_j = pid % num_pid_n
    
    # Skip lower triangular blocks
    if pid_i > pid_j:
        return
    
    i_start = pid_i * BLOCK_M
    j_start = pid_j * BLOCK_N
    
    # Offsets for rows and columns within the block
    offs_i = tl.arange(0, BLOCK_M)
    offs_j = tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Determine valid k range for this block
    # k must be >= i for A (upper triangular)
    # k must be <= j for B (upper triangular)
    # So k ranges from i_start to j_start + BLOCK_N - 1
    k_start = i_start
    k_end = j_start + BLOCK_N
    
    # Number of k blocks to process
    num_k_blocks = tl.cdiv(k_end - k_start, BLOCK_K)
    
    for k_pid in range(num_k_blocks):
        k_block_start = k_start + k_pid * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)
        
        # Load tile from A
        # A is upper triangular: A[i, k] is valid only if k >= i
        # Mask for A: offs_k >= (i_start + offs_i)[:, None]
        mask_A = (k_block_start + offs_k)[None, :] >= (i_start + offs_i)[:, None]
        A_tile = tl.load(A_ptr + (i_start + offs_i)[:, None] * N + (k_block_start + offs_k)[None, :], mask=mask_A, other=0.0)
        
        # Load tile from B
        # B is upper triangular: B[k, j] is valid only if k <= j
        # Mask for B: (k_block_start + offs_k)[:, None] <= (j_start + offs_j)[None, :]
        mask_B = (k_block_start + offs_k)[:, None] <= (j_start + offs_j)[None, :]
        B_tile = tl.load(B_ptr + (k_block_start + offs_k)[:, None] * N + (j_start + offs_j)[None, :], mask=mask_B, other=0.0)
        
        # Matrix multiply
        acc += tl.dot(A_tile, B_tile)
    
    # Compute output mask
    # C[i, j] is valid only if i <= j
    mask_out = (i_start + offs_i)[:, None] <= (j_start + offs_j)[None, :]
    
    # Store result
    # We need to write zeros for the lower triangular part within the block
    # Triton store with mask only writes where mask is true, so we do two stores
    tl.store(C_ptr + (i_start + offs_i)[:, None] * N + (j_start + offs_j)[None, :], acc, mask=mask_out)
    tl.store(C_ptr + (i_start + offs_i)[:, None] * N + (j_start + offs_j)[None, :], 0.0, mask=~mask_out)


def triton_triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape == B.shape, "Inputs must have same shape."
    N = A.shape[0]
    assert A.shape == (N, N), "Inputs must be square matrices."
    
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty_like(A)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)
    
    triangular_matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_triangular_matmul(A, B)


N = 4096

def get_inputs():
    A = torch.triu(torch.rand(N, N, device='cuda'))
    B = torch.triu(torch.rand(N, N, device='cuda'))
    return [A, B]


def get_init_inputs():
    return []