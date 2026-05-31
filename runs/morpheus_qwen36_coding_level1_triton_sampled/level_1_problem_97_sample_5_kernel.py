import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sdpa_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    scale,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = tl.program_id(1)
    qid = tl.program_id(2)

    # Compute base offsets for Q, K, V, Out
    # Assuming contiguous tensors with shape (B, H, L, D)
    # B is not needed explicitly as pid covers it
    q_off = pid * H * L * D + hid * L * D + qid * D
    k_off_base = pid * H * L * D + hid * L * D
    v_off_base = pid * H * L * D + hid * L * D
    out_off = pid * H * L * D + hid * L * D + qid * D

    # Load Q block (1 x BLOCK_D)
    q_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < D
    q = tl.load(Q_ptr + q_off + q_offs, mask=q_mask, other=0.0, dtype=tl.float32)

    # Initialize softmax statistics
    m = tl.full((BLOCK_D,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # SRAM buffer for scores (1 x BLOCK_K)
    scores = tl.empty(BLOCK_K, tl.float32)

    # First pass: Compute QK^T and update softmax stats
    for k_block in range(0, L, BLOCK_K):
        k_off = k_off_base + k_block * D
        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < (L - k_block)
        
        # Load K block (BLOCK_K x BLOCK_D)
        k = tl.load(K_ptr + k_off + tl.arange(0, BLOCK_D)[None, :] + k_offs[:, None] * D, 
                    mask=k_mask[:, None], other=0.0, dtype=tl.float32)
        
        # Transpose K for matmul: (BLOCK_K, BLOCK_D) -> (BLOCK_D, BLOCK_K)
        k_t = tl.trans(k)
        
        # Compute QK^T: (1, BLOCK_D) @ (BLOCK_D, BLOCK_K) -> (1, BLOCK_K)
        qk = tl.dot(q[None, :], k_t)
        qk = qk * scale
        
        # Update softmax stats
        m_new = tl.maximum(m, tl.max(qk, axis=1))
        l_new = tl.exp(m - m_new) * l + tl.sum(tl.exp(qk - m_new), axis=1)
        
        m = m_new
        l = l_new
        
        # Store qk in SRAM buffer
        tl.store(scores + tl.arange(0, BLOCK_K), qk[0, :])

    # Normalize l
    l_inv = 1.0 / l
    m = m + tl.log(l)

    # Second pass: Compute attention @ V
    out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    for k_block in range(0, L, BLOCK_K):
        k_off = k_off_base + k_block * D
        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < (L - k_block)
        
        # Load scores from SRAM
        qk = tl.load(scores + tl.arange(0, BLOCK_K), mask=k_mask, other=0.0)
        
        # Compute attention weights
        weights = tl.exp(qk - m) * l_inv
        
        # Load V block (BLOCK_K x BLOCK_D)
        v = tl.load(V_ptr + k_off + tl.arange(0, BLOCK_D)[None, :] + k_offs[:, None] * D, 
                    mask=k_mask[:, None], other=0.0, dtype=tl.float32)
        
        # Accumulate attention @ V
        out_acc += tl.dot(weights[None, :], v)

    # Store result
    tl.store(Out_ptr + out_off + tl.arange(0, BLOCK_D), out_acc, mask=q_mask)


def triton_fused_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    
    B, H, L, D = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape
    
    Out = torch.empty_like(Q, dtype=torch.float32)
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_D = 128
    BLOCK_K = 128
    
    grid = (B, H, L)
    
    fused_sdpa_kernel[grid](
        Q, K, V, Out,
        scale,
        H, L, D,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_fused_sdpa(Q, K, V)