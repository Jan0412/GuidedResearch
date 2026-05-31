import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    n_heads, seq_len, d_model,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Batch and Head indices
    batch_id = pid // n_heads
    head_id = pid % n_heads
    
    # Pointers to the start of Q, K, V for this batch and head
    q_ptr = Q + batch_id * stride_qb + head_id * stride_qh
    k_ptr = K + batch_id * stride_kb + head_id * stride_kh
    v_ptr = V + batch_id * stride_vb + head_id * stride_vh
    out_ptr = Out + batch_id * stride_ob + head_id * stride_oh
    
    # Offsets for the current block of Q
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, d_model)
    
    # Load Q block
    # Q shape: (seq_len, d_model)
    q = tl.load(q_ptr + rm[:, None] * stride_ql + rk[None, :] * stride_qd)
    
    # Initialize softmax statistics and output accumulator
    # Use float32 for precision as requested
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    
    # Scaling factor
    scale = 1.0 / (d_model ** 0.5)
    
    # Iterate over blocks of K and V
    for start_n in range(0, seq_len, BLOCK_N):
        # Load K block
        # K shape: (seq_len, d_model)
        k = tl.load(k_ptr + (start_n + rn)[:, None] * stride_kl + rk[None, :] * stride_kd)
        
        # Compute QK^T
        # q: (BLOCK_M, d_model), k: (BLOCK_N, d_model) -> qk: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q.to(tl.float16), tl.trans(k).to(tl.float16))
        qk = qk * scale
        
        # Online softmax
        m_curr = tl.max(qk, 1)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1)
        
        # Update running statistics
        m_next = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_curr - m_next)
        
        l_i = alpha * l_i + beta * l_curr
        
        # Load V block and update accumulator
        # V shape: (seq_len, d_model)
        v = tl.load(v_ptr + (start_n + rn)[:, None] * stride_vl + rk[None, :] * stride_vd)
        # weighted_v: (BLOCK_M, d_model)
        weighted_v = tl.dot(p.to(tl.float16), v.to(tl.float16))
        
        acc = acc * alpha[:, None] + weighted_v * beta[:, None]
        m_i = m_next

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store result
    out_offsets = rm[:, None] * stride_ol + rk[None, :] * stride_od
    tl.store(out_ptr + out_offsets, acc.to(Out.dtype.element_ty))

def triton_sdpa(Q, K, V):
    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    batch_size, num_heads, seq_len, d_model = Q.shape
    out = torch.empty_like(Q)
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid: (batch * heads, ceil(seq_len / BLOCK_M))
    grid = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_M))
    
    attention_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_heads, seq_len, d_model,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_sdpa(Q, K, V)