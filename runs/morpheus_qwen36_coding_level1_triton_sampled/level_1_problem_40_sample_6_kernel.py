import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_base = pid * N
    num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Pass 1: Compute mean
    sum_val = 0.0
    for i in tl.static_range(num_chunks):
        offset = offset_base + i * BLOCK_SIZE
        mask = tl.arange(0, BLOCK_SIZE) < N
        x_chunk = tl.load(x_ptr + offset, mask=mask, other=0.0)
        sum_val += tl.sum(x_chunk)
    mean = sum_val / N
    
    # Pass 2: Compute variance
    sum_sq = 0.0
    for i in tl.static_range(num_chunks):
        offset = offset_base + i * BLOCK_SIZE
        mask = tl.arange(0, BLOCK_SIZE) < N
        x_chunk = tl.load(x_ptr + offset, mask=mask, other=0.0)
        sum_sq += tl.sum((x_chunk - mean) ** 2)
    var = sum_sq / N
    
    # Pass 3: Normalize, scale, and shift
    inv_std = tl.rsqrt(var + eps)
    for i in tl.static_range(num_chunks):
        offset = offset_base + i * BLOCK_SIZE
        mask = tl.arange(0, BLOCK_SIZE) < N
        x_chunk = tl.load(x_ptr + offset, mask=mask, other=0.0)
        w_chunk = tl.load(weight_ptr + offset, mask=mask, other=0.0)
        b_chunk = tl.load(bias_ptr + offset, mask=mask, other=0.0)
        out_chunk = (x_chunk - mean) * inv_std * w_chunk + b_chunk
        tl.store(out_ptr + offset, out_chunk, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    B, N = x.shape[0], x.numel() // x.shape[0]
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 4096
    grid = lambda meta: (B,)
    
    layer_norm_kernel[grid](x, weight, bias, out, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        # Freeze parameters to match original behavior if needed, 
        # but we'll just use the learned parameters from self.ln
        self.weight = self.ln.weight
        self.bias = self.ln.bias
        self.eps = self.ln.eps
        self.normalized_shape = normalized_shape
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to (B, N) for the kernel
        original_shape = x.shape
        B = x.shape[0]
        N = x.numel() // B
        x_flat = x.view(B, N)
        
        out_flat = triton_layer_norm(x_flat, self.weight, self.bias, self.eps)
        return out_flat.view(original_shape)


def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    features = 64
    dim1 = 256
    dim2 = 256
    return [(features, dim1, dim2)]