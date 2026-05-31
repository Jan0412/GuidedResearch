import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    base_offset = pid * num_elements
    x_ptr += base_offset
    out_ptr += base_offset
    
    # Compute mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    n = 0
    
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.reduce(x, axis=0, fn=tl.sum)
        sum_x2 += tl.reduce(x * x, axis=0, fn=tl.sum)
        n += tl.reduce(mask, axis=0, fn=tl.sum)
    
    mean = sum_x / n
    var = sum_x2 / n - mean * mean
    
    # Normalize and store
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) / tl.sqrt(var + 1e-5)
        out = out * w + b
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous().view(-1)
    bias = bias.contiguous().view(-1)
    
    batch_size = x.shape[0]
    num_elements = x.numel() // batch_size
    
    out = torch.empty_like(x)
    
    grid = (batch_size,)
    BLOCK_SIZE = 128
    
    layernorm_kernel[grid](x, out, weight, bias, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_layernorm(x, self.ln.weight, self.ln.bias)