import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    num_features,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * num_features
    out_ptr += row_ptr * num_features
    
    max_val = float('-inf')
    sum_val = 0.0
    
    # First pass: compute max and sum using online algorithm
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        
        local_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, local_max)
        
        scale = tl.exp(max_val - new_max)
        sum_val = sum_val * scale + tl.sum(tl.exp(x - new_max), axis=0)
        
        max_val = new_max
    
    # Second pass: compute output
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        
        out = tl.exp(x - max_val) / sum_val
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    softmax_kernel[grid](x, out, num_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)