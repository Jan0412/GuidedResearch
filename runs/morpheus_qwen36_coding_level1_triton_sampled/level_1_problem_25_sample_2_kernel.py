import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # Use expm1 for better numerical stability when x is large negative
    # sigmoid(x) = 1 / (1 + exp(-x))
    # For large positive x, exp(-x) is small, so sigmoid(x) ≈ 1
    # For large negative x, exp(-x) is large, so sigmoid(x) ≈ 0
    # We can compute directly: 1.0 / (1.0 + tl.exp(-x))
    # However, for very large x, exp(-x) might underflow to 0, which is fine.
    # For very negative x, exp(-x) might overflow. To handle this, we can use:
    # if x > 0: sigmoid(x) = 1.0 / (1.0 + exp(-x))
    # else: sigmoid(x) = exp(x) / (1.0 + exp(x))
    # But for simplicity and given Swish is often used with moderate values, 
    # we can use the stable formulation:
    # sigmoid(x) = 0.5 * (1.0 + tanh(0.5 * x))
    # However, tanh might be slower. Alternatively, use the direct formula with care.
    # A common stable sigmoid implementation:
    # pos_mask = x > 0
    # neg_mask = ~pos_mask
    # exp_neg_x = tl.exp(-x)
    # sigmoid_pos = 1.0 / (1.0 + exp_neg_x)
    # exp_x = tl.exp(x)
    # sigmoid_neg = exp_x / (1.0 + exp_x)
    # sigmoid = tl.where(pos_mask, sigmoid_pos, sigmoid_neg)
    
    # For performance, we can use the direct formula if we assume inputs are not extreme.
    # But to be safe and accurate, let's use the stable version.
    
    pos_mask = x > 0.0
    neg_mask = ~pos_mask
    
    exp_neg_x = tl.exp(-x)
    sigmoid_pos = 1.0 / (1.0 + exp_neg_x)
    
    exp_x = tl.exp(x)
    sigmoid_neg = exp_x / (1.0 + exp_x)
    
    sigmoid = tl.where(pos_mask, sigmoid_pos, sigmoid_neg)
    
    # Swish = x * sigmoid(x)
    out = x * sigmoid
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable block size
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)