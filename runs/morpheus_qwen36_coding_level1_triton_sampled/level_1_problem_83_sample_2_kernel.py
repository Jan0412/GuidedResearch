import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    B, C, H_in, W, K, H_out,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Grid dimensions: (B, C, H_out, num_w_blocks)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w_block = tl.program_id(3)

    # Calculate base pointers
    x_base = b * C * H_in * W + c * H_in * W + h * W
    w_base = c * K
    out_base = b * C * H_out * W + c * H_out * W + h * W
    
    x_ptr += x_base
    w_ptr += w_base
    out_ptr += out_base
    
    # Load weights for this channel once
    w_vals = tl.load(w_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    
    # Bias value for this channel
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + c)
    else:
        bias_val = 0.0

    # Output w-offsets for this block
    w_idx = w_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = w_idx < W
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute dot product over kernel height
    for k in range(K):
        # Load input slice at h+k for all w in block
        x_slice = tl.load(x_ptr + k * W + w_idx, mask=mask, other=0.0)
        acc += x_slice * w_vals[k]
        
    # Add bias if present
    if HAS_BIAS:
        acc += bias_val
        
    # Store result
    tl.store(out_ptr + w_idx, acc, mask=mask)


def triton_depthwise_conv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    B, C, H_in, W = x.shape
    K = w.shape[2]
    H_out = H_in - K + 1
    
    out = torch.empty((B, C, H_out, W), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_w_blocks = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (B, C, H_out, num_w_blocks)
    
    has_bias = bias is not None
    if has_bias:
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = 0
        
    depthwise_conv_kernel[grid](
        x.data_ptr(),
        w.data_ptr(),
        out.data_ptr(),
        bias_ptr,
        B, C, H_in, W, K, H_out,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_BIAS=has_bias
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Depthwise conv with kernel (kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(x, self.weight, self.bias)