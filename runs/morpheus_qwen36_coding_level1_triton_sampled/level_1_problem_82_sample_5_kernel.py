import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    in_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    height_out,
    width_out,
    BLOCK_SIZE: tl.constexpr,
    has_bias: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Decompose program ID to (batch, channel, y, x)
    b = pid // (in_channels * height_out * width_out)
    rem = pid % (in_channels * height_out * width_out)
    c = rem // (height_out * width_out)
    rem = rem % (height_out * width_out)
    y = rem // width_out
    x = rem % width_out
    
    # Compute output coordinates
    out_offset = b * in_channels * height_out * width_out + c * height_out * width_out + y * width_out + x
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load bias if present
    if has_bias:
        bias_val = tl.load(b_ptr + c)
    else:
        bias_val = 0.0
        
    # Convolution loop
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            # Input coordinates
            in_y = y * stride + ky - padding
            in_x = x * stride + kx - padding
            
            # Mask for valid input coordinates
            mask = (in_y >= 0) & (in_y < height) & (in_x >= 0) & (in_x < width)
            
            if mask.any():
                # Load input tile
                in_offset = b * in_channels * height * width + c * height * width + in_y * width + in_x
                input_tile = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
                
                # Load weight tile
                w_offset = c * kernel_size * kernel_size + ky * kernel_size + kx
                weight_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += input_tile * weight_val
                
    # Add bias and store result
    if has_bias:
        acc += bias_val
        
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    has_bias: bool
) -> torch.Tensor:
    batch_size, in_channels, height, width = x.shape
    kernel_size = weight.shape[2]
    
    # Calculate output dimensions
    height_out = (height + 2 * padding - kernel_size) // stride + 1
    width_out = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, height_out, width_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (batch_size * in_channels * height_out * width_out,)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_size, stride, padding,
        height_out, width_out,
        BLOCK_SIZE=1,
        has_bias=has_bias
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and bias buffers
        self.register_buffer('weight', torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.register_buffer('bias', torch.zeros(in_channels))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_ptr = self.bias if self.bias is not None else None
        return triton_depthwise_conv2d(
            x, self.weight, bias_ptr, self.stride, self.padding, self.bias is not None
        )