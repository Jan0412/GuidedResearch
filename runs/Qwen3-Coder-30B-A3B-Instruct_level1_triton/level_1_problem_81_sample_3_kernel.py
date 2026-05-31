import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    dilation,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate global output position
    output_offset = batch_idx * (out_channels * height_out * width_out) + \
                   out_ch_idx * (height_out * width_out) + \
                   out_h_idx * width_out + out_w_idx
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # For each input channel and kernel position
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position
                # Transposed conv: input is at (out_h_idx * stride + kh * dilation - padding, out_w_idx * stride + kw * dilation - padding)
                ih = out_h_idx * stride + kh * dilation - padding
                iw = out_w_idx * stride + kw * dilation - padding
                
                # Check bounds
                if ih >= 0 and ih < height_in and iw >= 0 and iw < width_in:
                    # Calculate input offset
                    input_offset = batch_idx * (in_channels * height_in * width_in) + \
                                  ic * (height_in * width_in) + \
                                  ih * width_in + iw
                    
                    # Calculate weight offset (weight is stored as [out_channels, in_channels, kernel_h, kernel_w])
                    weight_offset = out_ch_idx * (in_channels * kernel_size * kernel_size) + \
                                   ic * (kernel_size * kernel_size) + \
                                   kh * kernel_size + kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_offset, mask=True)
                    weight_val = tl.load(weight_ptr + weight_offset, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + out_ch_idx, mask=True)
        acc += bias_val
    
    # Store output
    tl.store(output_ptr + output_offset, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_enabled = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, height_in, width_in = x.shape
        
        # Calculate output dimensions
        height_out = (height_in - 1) * self.stride - 2 * self.padding + (self.dilation * (self.kernel_size - 1) + 1)
        width_out = (width_in - 1) * self.stride - 2 * self.padding + (self.dilation * (self.kernel_size - 1) + 1)
        
        # Ensure output is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, height_out, width_out, device=x.device, dtype=torch.float32)
        
        # Create grid for launching kernel
        grid = (
            batch_size,
            self.out_channels,
            height_out,
            width_out
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x,
            self.weight,
            output,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            height_in,
            width_in,
            height_out,
            width_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.bias_enabled,
            BLOCK_SIZE=1024
        )
        
        return output