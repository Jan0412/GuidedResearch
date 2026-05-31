import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Calculate global output index
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_ch_idx * output_height * output_width + \
                   out_h_idx * output_width + out_w_idx
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for ch in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel and output channel
        weight_offset = out_ch_idx * in_channels * kernel_h * kernel_w + \
                       ch * kernel_h * kernel_w
        
        # Compute convolution for this channel group
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                ih = out_h_idx * stride_h - padding_h + kh * dilation_h
                iw = out_w_idx * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input value
                    input_offset = batch_idx * in_channels * input_height * input_width + \
                                 ch * input_height * input_width + \
                                 ih * input_width + iw
                    input_val = tl.load(input_ptr + input_offset, mask=True)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + weight_offset + kh * kernel_w + kw, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
                else:
                    # Out of bounds, treat as zero
                    pass
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx, mask=True)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc[0], mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        
        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Configure grid dimensions
        grid = (
            batch_size,           # Batch dimension
            self.out_channels,    # Output channels
            output_height,        # Output height
            output_width          # Output width
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            weight,
            output,
            bias_ptr,
            input_height,
            input_width,
            output_height,
            output_width,
            self.in_channels,
            self.out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            batch_size,
            BLOCK_SIZE=32,
            CHANNELS_PER_BLOCK=16,
            OUTPUTS_PER_BLOCK=16
        )
        
        return output