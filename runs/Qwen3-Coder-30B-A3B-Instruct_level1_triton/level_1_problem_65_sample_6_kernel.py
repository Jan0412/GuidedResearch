import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
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
    output_padding_h,
    output_padding_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Shared memory for weight tiles
    tile_weight = tl.shared.tensor([GROUP_SIZE, kernel_h, kernel_w], dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Calculate output position
    out_y_start = tl.program_id(2) * BLOCK_SIZE
    out_x_start = tl.program_id(3) * BLOCK_SIZE
    
    # Loop over input channels
    for c in range(0, in_channels, GROUP_SIZE):
        # Load weight tile
        weight_offsets = c + tl.arange(0, GROUP_SIZE)
        tile_weight = tl.load(weight_ptr + out_channel_id * in_channels * kernel_h * kernel_w + 
                             weight_offsets[:, None, None] * kernel_h * kernel_w +
                             tl.arange(0, kernel_h)[:, None] * kernel_w +
                             tl.arange(0, kernel_w)[None, :])
        
        # Loop over kernel positions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                input_y = out_y_start + kh * stride_h - padding_h
                input_x = out_x_start + kw * stride_w - padding_w
                
                # Check bounds
                if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                    # Load input data
                    input_offsets = batch_id * in_channels * input_height * input_width + \
                                   c + tl.arange(0, GROUP_SIZE) * input_height * input_width + \
                                   input_y * input_width + input_x
                    
                    input_data = tl.load(input_ptr + input_offsets, mask=(c + tl.arange(0, GROUP_SIZE)) < in_channels)
                    
                    # Accumulate
                    acc += input_data * tile_weight[:, kh, kw]
    
    # Store result
    output_offsets = batch_id * out_channels * output_height * output_width + \
                     out_channel_id * output_height * output_width + \
                     out_y_start * output_width + out_x_start
    
    # Apply bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_id)
        acc += bias_val
    
    # Store output
    tl.store(output_ptr + output_offsets, acc, mask=(out_y_start + tl.arange(0, BLOCK_SIZE)) < output_height)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
        
        # Reshape input for computation
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Launch kernel
        BLOCK_SIZE = 32
        GROUP_SIZE = 8
        
        # Grid configuration
        grid = (
            batch_size,
            self.out_channels,
            (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        
        # Call kernel
        conv_transpose2d_kernel[grid](
            x,
            weight,
            output,
            self.bias,
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
            output_padding_h,
            output_padding_w,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        return output