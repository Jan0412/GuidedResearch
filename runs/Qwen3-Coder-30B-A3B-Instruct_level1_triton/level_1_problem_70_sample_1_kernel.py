import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output position
    out_pos = out_d_idx * (output_height * output_width) + out_h_idx * output_width + out_w_idx
    
    # Calculate input positions for this output position
    # For transposed conv, we need to consider how the kernel maps to input
    # This is a simplified version - actual implementation would be more complex
    
    # Shared memory for input tiles
    input_tile = tl.shared_ptr(tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process kernel elements
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Calculate corresponding input position
                d_in = out_d_idx * stride - padding + k_d * dilation
                h_in = out_h_idx * stride - padding + k_h * dilation
                w_in = out_w_idx * stride - padding + k_w * dilation
                
                # Check bounds
                if (d_in >= 0 and d_in < input_depth and 
                    h_in >= 0 and h_in < input_height and 
                    w_in >= 0 and w_in < input_width):
                    
                    # Calculate input index
                    input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                               (out_ch_idx % (in_channels // groups)) * (input_depth * input_height * input_width) + \
                               d_in * (input_height * input_width) + \
                               h_in * input_width + \
                               w_in
                    
                    # Calculate weight index
                    weight_idx = out_ch_idx * (in_channels // groups * kernel_size * kernel_size * kernel_size) + \
                                (out_ch_idx % (in_channels // groups)) * (kernel_size * kernel_size * kernel_size) + \
                                k_d * (kernel_size * kernel_size) + \
                                k_h * kernel_size + \
                                k_w
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                out_ch_idx * (output_depth * output_height * output_width) + \
                out_d_idx * (output_height * output_width) + \
                out_h_idx * output_width + \
                out_w_idx
    
    tl.store(output_ptr + output_idx, acc[0], mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    """
    Triton implementation of ConvTranspose3d operation
    """
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + 1 + output_padding
    output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + 1 + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1 + output_padding
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Create grid
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        stride,
        padding,
        dilation,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'dilation={self.dilation}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])