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
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    output_padding_d,
    output_padding_w,
    output_padding_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_w = tl.program_id(3)
    out_h = tl.program_id(4)
    
    # Calculate group info
    group_size = out_channels // groups
    group_id = out_ch_idx // group_size
    
    # Shared memory for input tile
    tile_size = 32
    input_tile = tl.shared_ptr(input_ptr, (tile_size, tile_size, tile_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_w in range(kernel_width):
            for k_h in range(kernel_height):
                # Calculate input position
                in_d = out_d * stride_d - padding_d + k_d
                in_w = out_w * stride_w - padding_w + k_w
                in_h = out_h * stride_h - padding_h + k_h
                
                # Check bounds
                if (in_d >= 0 and in_d < input_depth and 
                    in_w >= 0 and in_w < input_width and 
                    in_h >= 0 and in_h < input_height):
                    
                    # Calculate input index
                    input_idx = (batch_idx * in_channels * input_depth * input_width * input_height + 
                                group_id * input_depth * input_width * input_height + 
                                in_d * input_width * input_height + 
                                in_w * input_height + 
                                in_h)
                    
                    # Calculate weight index
                    weight_idx = (out_ch_idx * in_channels * kernel_depth * kernel_width * kernel_height + 
                                 group_id * kernel_depth * kernel_width * kernel_height + 
                                 k_d * kernel_width * kernel_height + 
                                 k_w * kernel_height + 
                                 k_h)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Write output
    if (out_d < output_depth and out_w < output_width and out_h < output_height):
        output_idx = (batch_idx * out_channels * output_depth * output_width * output_height + 
                     out_ch_idx * output_depth * output_width * output_height + 
                     out_d * output_width * output_height + 
                     out_w * output_height + 
                     out_h)
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1):
    """
    Triton implementation of ConvTranspose3d
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    out_pad_d, out_pad_w, out_pad_h = output_padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth + out_pad_d
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + out_pad_w
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + out_pad_h
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_width,
        output_height
    )
    
    # Set up parameters for kernel launch
    BLOCK_SIZE = 128
    GROUP_SIZE = 8
    
    # Convert to appropriate pointers
    input_ptr = input_tensor.data_ptr()
    weight_ptr = weight.data_ptr()
    output_ptr = output.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_ptr,
        weight_ptr,
        output_ptr,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride_d,
        stride_w,
        stride_h,
        pad_d,
        pad_w,
        pad_h,
        out_pad_d,
        out_pad_w,
        out_pad_h,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )