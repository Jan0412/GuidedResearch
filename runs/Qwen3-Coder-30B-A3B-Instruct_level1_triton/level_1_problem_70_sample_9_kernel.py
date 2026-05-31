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
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channel_block_id = tl.program_id(2)
    
    # Calculate output dimensions per block
    output_elements_per_block = OUTPUT_ELEMENTS_PER_BLOCK
    output_elements_total = output_depth * output_height * output_width
    
    # Process output elements in chunks
    for output_offset in range(0, output_elements_total, output_elements_per_block):
        # Get output indices for this block
        output_idx = output_offset + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
        mask = output_idx < output_elements_total
        
        if not mask.any():
            break
            
        # Convert linear index to 3D coordinates
        out_d = output_idx // (output_height * output_width)
        out_h = (output_idx % (output_height * output_width)) // output_width
        out_w = output_idx % output_width
        
        # Filter valid output positions
        valid_mask = (out_d < output_depth) & (out_h < output_height) & (out_w < output_width) & mask
        
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Compute input coordinates
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input coordinates are valid
                    input_valid = (input_d >= 0) & (input_d < input_depth) & \
                                  (input_h >= 0) & (input_h < input_height) & \
                                  (input_w >= 0) & (input_w < input_width)
                    
                    # Combine all masks
                    final_mask = valid_mask & input_valid
                    
                    if not final_mask.any():
                        continue
                        
                    # Calculate weights indices
                    weight_idx = (group_id * (out_channels // groups) + channel_block_id * CHANNELS_PER_BLOCK + 
                                tl.arange(0, CHANNELS_PER_BLOCK))
                    weight_mask = weight_idx < (out_channels // groups) + group_id * (out_channels // groups)
                    
                    # Calculate input indices
                    input_idx = (batch_id * in_channels * input_depth * input_height * input_width + 
                               group_id * (in_channels // groups) * input_depth * input_height * input_width + 
                               (input_d * input_height * input_width + input_h * input_width + input_w))
                    
                    # Load input values
                    input_vals = tl.load(input_ptr + input_idx, mask=final_mask, other=0.0)
                    
                    # Load weights
                    weight_vals = tl.load(weight_ptr + 
                                        (group_id * (out_channels // groups) * kernel_depth * kernel_height * kernel_width + 
                                         channel_block_id * CHANNELS_PER_BLOCK * kernel_depth * kernel_height * kernel_width + 
                                         kd * kernel_height * kernel_width + 
                                         kh * kernel_width + 
                                         kw), mask=weight_mask, other=0.0)
                    
                    # Accumulate
                    acc += input_vals * weight_vals
        
        # Store results
        output_idx = batch_id * out_channels * output_depth * output_height * output_width + \
                    channel_block_id * output_depth * output_height * output_width + \
                    out_d * output_height * output_width + \
                    out_h * output_width + \
                    out_w
        
        tl.store(output_ptr + output_idx, acc, mask=valid_mask)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), 
                           output_padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + output_padding[0] + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + output_padding[1] + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + output_padding[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 32
    OUTPUT_ELEMENTS_PER_BLOCK = 128
    
    # Grid configuration
    grid = (
        batch_size,  # Batch dimension
        groups,      # Groups dimension  
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK  # Channel blocks
    )
    
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
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
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