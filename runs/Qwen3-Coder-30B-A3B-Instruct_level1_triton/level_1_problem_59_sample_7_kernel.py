import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    depth,
    kernel_h,
    kernel_w,
    kernel_d,
    out_height,
    out_width,
    out_depth,
    stride_h,
    stride_w,
    stride_d,
    padding_h,
    padding_w,
    padding_d,
    dilation_h,
    dilation_w,
    dilation_d,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_pos_idx = tl.program_id(2)
    
    # Calculate output position within block
    out_pos_offset = out_pos_idx * OUTPUT_ELEMENTS_PER_BLOCK + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
    mask = out_pos_offset < out_height * out_width * out_depth
    
    # Calculate output coordinates
    out_z = out_pos_offset // (out_width * out_height)
    out_y = (out_pos_offset % (out_width * out_height)) // out_width
    out_x = out_pos_offset % out_width
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Process groups
    group_size = in_channels // groups
    group_idx = out_channel_idx // (out_channels // groups)
    
    # For each channel in the group
    for c in range(0, group_size, CHANNELS_PER_BLOCK):
        channel_mask = (c + tl.arange(0, CHANNELS_PER_BLOCK)) < group_size
        channel_offsets = c + tl.arange(0, CHANNELS_PER_BLOCK)
        
        # Calculate input channel offset for this group
        input_channel_offset = group_idx * group_size + channel_offsets
        
        # Loop over kernel dimensions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for kd in range(kernel_d):
                    # Calculate input coordinates
                    input_z = out_z * stride_d - padding_d + kd * dilation_d
                    input_y = out_y * stride_h - padding_h + kh * dilation_h
                    input_x = out_x * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input coordinates are valid
                    valid_mask = (input_z >= 0) & (input_z < depth) & \
                                (input_y >= 0) & (input_y < height) & \
                                (input_x >= 0) & (input_x < width)
                    
                    # Apply masks
                    final_mask = mask & valid_mask & channel_mask
                    
                    # Load input values
                    input_vals = tl.load(input_ptr + 
                        batch_idx * (in_channels * height * width * depth) +
                        input_channel_offset * (height * width * depth) +
                        input_z * (height * width) + 
                        input_y * width + 
                        input_x, 
                        mask=final_mask, other=0.0)
                    
                    # Load weight values
                    weight_val = tl.load(weight_ptr + 
                        out_channel_idx * (group_size * kernel_h * kernel_w * kernel_d) +
                        channel_offsets * (kernel_h * kernel_w * kernel_d) +
                        kh * (kernel_w * kernel_d) + 
                        kw * kernel_d + 
                        kd, 
                        mask=channel_mask, other=0.0)
                    
                    # Accumulate
                    acc += input_vals * weight_val
    
    # Apply bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx, mask=True)
        acc += bias_val
    
    # Write output
    output_offset = batch_idx * (out_channels * out_height * out_width * out_depth) + \
                   out_channel_idx * (out_height * out_width * out_depth) + \
                   out_pos_offset
    
    tl.store(output_ptr + output_offset, acc, mask=mask)

def triton_conv3d(input_tensor, weight, bias, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Custom Triton implementation of 3D convolution
    """
    batch_size, in_channels, height, width, depth = input_tensor.shape
    out_channels, _, kernel_h, kernel_w, kernel_d = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    out_depth = (depth + 2 * padding[2] - (dilation[2] * (kernel_d - 1) + 1)) // stride[2] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, out_depth, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare kernel parameters
    stride_h, stride_w, stride_d = stride
    padding_h, padding_w, padding_d = padding
    dilation_h, dilation_w, dilation_d = dilation
    
    # Define block sizes
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 32
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Calculate grid dimensions
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channels
        (out_height * out_width * out_depth + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK  # output positions
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        depth,
        kernel_h,
        kernel_w,
        kernel_d,
        out_height,
        out_width,
        out_depth,
        stride_h,
        stride_w,
        stride_d,
        padding_h,
        padding_w,
        padding_d,
        dilation_h,
        dilation_w,
        dilation_d,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        # Use the original PyTorch conv3d for initialization and basic functionality
        # But we'll override the actual computation with our Triton kernel
        return triton_conv3d(
            x, 
            self.conv3d.weight, 
            self.conv3d.bias, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

# Keep the original class for comparison
class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.
        """
        return self.conv3d(x)