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
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_h_id = tl.program_id(2)
    out_w_id = tl.program_id(3)
    out_d_id = tl.program_id(4)
    
    # Calculate output indices
    out_h = out_h_id * BLOCK_SIZE
    out_w = out_w_id * BLOCK_SIZE
    out_d = out_d_id * BLOCK_SIZE
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=(BLOCK_SIZE + 2*padding_h, BLOCK_SIZE + 2*padding_w, BLOCK_SIZE + 2*padding_d))
    
    # Process multiple groups per block
    for g in range(GROUPS_PER_BLOCK):
        current_group = group_id * GROUPS_PER_BLOCK + g
        if current_group >= groups:
            break
            
        # Process multiple channels per block
        for c in range(CHANNELS_PER_BLOCK):
            channel_offset = current_group * (in_channels // groups) + c
            if channel_offset >= in_channels:
                continue
                
            # Initialize accumulator
            acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
            
            # Loop over kernel dimensions
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    for kd in range(kernel_d):
                        # Calculate input position
                        ih = out_h * stride_h + kh * dilation_h - padding_h
                        iw = out_w * stride_w + kw * dilation_w - padding_w
                        id = out_d * stride_d + kd * dilation_d - padding_d
                        
                        # Check bounds
                        if ih >= 0 and ih < height and iw >= 0 and iw < width and id >= 0 and id < depth:
                            # Load input value
                            input_val = tl.load(input_ptr + 
                                batch_id * (in_channels * height * width * depth) +
                                channel_offset * (height * width * depth) +
                                ih * (width * depth) +
                                iw * depth +
                                id)
                            
                            # Load weight
                            weight_val = tl.load(weight_ptr + 
                                current_group * (out_channels // groups) * in_channels * kernel_h * kernel_w * kernel_d +
                                0 * (in_channels * kernel_h * kernel_w * kernel_d) +
                                channel_offset * (kernel_h * kernel_w * kernel_d) +
                                kh * (kernel_w * kernel_d) +
                                kw * kernel_d +
                                kd)
                            
                            # Accumulate
                            acc += input_val * weight_val
            
            # Store output
            if out_h < out_height and out_w < out_width and out_d < out_depth:
                output_idx = (
                    batch_id * (out_channels * out_height * out_width * out_depth) +
                    0 * (out_height * out_width * out_depth) +
                    out_h * (out_width * out_depth) +
                    out_w * out_depth +
                    out_d
                )
                tl.store(output_ptr + output_idx, acc)

def triton_conv3d(input_tensor, weight, bias, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1):
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, height, width, depth = input_tensor.shape
    out_channels, _, kernel_h, kernel_w, kernel_d = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    out_depth = (depth + 2 * padding[2] - (dilation[2] * (kernel_d - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, out_depth, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    GROUPS_PER_BLOCK = 1
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    # Grid configuration
    grid = (
        batch_size,
        (groups + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK,
        (out_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (out_width + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (out_depth + BLOCK_SIZE - 1) // BLOCK_SIZE
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
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton implementation for 3D convolution
        return triton_conv3d(
            x, 
            self.conv3d.weight, 
            self.conv3d.bias,
            stride=self.conv3d.stride,
            padding=self.conv3d.padding,
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups
        )