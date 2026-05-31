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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * stride_h
    out_w_start = out_w_idx * stride_w
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            in_h = out_h_start - padding_h + kh * dilation_h
            in_w = out_w_start - padding_w + kw * dilation_w
            
            # Check bounds
            if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                # Calculate input index
                input_idx = batch_idx * (in_channels * input_height * input_width) + \
                           group_idx * channels_per_group * input_height * input_width + \
                           in_h * input_width + in_w
                
                # Calculate weight index
                weight_idx = group_idx * (channels_per_group * out_channels * kernel_height * kernel_width) + \
                            kh * (channels_per_group * out_channels * kernel_width) + \
                            kw * (channels_per_group * out_channels) + \
                            0  # Assuming single output channel for now
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = group_idx * channels_per_group
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Calculate output index
    output_idx = batch_idx * (out_channels * output_height * output_width) + \
                group_idx * (channels_per_group * output_height * output_width) + \
                out_h_idx * output_width + out_w_idx
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)


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
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1
        output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1
        
        # Ensure tensors are contiguous and on GPU
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, dtype=torch.float32, device=x.device)
        
        # Handle bias
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Define grid configuration
        grid = (
            batch_size,
            groups,
            math.ceil(output_height / stride_h),
            math.ceil(output_width / stride_w)
        )
        
        # Define block sizes
        BLOCK_SIZE = 16
        GROUPS_BLOCK_SIZE = 16
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x,
            weight,
            output,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            self.in_channels // groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE
        )
        
        return output