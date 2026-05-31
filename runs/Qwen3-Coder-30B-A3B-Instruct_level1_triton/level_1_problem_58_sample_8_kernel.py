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
    pad_d,
    pad_h,
    pad_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    output_idx = pid
    
    # Decompose output index into coordinates
    out_w = output_idx % output_width
    out_h = (output_idx // output_width) % output_height
    out_d = (output_idx // (output_width * output_height)) % output_depth
    batch_idx = (output_idx // (output_width * output_height * output_depth)) % batch_size
    out_c = (output_idx // (output_width * output_height * output_depth * batch_size)) % out_channels
    
    # Calculate input coordinates
    in_d = out_d * stride_d - pad_d
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions and input channels
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                for ic in range(in_channels // groups):
                    # Calculate input position
                    d = in_d + k_d
                    h = in_h + k_h
                    w = in_w + k_w
                    
                    # Check bounds
                    if d >= 0 and d < input_depth and h >= 0 and h < input_height and w >= 0 and w < input_width:
                        # Calculate input index
                        input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                                   (ic + (out_c // (out_channels // groups)) * (in_channels // groups)) * (input_depth * input_height * input_width) + \
                                   d * (input_height * input_width) + h * input_width + w
                        
                        # Calculate weight index
                        weight_idx = out_c * (in_channels // groups * kernel_depth * kernel_height * kernel_width) + \
                                    ic * (kernel_depth * kernel_height * kernel_width) + \
                                    k_d * (kernel_height * kernel_width) + k_h * kernel_width + k_w
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_c
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Store result
    output_idx_final = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                      out_c * (output_depth * output_height * output_width) + \
                      out_d * (output_height * output_width) + out_h * output_width + out_w
    
    tl.store(output_ptr + output_idx_final, acc, mask=True)

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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Set up kernel parameters
        self.kernel_depth, self.kernel_height, self.kernel_width = kernel_size
        self.stride_d, self.stride_h, self.stride_w = stride
        self.pad_d, self.pad_h, self.pad_w = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride_d - 2 * self.pad_d + self.kernel_depth + self.output_padding[0]
        output_height = (input_height - 1) * self.stride_h - 2 * self.pad_h + self.kernel_height + self.output_padding[1]
        output_width = (input_width - 1) * self.stride_w - 2 * self.pad_w + self.kernel_width + self.output_padding[2]
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Calculate total output elements
        total_output_elements = batch_size * self.out_channels * output_depth * output_height * output_width
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        conv_transpose3d_kernel[grid_size](
            x,
            weight,
            output,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_depth,
            self.kernel_height,
            self.kernel_width,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.pad_d,
            self.pad_h,
            self.pad_w,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS=self.groups
        )
        
        return output