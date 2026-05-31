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
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    out_h_id = tl.program_id(3)
    out_w_id = tl.program_id(4)
    
    # Calculate output position
    out_d = out_d_id * BLOCK_SIZE
    out_h = out_h_id * BLOCK_SIZE
    out_w = out_w_id * BLOCK_SIZE
    
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group handling
    group_offset = (out_ch_id // group_size) * group_size
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Compute input coordinates
                input_d = out_d * stride_depth - padding_depth + kd
                input_h = out_h * stride_height - padding_height + kh
                input_w = out_w * stride_width - padding_width + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_id * in_channels * input_depth * input_height * input_width +
                                       (out_ch_id % group_size) * input_depth * input_height * input_width +
                                       input_d * input_height * input_width +
                                       input_h * input_width +
                                       input_w)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_ch_id * groups * kernel_depth * kernel_height * kernel_width * group_size +
                                        group_offset * kernel_depth * kernel_height * kernel_width * group_size +
                                        kd * kernel_height * kernel_width * group_size +
                                        kh * kernel_width * group_size +
                                        kw * group_size +
                                        (out_ch_id % group_size))
                    
                    acc += input_val * weight_val
    
    # Add bias if exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id)
        acc += bias_val
    
    # Store result
    if (out_d < output_depth and out_h < output_height and out_w < output_width):
        tl.store(output_ptr + 
                batch_id * out_channels * output_depth * output_height * output_width +
                out_ch_id * output_depth * output_height * output_width +
                out_d * output_height * output_width +
                out_h * output_width +
                out_w,
                acc)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 8
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels,
        (output_depth + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        groups,
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Ensure proper initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )

# Note: This implementation has limitations compared to PyTorch's native implementation
# For production use, more sophisticated optimizations would be needed