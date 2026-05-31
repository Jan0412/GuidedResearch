import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,   # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    output_ptr,  # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_width,
    input_height,
    input_depth,
    output_width,
    output_height,
    output_depth,
    kernel_width,
    kernel_height,
    kernel_depth,
    stride_w,
    stride_h,
    stride_d,
    padding_w,
    padding_h,
    padding_d,
    dilation_w,
    dilation_h,
    dilation_d,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    out_z = tl.program_id(4)
    
    # Calculate output position
    out_y_start = out_y * stride_h
    out_x_start = out_x * stride_w
    out_z_start = out_z * stride_d
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=(1, 1, input_height + 2*padding_h, input_width + 2*padding_w, input_depth + 2*padding_d), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kw in range(kernel_width):
            for kh in range(kernel_height):
                for kd in range(kernel_depth):
                    # Calculate input coordinates
                    input_y = out_y_start + kh * dilation_h - padding_h
                    input_x = out_x_start + kw * dilation_w - padding_w
                    input_z = out_z_start + kd * dilation_d - padding_d
                    
                    # Check bounds
                    if input_y >= 0 and input_y < input_height and \
                       input_x >= 0 and input_x < input_width and \
                       input_z >= 0 and input_z < input_depth:
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_id * in_channels * input_height * input_width * input_depth +
                                          ic * input_height * input_width * input_depth +
                                          input_y * input_width * input_depth +
                                          input_x * input_depth +
                                          input_z)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           out_channel_id * in_channels * kernel_height * kernel_width * kernel_depth +
                                           ic * kernel_height * kernel_width * kernel_depth +
                                           kh * kernel_width * kernel_depth +
                                           kw * kernel_depth +
                                           kd)
                        
                        acc += input_val * weight_val
    
    # Store result
    if out_x < output_width and out_y < output_height and out_z < output_depth:
        output_idx = batch_id * out_channels * output_height * output_width * output_depth + \
                     out_channel_id * output_height * output_width * output_depth + \
                     out_y * output_width * output_depth + \
                     out_x * output_depth + \
                     out_z
        tl.store(output_ptr + output_idx, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_height, input_width, input_depth = input_tensor.shape
    out_channels, _, kernel_height, kernel_width, kernel_depth = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_depth = (input_depth + 2 * padding[2] - (dilation[2] * (kernel_depth - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, output_depth, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_height + 15) // 16,  # Assuming BLOCK_SIZE=16 for Y dimension
        (output_width + 15) // 16,
        (output_depth + 15) // 16
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_width,
        input_height,
        input_depth,
        output_width,
        output_height,
        output_depth,
        kernel_width,
        kernel_height,
        kernel_depth,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE=16,
        OUTPUT_BLOCK_SIZE=16
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# For compatibility with original interface
def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    width = 64
    height = 64
    depth = 64
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [3, 64, (3, 5, 7)]