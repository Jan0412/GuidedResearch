import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (optional)
    out_ptr,  # Output tensor pointer
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    depth, height, width,  # Input dimensions
    out_depth, out_height, out_width,  # Output dimensions
    kernel_size,  # K
    stride,  # S
    padding,  # P
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_z = tl.program_id(2)  # depth dimension
    pid_y = tl.program_id(3)  # height dimension
    pid_x = tl.program_id(4)  # width dimension
    
    # Calculate output position
    out_z = pid_z
    out_y = pid_y
    out_x = pid_x
    
    # Calculate input position (accounting for stride and padding)
    in_z = out_z * stride - padding + pid_z_kernel * 0  # Will be handled by loop
    in_y = out_y * stride - padding + pid_y_kernel * 0
    in_x = out_x * stride - padding + pid_x_kernel * 0
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        # Loop over kernel dimensions
        for kz in range(kernel_size):
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    # Calculate input position with kernel offset
                    input_z = out_z * stride + kz - padding
                    input_y = out_y * stride + ky - padding
                    input_x = out_x * stride + kx - padding
                    
                    # Check bounds for input
                    if (0 <= input_z < depth and 
                        0 <= input_y < height and 
                        0 <= input_x < width):
                        
                        # Calculate input pointer offset
                        x_offset = (pid_batch * in_channels * depth * height * width +
                                   c_in * depth * height * width +
                                   input_z * height * width +
                                   input_y * width +
                                   input_x)
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Calculate weight pointer offset
                        # Weight shape for ConvTranspose3d: (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
                        # But we're doing grouped convolution, so we need to handle groups properly
                        w_offset = (c_in * out_channels * kernel_size * kernel_size * kernel_size +
                                   pid_out_c * kernel_size * kernel_size * kernel_size +
                                   kz * kernel_size * kernel_size +
                                   ky * kernel_size +
                                   kx)
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        acc += bias
    
    # Store result
    out_offset = (pid_batch * out_channels * out_depth * out_height * out_width +
                 pid_out_c * out_depth * out_height * out_width +
                 out_z * out_height * out_width +
                 out_y * out_width +
                 out_x)
    tl.store(out_ptr + out_offset, acc.to(tl.float32))


# Optimized version using better blocking strategy
@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth, height, width,
    out_depth, out_height, out_width,
    kernel_size,
    stride,
    padding,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 8,
    BLOCK_SIZE_N: tl.constexpr = 4,
    BLOCK_SIZE_K: tl.constexpr = 8,
):
    # Output dimensions: [batch, out_channels, depth, height, width]
    # Program IDs for output indexing
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_y = tl.program_id(3)
    pid_x = tl.program_id(4)
    
    # Calculate output position
    out_z = pid_z
    out_y = pid_y
    out_x = pid_x
    
    # Determine output channels for this group
    channels_per_group = out_channels // groups
    out_c_start = pid_group * channels_per_group
    
    # Loop over output channels in the group
    for out_c_offset in range(BLOCK_SIZE_M):
        out_c = out_c_start + out_c_offset
        if out_c >= out_channels:
            break
            
        # Accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over input channels in this group
        in_c_start = out_c  # For grouped conv, input channel = output channel in same group
        for in_c in range(in_c_start, in_channels, groups):
            # Loop over kernel
            for kz in range(kernel_size):
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        # Calculate input position
                        input_z = out_z * stride + kz - padding
                        input_y = out_y * stride + ky - padding
                        input_x = out_x * stride + kx - padding
                        
                        # Check bounds
                        if (0 <= input_z < depth and 
                            0 <= input_y < height and 
                            0 <= input_x < width):
                            
                            # Load input
                            x_offset = (pid_batch * in_channels * depth * height * width +
                                       in_c * depth * height * width +
                                       input_z * height * width +
                                       input_y * width +
                                       input_x)
                            x_val = tl.load(x_ptr + x_offset)
                            
                            # Load weight
                            # Weight layout: (in_channels, out_channels // groups, k, k, k)
                            # For grouped conv: weight[c_in, c_out//groups, kx, ky, kz]
                            w_offset = (in_c * channels_per_group * kernel_size * kernel_size * kernel_size +
                                       (out_c % channels_per_group) * kernel_size * kernel_size * kernel_size +
                                       kz * kernel_size * kernel_size +
                                       ky * kernel_size +
                                       kx)
                            w_val = tl.load(w_ptr + w_offset)
                            
                            acc += x_val * w_val
        
        # Add bias
        if b_ptr is not None:
            bias = tl.load(b_ptr + out_c)
            acc += bias
        
        # Store output
        out_offset = (pid_batch * out_channels * out_depth * out_height * out_width +
                     out_c * out_depth * out_height * out_width +
                     out_z * out_height * out_width +
                     out_y * out_width +
                     out_x)
        tl.store(out_ptr + out_offset, acc[0])


# Final optimized kernel with proper blocking
@triton.jit
def conv_transpose3d_optimized_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth, height, width,
    out_depth, out_height, out_width,
    kernel_size,
    stride,
    padding,
    groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    kernel_volume: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE_N: tl.constexpr = 4,
    BLOCK_SIZE_K: tl.constexpr = 4,
):
    # Output indices
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_y = tl.program_id(3)
    pid_x = tl.program_id(4)
    
    # Calculate output position
    out_z = pid_z
    out_y = pid_y
    out_x = pid_x
    
    # Determine output channels for this group
    out_c_start = pid_group * channels_per_group
    
    # Loop over output channels
    for out_c_offset in range(BLOCK_SIZE_M):
        out_c = out_c_start + out_c_offset
        if out_c >= out_channels:
            break
            
        # Accumulator
        acc = 0.0
        
        # Loop over input channels in this group
        for in_c_offset in range(BLOCK_SIZE_K):
            in_c = out_c + in_c_offset * groups
            if in_c >= in_channels:
                break
                
            # Loop over kernel
            for kz in range(kernel_size):
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        # Calculate input position
                        input_z = out_z * stride + kz - padding
                        input_y = out_y * stride + ky - padding
                        input_x = out_x * stride + kx - padding
                        
                        # Check bounds
                        if (0 <= input_z < depth and 
                            0 <= input_y < height and 
                            0 <= input_x < width):
                            
                            # Load input
                            x_offset = (pid_batch * in_channels * depth * height * width +
                                       in_c * depth * height * width +
                                       input_z * height * width +
                                       input_y * width +
                                       input_x)
                            x_val = tl.load(x_ptr + x_offset)
                            
                            # Load weight
                            w_offset = (in_c * channels_per_group * kernel_size * kernel_size * kernel_size +
                                       (out_c % channels_per_group) * kernel_size * kernel_size * kernel_size +
                                       kz * kernel_size * kernel_size +
                                       ky * kernel_size +
                                       kx)
                            w_val = tl.load(w_ptr + w_offset)
                            
                            acc += x_val * w_val
        
        # Add bias
        if b_ptr is not None:
            bias = tl.load(b_ptr + out_c)
            acc += bias
        
        # Store output
        out_offset = (pid_batch * out_channels * out_depth * out_height * out_width +
                     out_c * out_depth * out_height * out_width +
                     out_z * out_height * out_width +
                     out_y * out_width +
                     out_x)
        tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose3d(x, weight, bias, stride, padding, groups):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, depth, height, width = x.shape
    out_channels = weight.shape[1] * groups  # weight shape: (in_channels, out_channels // groups, k, k, k)
    
    # Calculate output dimensions
    out_depth = (depth - 1) * stride - 2 * padding + (kernel_size := weight.shape[2]) + max(0, 0)  # output_padding=0
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size
    
    # Initialize output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, 
                     dtype=x.dtype, device=x.device)
    
    # Kernel parameters
    channels_per_group = out_channels // groups
    kernel_size = weight.shape[2]  # Assuming square kernel
    
    # Grid dimensions: [batch_size, groups, out_depth, out_height, out_width]
    grid = (batch_size, groups, out_depth, out_height, out_width)
    
    # Launch kernel
    conv_transpose3d_optimized_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        groups,
        channels_per_group,
        kernel_size * kernel_size * kernel_size,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_K=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same weights as the original ConvTranspose3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights similar to nn.ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using kaiming_uniform_ like nn.ConvTranspose3d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.groups)