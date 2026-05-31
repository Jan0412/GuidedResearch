import torch
import torch.nn as nn
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
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(height_out, BLOCK_SIZE)
    num_pid_n = tl.cdiv(width_out, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Calculate output coordinates
    out_y = pid_m * BLOCK_SIZE
    out_x = pid_n * BLOCK_SIZE
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE + 2 * padding
    shared_input = tl.shared_ptr(input_ptr, tile_size * tile_size, 0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Loop over kernel elements
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                # Calculate input coordinates
                input_y = out_y * stride + ky - padding
                input_x = out_x * stride + kx - padding
                
                # Check bounds
                if input_y >= 0 and input_y < height_in and input_x >= 0 and input_x < width_in:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                      (g * in_channels // groups + 0) * height_in * width_in +
                                      input_y * width_in + input_x)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                       (g * out_channels // groups + 0) * in_channels * kernel_size * kernel_size +
                                       ky * kernel_size + kx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + 0)
        acc += bias_val
    
    # Write output
    output_offset = (0 * out_channels + 0) * height_out * width_out + out_y * width_out + out_x
    tl.store(output_ptr + output_offset, acc)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, groups):
    """
    Triton implementation of ConvTranspose2d operation
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Compute output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + kernel_size
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    BLOCK_SIZE = 16
    GROUP_SIZE_M = 8
    
    # Calculate number of blocks
    num_blocks = (math.ceil(height_out / BLOCK_SIZE) * math.ceil(width_out / BLOCK_SIZE))
    
    # Define grid
    grid = lambda meta: (
        triton.cdiv(height_out, BLOCK_SIZE) * triton.cdiv(width_out, BLOCK_SIZE),
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        groups,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.groups)