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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output dimensions
    out_h = height_out
    out_w = width_out
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2*padding, BLOCK_SIZE_W + 2*padding))
    
    # Calculate global positions
    start_h = out_h_idx * BLOCK_SIZE_H
    start_w = out_w_idx * BLOCK_SIZE_W
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Loop over kernel spatial dimensions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input positions
                h_start = start_h + kh * stride - padding
                w_start = start_w + kw * stride - padding
                
                # Load input data with boundary checks
                for ih in range(BLOCK_SIZE_H):
                    for iw in range(BLOCK_SIZE_W):
                        h_in = h_start + ih
                        w_in = w_start + iw
                        
                        if 0 <= h_in < height_in and 0 <= w_in < width_in:
                            input_val = tl.load(input_ptr + 
                                              batch_idx * (in_channels * height_in * width_in) +
                                              g * (group_in_channels * height_in * width_in) +
                                              h_in * (group_in_channels * width_in) +
                                              w_in * group_in_channels)
                            
                            # Load weight
                            weight_val = tl.load(weight_ptr +
                                               g * (group_out_channels * group_in_channels * kernel_size * kernel_size) +
                                               0 * (group_in_channels * kernel_size * kernel_size) +
                                               kh * (group_in_channels * kernel_size) +
                                               kw * group_in_channels)
                            
                            acc[ih, iw] += input_val * weight_val
                
        # Store output
        for ih in range(BLOCK_SIZE_H):
            for iw in range(BLOCK_SIZE_W):
                if start_h + ih < height_out and start_w + iw < width_out:
                    tl.store(output_ptr + 
                           batch_idx * (out_channels * height_out * width_out) +
                           0 * (height_out * width_out) +
                           (start_h + ih) * width_out +
                           (start_w + iw),
                           acc[ih, iw])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid configuration
    grid = (
        batch_size,
        (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
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
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized with custom Triton kernels.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use our Triton kernel implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )

# For testing purposes, this would be the original implementation
class Model(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)