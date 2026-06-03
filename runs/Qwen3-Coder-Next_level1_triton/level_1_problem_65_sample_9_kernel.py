import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (N, C_in, H_in, W_in)
    w_ptr,  # Weight tensor (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias tensor (C_out,)
    out_ptr,  # Output tensor (N, C_out, H_out, W_out)
    N, C_in, C_out, groups,
    H_in, W_in, H_out, W_out,
    kH, kW,
    stride, padding, output_padding,
    # Block sizes for tiling
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_H_OUT: tl.constexpr,
    BLOCK_SIZE_W_OUT: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h_out = tl.program_id(2)
    pid_w_out = tl.program_id(3)
    
    # Compute batch index and output channel index
    batch_idx = pid_n
    out_channel_start = pid_c_out * BLOCK_SIZE_C_OUT
    h_out_start = pid_h_out * BLOCK_SIZE_H_OUT
    w_out_start = pid_w_out * BLOCK_SIZE_W_OUT
    
    # Compute output position
    h_out = h_out_start + tl.arange(0, BLOCK_SIZE_H_OUT)[:, None]
    w_out = w_out_start + tl.arange(0, BLOCK_SIZE_W_OUT)[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H_OUT, BLOCK_SIZE_W_OUT), dtype=tl.float32)
    
    # Iterate over input channels and groups
    for c_in_idx in range(0, C_in, BLOCK_SIZE_C_IN):
        # Compute input position based on output position and kernel position
        for kh in range(kH):
            for kw in range(kW):
                # Compute corresponding input position
                h_in = h_out * stride - padding + kh
                w_in = w_out * stride - padding + kw
                
                # Check if input position is valid
                mask = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                
                # Load input values
                x_offsets = batch_idx * (C_in * H_in * W_in) + c_in_idx * (H_in * W_in) + h_in[:, None] * W_in + w_in[None, :]
                x_mask = (c_in_idx < C_in) & mask
                x_val = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
                
                # Load weight values
                w_offsets = c_in_idx * (C_out * kH * kW) + out_channel_start * (kH * kW) + kh * kW + kw
                w_val = tl.load(w_ptr + w_offsets, mask=(c_in_idx < C_in), other=0.0)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_C_OUT)[:, None]
        bias_val = tl.load(b_ptr + bias_offsets, mask=(out_channel_start < C_out), other=0.0)
        acc += bias_val
    
    # Store output
    out_offsets = batch_idx * (C_out * H_out * W_out) + out_channel_start * (H_out * W_out) + h_out[:, None] * W_out + w_out[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=(out_channel_start < C_out))


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """Custom Triton implementation of ConvTranspose2d"""
    N, C_in, H_in, W_in = x.shape
    C_in_group, C_out_per_group, kH, kW = weight.shape
    C_out = C_in_group * C_out_per_group * groups
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding
    
    # Allocate output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_C_IN = 16
    BLOCK_SIZE_H_OUT = 8
    BLOCK_SIZE_W_OUT = 8
    
    grid = (
        N,
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
        (H_out + BLOCK_SIZE_H_OUT - 1) // BLOCK_SIZE_H_OUT,
        (W_out + BLOCK_SIZE_W_OUT - 1) // BLOCK_SIZE_W_OUT,
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, groups,
        H_in, W_in, H_out, W_out,
        kH, kW,
        stride, padding, output_padding,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_H_OUT=BLOCK_SIZE_H_OUT,
        BLOCK_SIZE_W_OUT=BLOCK_SIZE_W_OUT,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters (matching nn.ConvTranspose2d format)
        kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters (matching nn.ConvTranspose2d initialization)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias similar to PyTorch's ConvTranspose2d"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call our Triton implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )


import math

# Re-define ModelNew to include math import at the right scope
class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters (matching nn.ConvTranspose2d format)
        kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters (matching nn.ConvTranspose2d initialization)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias similar to PyTorch's ConvTranspose2d"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call our Triton implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )