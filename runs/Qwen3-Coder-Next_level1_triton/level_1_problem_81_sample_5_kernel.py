import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    # Input tensor
    x_ptr, 
    # Weight tensor
    w_ptr, 
    # Bias tensor (optional)
    b_ptr,
    # Output tensor
    y_ptr,
    # Dimensions
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
    dilation,
    # Strides
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    w_out_channel_stride, w_in_channel_stride, w_height_stride, w_width_stride,
    y_batch_stride, y_channel_stride, y_height_stride, y_width_stride,
    # Block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,  # Output channels block size
    BLOCK_SIZE_N: tl.constexpr,  # Input channels block size
    BLOCK_SIZE_K: tl.constexpr,  # Kernel size block size
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_c = pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Loop over kernel positions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position based on transposed convolution formula
                in_h = (out_h - kh * dilation + padding) // stride
                in_w = (out_w - kw * dilation + padding) // stride
                
                # Check if input position is valid
                valid = (in_h >= 0) & (in_h < height_in) & (in_w >= 0) & (in_w < width_in)
                
                if valid:
                    # Calculate input pointer offset
                    x_offset = (pid_batch * x_batch_stride + 
                               in_c * x_channel_stride + 
                               in_h * x_height_stride + 
                               in_w * x_width_stride)
                    
                    # Calculate weight pointer offset
                    w_offset = (pid_out_ch * w_out_channel_stride + 
                               in_c * w_in_channel_stride + 
                               kh * w_height_stride + 
                               kw * w_width_stride)
                    
                    # Load input and weight values
                    x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + w_offset, mask=valid, other=0.0)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        acc += tl.load(b_ptr + b_offset)
    
    # Store output
    y_offset = (pid_batch * y_batch_stride + 
               pid_out_ch * y_channel_stride + 
               pid_h * y_height_stride + 
               pid_w * y_width_stride)
    
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=(pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < out_channels))


def triton_transposed_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                             stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Triton implementation of 2D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size_h - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size_w - 1) + 1
    
    # Allocate output tensor
    y = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    x_batch_stride = x.stride(0)
    x_channel_stride = x.stride(1)
    x_height_stride = x.stride(2)
    x_width_stride = x.stride(3)
    
    w_out_channel_stride = weight.stride(0)
    w_in_channel_stride = weight.stride(1)
    w_height_stride = weight.stride(2)
    w_width_stride = weight.stride(3)
    
    y_batch_stride = y.stride(0)
    y_channel_stride = y.stride(1)
    y_height_stride = y.stride(2)
    y_width_stride = y.stride(3)
    
    # Grid dimensions
    grid = (batch_size, 
            triton.cdiv(out_channels, 16),  # Out channels blocks
            height_out, 
            width_out)
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        height_in, width_in, height_out, width_out,
        kernel_size_h, stride, padding, dilation,
        x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
        w_out_channel_stride, w_in_channel_stride, w_height_stride, w_width_stride,
        y_batch_stride, y_channel_stride, y_height_stride, y_width_stride,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=1, BLOCK_SIZE_K=1
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with the same parameters as original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters (mimicking PyTorch's initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_transposed_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )


# Import math for initialization
import math