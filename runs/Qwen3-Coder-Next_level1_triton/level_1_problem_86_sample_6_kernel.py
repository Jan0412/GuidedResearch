import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    w_ptr,  # Weight tensor: (C, 1, K_h, K_w)
    b_ptr,  # Bias tensor: (C,)
    out_ptr,  # Output tensor: (B, C, H_out, W_out)
    batch_size, in_channels, in_height, in_width,
    out_height, out_width,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs for batch and channel
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate output position
    pid_h = tl.program_id(2) // (tl.cdiv(out_width, BLOCK_SIZE_W))
    pid_w = tl.program_id(2) % (tl.cdiv(out_width, BLOCK_SIZE_W))
    
    # Calculate output coordinates
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    # Create offsets for output
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    h_mask = h_offsets < out_height
    w_mask = w_offsets < out_width
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Depthwise convolution loop
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input coordinates
            h_in = h_start * stride + kh * dilation - padding
            w_in = w_start * stride + kw * dilation - padding
            
            # Input offsets
            h_in_offsets = h_in + tl.arange(0, BLOCK_SIZE_H)
            w_in_offsets = w_in + tl.arange(0, BLOCK_SIZE_W)
            
            # Input masks
            h_in_mask = (h_in_offsets >= 0) & (h_in_offsets < in_height)
            w_in_mask = (w_in_offsets >= 0) & (w_in_offsets < in_width)
            hw_in_mask = h_in_mask[:, None] & w_in_mask[None, :]
            
            # Calculate input pointer offset
            input_offset = pid_b * (in_channels * in_height * in_width) + \
                          pid_c * (in_height * in_width) + \
                          h_in_offsets[:, None] * in_width + w_in_offsets[None, :]
            
            # Load input values
            x = tl.load(x_ptr + input_offset, mask=hw_in_mask & hw_mask, other=0.0)
            
            # Load weight value
            weight_offset = pid_c * kernel_size * kernel_size + \
                           kh * kernel_size + kw
            w = tl.load(w_ptr + weight_offset)
            
            # Accumulate
            acc += x * w
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c)
        acc += bias
    
    # Store result
    out_offset = pid_b * (in_channels * out_height * out_width) + \
                pid_c * (out_height * out_width) + \
                h_offsets[:, None] * out_width + w_offsets[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, 1, 1)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, H, W)
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c_out = tl.program_id(2)
    
    # Calculate output coordinates
    h_start = pid_h * BLOCK_SIZE_H
    w_start = 0  # Process entire width
    
    # Create offsets for output
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Pointwise convolution (1x1) - sum over input channels
    for c_in_start in range(0, in_channels, BLOCK_SIZE_C):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_offsets < in_channels
        
        # Create input tensor offset
        input_offset = pid_b * (in_channels * height * width) + \
                      c_in_offsets[None, None, :] * (height * width) + \
                      h_offsets[:, None, None] * width + w_offsets[None, :, None]
        
        # Load input values (B, H, W, C_in_block)
        x = tl.load(x_ptr + input_offset, mask=hw_mask[:, :, None] & c_in_mask[None, None, :], other=0.0)
        
        # Create weight tensor offset
        weight_offset = pid_c_out * in_channels + c_in_offsets
        w = tl.load(w_ptr + weight_offset, mask=c_in_mask, other=0.0)
        
        # Reshape for broadcasting: w is (C_in_block,), x is (H, W, C_in_block)
        acc += tl.sum(x * w[None, None, :], axis=2)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_offset = pid_b * (out_channels * height * width) + \
                pid_c_out * (height * width) + \
                h_offsets[:, None] * width + w_offsets[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Perform depthwise convolution using Triton kernel."""
    batch_size, in_channels, in_height, in_width = x.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty(batch_size, in_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    grid = (
        batch_size,  # batch
        in_channels,  # channels
        (tl.cdiv(out_height, BLOCK_SIZE_H) * tl.cdiv(out_width, BLOCK_SIZE_W))  # spatial blocks
    )
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, in_height, in_width,
        out_height, out_width,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


def triton_pointwise_conv2d(x, weight, bias=None):
    """Perform pointwise convolution using Triton kernel."""
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    out = torch.empty(batch_size, out_channels, height, width, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 32
    
    grid = (
        batch_size,
        tl.cdiv(height, BLOCK_SIZE_H),
        out_channels
    )
    
    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize depthwise and pointwise convolutions
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform depthwise convolution using Triton kernel
        x_depthwise = triton_depthwise_conv2d(x, self.depthwise.weight, 
                                             self.depthwise.bias if self.bias else None,
                                             self.stride, self.padding, self.dilation)
        
        # Perform pointwise convolution using Triton kernel
        x_out = triton_pointwise_conv2d(x_depthwise, self.pointwise.weight,
                                       self.pointwise.bias if self.bias else None)
        
        return x_out