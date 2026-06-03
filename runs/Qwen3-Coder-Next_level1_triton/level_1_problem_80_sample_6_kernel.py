import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size, in_channels, out_channels,
    height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
    dilation_h, dilation_w,
    output_h, output_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Spatial position block
    
    # Compute output channel range
    out_channel_offset = pid_m * BLOCK_SIZE_M
    out_channels_range = tl.arange(0, BLOCK_SIZE_M)
    out_channels_mask = out_channel_offset + out_channels_range < out_channels
    
    # Compute spatial position (flattened)
    spatial_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_w = spatial_idx % output_w
    out_h = spatial_idx // output_w
    batch_idx = tl.arange(0, BLOCK_SIZE_N) // (output_w * output_h)
    
    # Check bounds for spatial position
    spatial_mask = (out_h >= 0) & (out_h < output_h) & (out_w >= 0) & (out_w < output_w)
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        in_channel_offset = k
        in_channels_range = tl.arange(0, BLOCK_SIZE_K)
        in_channels_mask = in_channel_offset + in_channels_range < in_channels
        
        # Load input data
        # Compute input coordinates for each output position
        in_h = out_h * stride_h - pad_h_top + in_channels_range[None, :] * 0  # Will be updated below
        in_w = out_w * stride_w - pad_w_left + in_channels_range[None, :] * 0  # Will be updated below
        
        # Actually compute input coordinates considering dilation
        # For each kernel position
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input position
                input_h = out_h * stride_h - pad_h_top + kh * dilation_h
                input_w = out_w * stride_w - pad_w_left + kw * dilation_w
                
                # Check if input position is valid
                input_valid = (input_h >= 0) & (input_h < height) & (input_w >= 0) & (input_w < width)
                
                # Compute input pointer offset for this kernel position
                # Input layout: (batch, channel, height, width)
                # For a given batch, channel, height, width:
                # offset = batch * (C*H*W) + channel * (H*W) + height * W + width
                
                # We need to gather input values for all batches, input channels at each output position
                # This is complex, so we'll use a simpler approach with direct indexing
                
                # For simplicity in Triton, we'll iterate over output positions and compute input indices
                # Create indices for current kernel position
                
                # Compute batch offset
                batch_offset = batch_idx * (in_channels * height * width)
                # Compute channel offset
                channel_offset = in_channels_range[None, :] * (height * width)
                # Compute height offset
                height_offset = input_h[None, :] * width
                # Compute width offset
                width_offset = input_w[None, :]
                
                input_offset = batch_offset + channel_offset + height_offset + width_offset
                
                # Load input values (only valid positions)
                x_block = tl.load(
                    x_ptr + input_offset,
                    mask=input_valid & (batch_idx < batch_size)[None, :],
                    other=0.0
                )
                
                # Load corresponding weight values
                # Weight layout: (out_channels, in_channels, kernel_h, kernel_w)
                # For each output channel, input channel, kernel position
                weight_offset = (
                    (out_channel_offset + out_channels_range[:, None]) * (in_channels * kernel_h * kernel_w) +
                    in_channels_range[None, :] * (kernel_h * kernel_w) +
                    kh * kernel_w + kw
                )
                
                w_block = tl.load(
                    w_ptr + weight_offset,
                    mask=out_channels_mask[:, None] & in_channels_mask[None, :],
                    other=0.0
                )
                
                # Accumulate: acc[oc, sp] += sum(ic) x[ic, sp] * w[oc, ic, kh, kw]
                # For each output channel and spatial position
                # x_block has shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
                # w_block has shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
                acc += tl.dot(w_block, x_block, out_dtype=tl.float32)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_offset + out_channels_range, mask=out_channels_mask)
        acc += bias[:, None]
    
    # Store output
    # Output layout: (batch, out_channels, output_h, output_w)
    out_batch = batch_idx
    out_offset = (
        out_batch * (out_channels * output_h * output_w) +
        (out_channel_offset + out_channels_range[:, None]) * (output_h * output_w) +
        spatial_idx[None, :]
    )
    
    # Store result
    tl.store(
        out_ptr + out_offset,
        acc,
        mask=out_channels_mask[:, None] & spatial_mask[None, :] & (out_batch < batch_size)[None, :]
    )


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Handle stride, padding, dilation
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h_top = pad_h_bottom = pad_w_left = pad_w_right = padding
    else:
        if len(padding) == 2:
            pad_h_top = pad_h_bottom = padding[0]
            pad_w_left = pad_w_right = padding[1]
        else:  # 4 values
            pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_h = (height + pad_h_top + pad_h_bottom - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (width + pad_w_left + pad_w_right - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, output_h, output_w), dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 32  # Output channel block size
    BLOCK_SIZE_N = 64  # Spatial block size
    BLOCK_SIZE_K = 32  # Input channel block size
    
    # Grid definition
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size * output_h * output_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    conv2d_kernel[grid_m, grid_n](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        dilation_h, dilation_w,
        output_h, output_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model using custom Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the convolution layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create learnable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights (simple kaiming initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation)


import math