import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_asymmetric_kernel(
    x_ptr,  # Input tensor pointer: (batch, in_channels, H, W)
    w_ptr,  # Weight tensor pointer: (out_channels, in_channels, kH, kW)
    b_ptr,  # Bias tensor pointer: (out_channels,) or None
    out_ptr,  # Output tensor pointer: (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    height, width,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    out_h, out_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation (in_channels)
):
    # Output tensor indices
    batch_idx = tl.program_id(1)
    out_c_idx = tl.program_id(0)
    
    # Compute the start indices for output spatial positions
    # We'll process multiple output positions per program for better efficiency
    # Using a 2D grid: (out_channels, batch) with tiling over spatial positions
    
    # Calculate spatial position for this thread block
    # We'll use a sliding window approach - each program handles one output position
    
    # Compute output spatial coordinates
    oh = tl.program_id(2) // out_w
    ow = tl.program_id(2) % out_w
    
    # Compute input spatial coordinates
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel height
        for kh in range(kH):
            ih = ih_start + kh * dil_h
            # Check if within bounds
            if ih >= 0 and ih < height:
                # Loop over kernel width
                for kw in range(kW):
                    iw = iw_start + kw * dil_w
                    if iw >= 0 and iw < width:
                        # Load input value
                        x_offset = batch_idx * (in_channels * height * width) + \
                                  ic * (height * width) + \
                                  ih * width + iw
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight value
                        w_offset = out_c_idx * (in_channels * kH * kW) + \
                                  ic * (kH * kW) + \
                                  kh * kW + kw
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    out_offset = batch_idx * (out_channels * out_h * out_w) + \
                out_c_idx * (out_h * out_w) + \
                oh * out_w + ow
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution for the specific asymmetric kernel case.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # Handle stride, padding, dilation as tuples or integers
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_h = (height + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Check for bias
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
    
    # Configure kernel launch
    # Grid: (out_channels, batch, out_h * out_w)
    grid = (out_channels, batch_size, out_h * out_w)
    
    # Tunable parameters
    BLOCK_SIZE_M = 1  # Since we're computing one output channel at a time
    BLOCK_SIZE_N = 1  # One batch at a time
    BLOCK_SIZE_K = 32  # Accumulation block size (can be optimized)
    
    # Launch kernel
    conv2d_asymmetric_kernel[grid](
        x, weight, bias if has_bias else None, out,
        batch_size, in_channels, out_channels,
        height, width,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        out_h, out_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and create weight/bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights similar to nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming_uniform as in PyTorch's Conv2d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        # For this specific implementation, we assume groups=1 as per the original architecture
        # If groups > 1, we would need to implement grouped convolution
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


import math