import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    height, width,
    out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns
    BLOCK_SIZE_K: tl.constexpr,  # Block size for channel dimension
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    
    # Calculate starting positions for output
    out_row_start = pid_out_h * BLOCK_SIZE_M
    out_col_start = pid_out_w * BLOCK_SIZE_N
    
    # Create ranges for output row and column blocks
    out_row_offsets = out_row_start + tl.arange(0, BLOCK_SIZE_M)
    out_col_offsets = out_col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid output positions
    mask_m = out_row_offsets < out_height
    mask_n = out_col_offsets < out_width
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Calculate input channel range
        in_channel_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = in_channel_offsets < in_channels
        
        # Process each kernel position
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                in_row = pid_out_h * stride_h - pad_h + kh * dilation_h
                in_col = pid_out_w * stride_w - pad_w + kw * dilation_w
                
                # Load input block: (batch, in_channels_block, BLOCK_SIZE_M, BLOCK_SIZE_N)
                # We need to handle boundary conditions
                in_row_offsets = in_row + tl.arange(0, BLOCK_SIZE_M)[:, None]
                in_col_offsets = in_col + tl.arange(0, BLOCK_SIZE_N)[None, :]
                
                # Create masks for input boundaries
                mask_in_h = (in_row_offsets >= 0) & (in_row_offsets < height)
                mask_in_w = (in_col_offsets >= 0) & (in_col_offsets < width)
                mask_in = mask_in_h & mask_in_w & mask_m[:, None] & mask_n[None, :]
                
                # Load input values
                # Compute flat index for input
                input_indices = (
                    pid_batch * (in_channels * height * width) +
                    in_channel_offsets[None, None, :] * (height * width) +
                    in_row_offsets[:, :, None] * width +
                    in_col_offsets[:, :, None]
                )
                
                # Reshape for proper broadcasting
                input_block = tl.load(
                    x_ptr + input_indices,
                    mask=mask_in[:, :, None] & mask_k[None, None, :],
                    other=0.0
                )
                
                # Load corresponding weight values
                weight_indices = (
                    tl.arange(0, BLOCK_SIZE_M)[:, None, None] * 0 +  # broadcast across output rows
                    tl.arange(0, BLOCK_SIZE_N)[None, :, None] * 0 +  # broadcast across output cols
                    in_channel_offsets[None, None, :] * (kernel_h * kernel_w) +
                    kh * kernel_w +
                    kw
                )
                weight_block = tl.load(
                    w_ptr + weight_indices,
                    mask=mask_k[None, None, :],
                    other=0.0
                )
                
                # Accumulate multiplication
                output += tl.sum(input_block * weight_block, axis=2)
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N) % out_channels
        bias = tl.load(b_ptr + bias_offsets, mask=mask_m & mask_n)
        output += bias.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
    
    # Store output
    out_indices = (
        pid_batch * (out_channels * out_height * out_width) +
        tl.arange(0, BLOCK_SIZE_M)[:, None] * (out_width) +
        tl.arange(0, BLOCK_SIZE_N)[None, :]
    )
    tl.store(
        out_ptr + out_indices,
        output.to(x_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor:
    """
    Performs 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        groups: Number of groups (must be 1 for this implementation)
        
    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    stride_h = stride_w = stride if isinstance(stride, int) else stride
    pad_h = pad_w = padding if isinstance(padding, int) else padding
    dilation_h = dilation_w = dilation if isinstance(dilation, int) else dilation
    
    out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions
    BLOCK_SIZE_M = 8  # Block size for output rows
    BLOCK_SIZE_N = 8  # Block size for output columns
    BLOCK_SIZE_K = 8  # Block size for channels
    
    # Grid: (batch, out_height_blocks, out_width_blocks)
    grid = (
        batch_size,
        (out_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width,
        out_height, out_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.Conv2d)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights using Kaiming initialization (same as nn.Conv2d default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )


import math