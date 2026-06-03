import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, height, width)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    height, width,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial positions
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output channel block start
    out_channel_start = pid_m * BLOCK_SIZE_M
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
    out_channel_mask = out_channel_offsets < out_channels
    
    # Calculate spatial position block start
    spatial_start = pid_n * BLOCK_SIZE_N
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_N)
    spatial_mask = spatial_offsets < out_h * out_w
    
    # Create meshgrid for output positions
    oh = spatial_offsets // out_w
    ow = spatial_offsets % out_w
    
    # Calculate input position for top-left of kernel
    ih_base = oh * stride_h - pad_h
    iw_base = ow * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                ih = ih_base + kh * dilation_h
                iw = iw_base + kw * dilation_w
                
                # Check if input position is valid
                valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                
                # Load input values
                x_indices = pid_b * (in_channels * height * width) + \
                           ic * (height * width) + \
                           ih * width + iw
                x_val = tl.load(x_ptr + x_indices, mask=valid, other=0.0)
                
                # Load weight values
                w_indices = (out_channel_offsets[:, None] * (in_channels * kernel_h * kernel_w) +
                            ic * (kernel_h * kernel_w) +
                            kh * kernel_w + kw)
                w_val = tl.load(w_ptr + w_indices, mask=out_channel_mask[:, None], other=0.0)
                
                # Accumulate
                acc += tl.sum(w_val * x_val, axis=1)
    
    # Apply bias if present
    if b_ptr is not None:
        b_indices = out_channel_offsets
        bias = tl.load(b_ptr + b_indices, mask=out_channel_mask, other=0.0)
        acc += bias
    
    # Convert to output type and store
    acc = acc.to(tl.float32)
    
    # Calculate output position
    out_indices = pid_b * (out_channels * out_h * out_w) + \
                 out_channel_offsets[:, None] * (out_h * out_w) + \
                 oh[None, :] * out_w + ow[None, :]
    
    # Mask for valid output positions
    out_mask = out_channel_mask[:, None] & spatial_mask[None, :]
    
    tl.store(out_ptr + out_indices, acc, mask=out_mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)
        dilation: Tuple (dilation_h, dilation_w)
        groups: Number of groups (must be 1 for this implementation)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    assert groups == 1, "Triton conv2d only supports groups=1"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Handle bias
    bias_ptr = bias.contiguous() if bias is not None else None
    
    # Set up kernel parameters
    BLOCK_SIZE_M = 32  # Block size for output channels
    BLOCK_SIZE_N = 64  # Block size for spatial positions
    BLOCK_SIZE_K = 8   # Block size for accumulation (can be tuned)
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(out_h * out_w, BLOCK_SIZE_N),
        batch_size
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias_ptr, out,
        batch_size, in_channels, out_channels,
        height, width,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the Triton convolution kernel.
        """
        # Ensure x is on the same device as weight
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for parameter initialization
import math