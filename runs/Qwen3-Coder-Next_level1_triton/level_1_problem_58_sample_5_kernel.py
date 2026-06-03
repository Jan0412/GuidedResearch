import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.ops.matmul import get_configs


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, depth_in, height_in, width_in)
    w_ptr,  # Weight tensor pointer (in_channels, out_channels, k_d, k_h, k_w)
    b_ptr,  # Bias pointer (out_channels,) or None
    out_ptr,  # Output tensor pointer (batch, out_channels, depth_out, height_out, width_out)
    batch_size, in_channels, out_channels,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,  # Block size for out_channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for in_channels
    BLOCK_SIZE_KD: tl.constexpr,  # Block size for kernel depth
    BLOCK_SIZE_KH: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Block size for kernel width
):
    # Get program IDs for output tensor dimensions
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_depth = tl.program_id(2)
    pid_height = tl.program_id(3)
    pid_width = tl.program_id(4)
    
    # Calculate the output element coordinates
    out_offset = (pid_batch * out_channels * depth_out * height_out * width_out +
                  pid_out_c * depth_out * height_out * width_out +
                  pid_depth * height_out * width_out +
                  pid_height * width_out +
                  pid_width)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        # For transposed convolution, compute which input element contributes to this output
        # The relationship is: out_idx = in_idx * stride + (kernel_idx - 1) - pad + output_pad
        # So: in_idx = (out_idx - (kernel_idx - 1) + pad - output_pad) // stride
        
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    # Compute corresponding input coordinates
                    in_d = pid_depth - kd + pad_d - output_pad_d
                    in_h = pid_height - kh + pad_h - output_pad_h
                    in_w = pid_width - kw + pad_w - output_pad_w
                    
                    # Check if input coordinates are valid (stride must divide evenly)
                    if (in_d >= 0 and in_d < depth_in and 
                        in_h >= 0 and in_h < height_in and 
                        in_w >= 0 and in_w < width_out):
                        # Compute input offset
                        in_offset = (pid_batch * in_channels * depth_in * height_in * width_in +
                                    ic * depth_in * height_in * width_in +
                                    in_d * height_in * width_in +
                                    in_h * width_in +
                                    in_w)
                        
                        # Load input value
                        x_val = tl.load(x_ptr + in_offset)
                        
                        # Load corresponding weight
                        w_offset = (ic * out_channels * k_d * k_h * k_w +
                                   pid_out_c * k_d * k_h * k_w +
                                   kd * k_h * k_w +
                                   kh * k_w +
                                   kw)
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        acc += bias
    
    # Store result
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, k_d, k_h, k_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Tuple (stride_d, stride_h, stride_w)
        padding: Tuple (pad_d, pad_h, pad_w)
        output_padding: Tuple (output_pad_d, output_pad_h, output_pad_w)
        groups: Number of groups (must be 1 for this implementation)
    
    Returns:
        Output tensor of shape (batch, out_channels, depth_out, height_out, width_out)
    """
    assert groups == 1, "Groups > 1 not supported in this Triton implementation"
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    _, out_channels, k_d, k_h, k_w = weight.shape
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    output_pad_d, output_pad_h, output_pad_w = output_padding
    
    # Calculate output dimensions
    depth_out = (depth_in - 1) * stride_d - 2 * pad_d + k_d + output_pad_d
    height_out = (height_in - 1) * stride_h - 2 * pad_h + k_h + output_pad_h
    width_out = (width_in - 1) * stride_w - 2 * pad_w + k_w + output_pad_w
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, 
                     dtype=x.dtype, device=x.device)
    
    # Launch configuration
    # Grid: (batch, out_channels, depth_out, height_out, width_out)
    grid = (batch_size, out_channels, depth_out, height_out, width_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        k_d, k_h, k_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w,
        BLOCK_SIZE_M=1,  # One output channel per block
        BLOCK_SIZE_N=1,  # One input channel per block (no tiling needed)
        BLOCK_SIZE_KD=1,  # Process kernel depth element by element
        BLOCK_SIZE_KH=1,  # Process kernel height element by element
        BLOCK_SIZE_KW=1,  # Process kernel width element by element
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
    
    def _apply(self, fn):
        """Override _apply to ensure parameters are properly moved with the module."""
        super()._apply(fn)
        # Re-initialize parameters to ensure they're on the correct device
        if self.weight is not None:
            self.weight = nn.Parameter(fn(self.weight))
        if self.bias is not None:
            self.bias = nn.Parameter(fn(self.bias))
        return self


import math