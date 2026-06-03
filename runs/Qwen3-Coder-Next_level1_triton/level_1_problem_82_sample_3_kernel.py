import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (in_channels, 1, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (in_channels,) or None
    out_ptr,  # Output tensor: (batch, in_channels, out_h, out_w)
    batch_size, in_channels, height, width, out_h, out_w,
    kernel_h, kernel_w, stride, padding,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Program IDs: 
    #   pid_b: batch index
    #   pid_c: channel index (we process multiple channels per block for efficiency)
    #   pid_h, pid_w: spatial indices in output
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute start indices for input and output
    out_row = pid_h * stride
    out_col = pid_w * stride

    # Input offsets for this output position
    # We'll process channels in blocks of BLOCK_SIZE_C
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < in_channels

    # Prepare output accumulator
    output = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)

    # Iterate over kernel positions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute input position
            in_row = out_row + kh - padding
            in_col = out_col + kw - padding

            # Check bounds for input
            valid_h = (in_row >= 0) & (in_row < height)
            valid_w = (in_col >= 0) & (in_col < width)
            valid = valid_h & valid_w & c_mask

            if tl.any(valid):
                # Compute input pointer
                x_batch_offset = pid_b * in_channels * height * width
                x_channel_offset = c_offsets * height * width
                x_row_offset = in_row * width
                x_col_offset = in_col
                
                x_idx = x_batch_offset + x_channel_offset + x_row_offset + x_col_offset
                x_vals = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

                # Compute weight pointer: weights are stored as (in_channels, 1, kernel_h, kernel_w)
                w_channel_offset = c_offsets * kernel_h * kernel_w
                w_kernel_offset = kh * kernel_w + kw
                w_idx = w_channel_offset + w_kernel_offset
                w_vals = tl.load(w_ptr + w_idx, mask=c_mask, other=0.0)

                # Accumulate convolution
                output += x_vals * w_vals

    # Add bias if available
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + c_offsets, mask=c_mask)
        output += b_vals

    # Store result
    out_batch_offset = pid_b * in_channels * out_h * out_w
    out_channel_offset = c_offsets * out_h * out_w
    out_row_offset = pid_h * out_w
    out_col_offset = pid_w
    out_idx = out_batch_offset + out_channel_offset + out_row_offset + out_col_offset

    out_mask = c_mask
    tl.store(out_ptr + out_idx, output.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: input tensor of shape (batch_size, in_channels, height, width)
        weight: weight tensor of shape (in_channels, 1, kernel_h, kernel_w)
        bias: optional bias tensor of shape (in_channels,)
        stride: stride for the convolution
        padding: padding applied to the input
    """
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Kernel launch configuration
    BLOCK_SIZE_C = 32  # Channels per block
    BLOCK_SIZE_H = 8   # Output height per block
    BLOCK_SIZE_W = 8   # Output width per block
    
    grid = (
        batch_size,           # batch dimension
        triton.cdiv(in_channels, BLOCK_SIZE_C),  # channels dimension
        out_h // BLOCK_SIZE_H + (1 if out_h % BLOCK_SIZE_H != 0 else 0),  # height dimension
        out_w // BLOCK_SIZE_W + (1 if out_w % BLOCK_SIZE_W != 0 else 0),  # width dimension
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width, out_h, out_w,
        kernel_h, kernel_w, stride, padding,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        HAS_BIAS=bias is not None,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Optimized with custom Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create weight tensor with proper shape for depthwise conv
        # (in_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        # Handle bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Ensure input is contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Use Triton kernel for depthwise convolution
        out = triton_depthwise_conv2d(
            x, weight, self.bias, 
            stride=self.stride, padding=self.padding
        )
        
        return out