import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to input/output tensors
    x_ptr, y_ptr, out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels, height, width,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups: tl.constexpr,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_g, y_stride_out, y_stride_in, y_stride_kh, y_stride_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_g = tl.program_id(3)
    
    # Calculate output block start positions
    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W
    
    # Create offset ranges
    h_offsets = tl.arange(0, BLOCK_H)
    w_offsets = tl.arange(0, BLOCK_W)
    h_mask = (out_h_start + h_offsets) < out_h
    w_mask = (out_w_start + w_offsets) < out_w
    
    # Initialize accumulator
    out_block = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Process input channels in groups
    # Each group handles out_channels // groups output channels
    group_out_channels = out_channels // groups
    group_in_channels = in_channels // groups
    
    # Start channel index for this group
    group_start_channel = pid_g * group_out_channels
    
    # Iterate over input channels in the group
    for c_in in range(group_in_channels):
        c_in_global = pid_g * group_in_channels + c_in
        
        # Calculate corresponding input position
        # For transposed convolution: out_h = (in_h - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + 1
        # So in_h = (out_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        # But we need to iterate over all valid in_h, in_w that contribute to the output
        
        # For each output position, iterate over kernel positions
        kh_offsets = tl.arange(0, BLOCK_KH)
        kw_offsets = tl.arange(0, BLOCK_KW)
        
        # Calculate input positions that contribute to output region
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input position corresponding to this kernel element
                in_h = (out_h_start + h_offsets - dil_h * kh + pad_h) // stride_h
                in_w = (out_w_start + w_offsets - dil_w * kw + pad_w) // stride_w
                
                # Check if input position is valid
                h_valid = (in_h >= 0) & (in_h < height) & h_mask[:, None]
                w_valid = (in_w >= 0) & (in_w < width) & w_mask[None, :]
                valid_mask = h_valid & w_valid
                
                # Load input values where valid
                in_h_flat = in_h * x_stride_h + in_w * x_stride_w
                x_offsets = pid_b * x_stride_b + c_in_global * x_stride_c + in_h_flat
                x_val = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
                
                # Load kernel values
                y_h_offset = kh * y_stride_kh + kw * y_stride_kw
                y_offsets = pid_g * y_stride_g + (group_start_channel - pid_g * group_out_channels) * y_stride_out + c_in * y_stride_in + y_h_offset
                y_val = tl.load(y_ptr + y_offsets, mask=kh == tl.arange(0, BLOCK_KH)[None, :] if BLOCK_KH > 1 else True, other=0.0)
                
                # Accumulate
                out_block += x_val * y_val
    
    # Store result
    out_offsets = pid_b * out_stride_b + group_start_channel * out_stride_c + (out_h_start + h_offsets[:, None]) * out_stride_h + (out_w_start + w_offsets[None, :]) * out_stride_w
    tl.store(out_ptr + out_offsets, out_block, mask=h_mask[:, None] & w_mask[None, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, out_channels // groups, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
    """
    batch_size, in_channels, height, width = x.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_channels = weight.shape[1] * groups
    
    # Calculate output dimensions
    out_h = (height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + 1
    out_w = (width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Calculate strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    y_stride_g = weight.stride(0)  # in_channels stride
    y_stride_out = weight.stride(1)  # out_channels_per_group stride
    y_stride_in = weight.stride(2)  # kernel_h stride
    y_stride_kw = weight.stride(3)  # kernel_w stride
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()
    
    # Grid dimensions: (batch, out_h_blocks, out_w_blocks, groups)
    BLOCK_H, BLOCK_W = 8, 8
    BLOCK_C = 16
    BLOCK_KH, BLOCK_KW = 4, 4
    
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_h, meta["BLOCK_H"]),
        triton.cdiv(out_w, meta["BLOCK_W"]),
        groups,
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width,
        out_h, out_w,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        y_stride_g, y_stride_out, y_stride_in, y_stride_kw,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C, BLOCK_KH=BLOCK_KH, BLOCK_KW=BLOCK_KW,
    )
    
    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}')


import math