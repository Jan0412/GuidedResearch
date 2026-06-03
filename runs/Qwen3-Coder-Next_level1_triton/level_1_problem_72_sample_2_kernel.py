import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, D, H, W)
    w_ptr,  # Weight tensor pointer (C_in, C_out // groups, kD, kH, kW)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    groups: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_depth: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    out_pad_d: tl.constexpr,
    out_pad_h: tl.constexpr,
    out_pad_w: tl.constexpr,
    output_depth: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 32,
):
    # Calculate which output element this program instance handles
    # Each block processes a chunk of output elements
    pid = tl.program_id(0)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    # Compute indices for output tensor
    # Output shape: (B, C_out, D_out, H_out, W_out)
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < total_elements
    
    # Extract individual indices from flattened index
    # W_out is fastest changing, then H_out, D_out, C_out, B
    w_idx = out_idx % output_width
    tmp = out_idx // output_width
    h_idx = tmp % output_height
    tmp = tmp // output_height
    d_idx = tmp % output_depth
    tmp = tmp // output_depth
    c_out_idx = tmp % out_channels
    b_idx = tmp // out_channels
    
    # Initialize accumulator for this output element
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Calculate input channel group for this output channel
    c_in_group_size = in_channels // groups
    group_idx = c_out_idx // c_in_group_size
    c_in_start = group_idx * c_in_group_size
    
    # Calculate kernel offsets
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate corresponding input position
                in_d = d_idx - k_d + pad_d
                in_h = h_idx - k_h + pad_h
                in_w = w_idx - k_w + pad_w
                
                # Check if input position is within valid range (accounting for stride)
                valid = (
                    (in_d % stride_d == 0) & 
                    (in_h % stride_h == 0) & 
                    (in_w % stride_w == 0)
                )
                in_d = in_d // stride_d
                in_h = in_h // stride_h
                in_w = in_w // stride_w
                
                # Additional bounds check for input
                valid = valid & (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                
                # Calculate input index
                in_idx = (
                    b_idx * (in_channels * depth * height * width) +
                    (c_in_start + tl.arange(0, BLOCK_SIZE) % c_in_group_size)[:, None] * (depth * height * width) +
                    in_d[None, :] * (height * width) +
                    in_h[None, :] * width +
                    in_w[None, :]
                )
                
                # Calculate weight index
                weight_idx = (
                    (c_in_start + tl.arange(0, BLOCK_SIZE)[:, None] % c_in_group_size) * (out_channels * kernel_depth * kernel_height * kernel_width) +
                    c_out_idx[None, :] * (kernel_depth * kernel_height * kernel_width) +
                    k_d * (kernel_height * kernel_width) +
                    k_h * kernel_width +
                    k_w
                )
                
                # Load input and weight values with masks
                in_val = tl.load(x_ptr + in_idx, mask=valid[None, :], other=0.0)
                w_val = tl.load(w_ptr + weight_idx, mask=valid[None, :], other=0.0)
                
                # Accumulate product
                acc += tl.sum(in_val * w_val, axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c_out_idx, mask=mask)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc.to(out_ptr.type.element_ty), mask=mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels // groups, kD, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride tuple (stride_d, stride_h, stride_w)
        padding: Padding tuple (pad_d, pad_h, pad_w)
        output_padding: Output padding tuple (out_pad_d, out_pad_h, out_pad_w)
        groups: Number of groups
    
    Returns:
        Output tensor of shape (batch_size, out_channels, D_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, depth, height, width = x.shape
    _, out_channels, kernel_depth, kernel_height, kernel_width = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    # Calculate output dimensions
    output_depth = (depth - 1) * stride_d - 2 * pad_d + kernel_depth + out_pad_d
    output_height = (height - 1) * stride_h - 2 * pad_h + kernel_height + out_pad_h
    output_width = (width - 1) * stride_w - 2 * pad_w + kernel_width + out_pad_w
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, 
                      dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    # Configure kernel launch
    BLOCK_SIZE = 32
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    triton_conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, groups,
        depth, height, width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        output_depth, output_height, output_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using Triton kernel.
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
        
        # Initialize weights using Xavier initialization
        kernel_depth, kernel_height, kernel_width = kernel_size
        fan_in = in_channels * kernel_depth * kernel_height * kernel_width
        fan_out = out_channels * kernel_depth * kernel_height * kernel_width // groups
        
        # Initialize weight tensor
        weight_std = (2.0 / (fan_in + fan_out)) ** 0.5
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_depth, kernel_height, kernel_width) * weight_std
        )
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernel for 3D transposed convolution.
        """
        # Ensure x is contiguous
        x = x.contiguous()
        
        # Call Triton implementation
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, 
            self.groups
        )