import torch
import torch.nn as nn
import triton
import triton.language as tl

# Custom Triton kernel for depthwise convolution
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Depthwise filter (C, 1, kH, kW)
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
):
    # Get batch, channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute the output position
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Output offset calculation
    out_offsets_h = tl.arange(0, BLOCK_SIZE_H)
    out_offsets_w = tl.arange(0, BLOCK_SIZE_W)
    out_h, out_w = tl.meshgrid(out_offsets_h, out_offsets_w)
    out_h = out_h.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W,)
    out_w = out_w.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W,)
    
    # Create masks for output bounds
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h & mask_w
    
    # Compute input position for each output position
    # Input position = (output_pos - padding) / stride - (kernel_size - 1) * dilation / stride
    # But more precisely: input_pos = (output_pos * stride - padding + dilation * (kernel_pos - 1))
    input_h = out_h * stride - padding + dilation * (tl.arange(0, kH).reshape(kH, 1) - 1)
    input_w = out_w * stride - padding + dilation * (tl.arange(0, kW).reshape(1, kW) - 1)
    
    # Flatten kernel indices
    kh_offsets = tl.arange(0, kH).reshape(kH, 1)
    kw_offsets = tl.arange(0, kW).reshape(1, kW)
    
    # Compute input and weight offsets
    x_base_offset = batch_idx * C * H * W + channel_idx * H * W
    w_base_offset = channel_idx * kH * kW
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H * BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(kH):
        for kw in range(kW):
            h_pos = input_h + kh * dilation
            w_pos = input_w + kw * dilation
            
            # Check if positions are within input bounds
            h_mask = (h_pos >= 0) & (h_pos < H)
            w_mask = (w_pos >= 0) & (w_pos < W)
            valid_mask = h_mask & w_mask
            
            # Compute flattened indices
            x_indices = x_base_offset + h_pos * W + w_pos
            w_indices = w_base_offset + kh * kW + kw
            
            # Load values
            x_vals = tl.load(x_ptr + x_indices, mask=valid_mask, other=0.0)
            w_vals = tl.load(w_ptr + w_indices)
            
            # Accumulate
            acc += x_vals * w_vals
    
    # Store result
    out_indices = batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + out_h * W_out + out_w
    tl.store(out_ptr + out_indices, acc.to(x_ptr.dtype.element_ty), mask=mask)


# Custom Triton kernel for pointwise convolution (1x1 conv)
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Pointwise filter (C_out, C, 1, 1)
    b_ptr,  # Bias (C_out) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    # Output positions
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Create offsets for output positions
    out_h_offsets = tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = tl.arange(0, BLOCK_SIZE_W)
    out_h, out_w = tl.meshgrid(out_h_offsets, out_w_offsets)
    out_h = out_h.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W,)
    out_w = out_w.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W,)
    
    # Masks
    mask_h = out_h < H
    mask_w = out_w < W
    mask = mask_h & mask_w
    
    # Compute base offsets
    x_base_offset = batch_idx * C * H * W
    out_base_offset = batch_idx * C_out * H * W + out_channel_idx * H * W
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H * BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_indices = c_offset + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_indices < C
        
        # Get input values: x[batch, c, h, w]
        # Reshape for broadcasting
        x_indices = x_base_offset + c_indices[:, None] * H * W + out_h[None, :] * W + out_w[None, :]
        x_vals = tl.load(x_ptr + x_indices, mask=c_mask[:, None] & mask[None, :], other=0.0)
        
        # Get weights: w[out_c, c, 0, 0]
        w_indices = out_channel_idx * C + c_indices
        w_vals = tl.load(w_ptr + w_indices, mask=c_mask, other=0.0)
        
        # Accumulate: sum over c
        acc += tl.sum(x_vals * w_vals[:, None], axis=0)
    
    # Add bias if available
    if has_bias:
        bias_val = tl.load(b_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    out_indices = out_base_offset + out_h * W + out_w
    tl.store(out_ptr + out_indices, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_depthwise_conv(x, weight, stride=1, padding=0, dilation=1):
    """
    Performs depthwise convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Depthwise kernel of shape (C, 1, kH, kW)
        stride, padding, dilation: Convolution parameters
    
    Returns:
        Output tensor of shape (B, C, H_out, W_out)
    """
    B, C, H, W = x.shape
    _, _, kH, kW = weight.shape
    
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Determine block sizes
    BLOCK_SIZE_H = min(16, H_out)
    BLOCK_SIZE_W = min(16, W_out)
    
    # Grid configuration
    grid = (B, C, (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, out,
        B, C, H, W, kH, kW,
        stride, padding, dilation, H_out, W_out,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


def triton_pointwise_conv(x, weight, bias=None):
    """
    Performs pointwise convolution (1x1 conv) using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Pointwise kernel of shape (C_out, C, 1, 1)
        bias: Optional bias tensor of shape (C_out,)
    
    Returns:
        Output tensor of shape (B, C_out, H, W)
    """
    B, C, H, W = x.shape
    C_out, _, _, _ = weight.shape
    
    # Create output tensor
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Determine block sizes
    BLOCK_SIZE_H = min(16, H)
    BLOCK_SIZE_W = min(16, W)
    BLOCK_SIZE_C = min(32, C)
    
    # Grid configuration
    grid = (B, C_out, (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    has_bias = bias is not None
    pointwise_conv2d_kernel[grid](
        x, weight, bias if has_bias else None, out,
        B, C, H, W, C_out, has_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create depthwise convolution layer
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        
        # Create pointwise convolution layer
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        # Perform depthwise convolution using Triton kernel
        x = triton_depthwise_conv(
            x, 
            self.depthwise.weight,
            stride=self.depthwise.stride[0],
            padding=self.depthwise.padding[0],
            dilation=self.depthwise.dilation[0]
        )
        
        # Perform pointwise convolution using Triton kernel
        x = triton_pointwise_conv(
            x, 
            self.pointwise.weight,
            bias=self.pointwise.bias
        )
        
        return x