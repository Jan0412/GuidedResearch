import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for depthwise convolution with bias support
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer: (batch, channels, H, W)
    w_ptr,  # Weight tensor pointer: (channels, 1, kH, kW)
    b_ptr,  # Bias tensor pointer: (channels,) - optional
    y_ptr,  # Output tensor pointer: (batch, channels, H_out, W_out)
    batch_size, n_channels, in_h, in_w, 
    out_h, out_w,
    kH, kW,
    stride: tl.constexpr, padding: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr = 8, BLOCK_W: tl.constexpr = 8, 
    BLOCK_KH: tl.constexpr = 3, BLOCK_KW: tl.constexpr = 3,
):
    # Compute batch index
    batch_id = tl.program_id(0)
    # Compute channel index
    channel_id = tl.program_id(1)
    
    # Compute output spatial position
    out_h_id = tl.program_id(2) // (out_w // BLOCK_W)
    out_w_id = tl.program_id(2) % (out_w // BLOCK_W)
    
    # Compute input spatial position corresponding to output position
    in_h_start = out_h_id * stride - padding
    in_w_start = out_w_id * stride - padding
    
    # Compute offsets for output
    out_h_offsets = tl.arange(0, BLOCK_H)
    out_w_offsets = tl.arange(0, BLOCK_W)
    out_h_mask = (out_h_id * BLOCK_H + out_h_offsets) < out_h
    out_w_mask = (out_w_id * BLOCK_W + out_w_offsets) < out_w
    out_h_indices = out_h_id * BLOCK_H + out_h_offsets
    out_w_indices = out_w_id * BLOCK_W + out_w_offsets
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(kH):
        for kw in range(kW):
            # Compute input position for this kernel element
            in_h_idx = in_h_start + kh
            in_w_idx = in_w_start + kw
            
            # Check bounds for input
            h_valid = (in_h_idx >= 0) & (in_h_idx < in_h)
            w_valid = (in_w_idx >= 0) & (in_w_idx < in_w)
            valid = h_valid[:, None] & w_valid[None, :]
            
            # Compute input pointers
            # Input is (batch, channels, H, W), so for batch_id, channel_id:
            x_offset = batch_id * (n_channels * in_h * in_w) + \
                       channel_id * (in_h * in_w) + \
                       in_h_idx * in_w + in_w_idx
            
            # Compute weight pointer for this kernel element
            # Weight is (channels, 1, kH, kW), so for channel_id:
            w_offset = channel_id * (kH * kW) + \
                       kh * kW + kw
            
            # Load input and weight values
            x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
            w_val = tl.load(w_ptr + w_offset)
            
            # Broadcast values to (BLOCK_H, BLOCK_W)
            x_val = x_val[:, :, None, None] if BLOCK_H > 1 else x_val[None, None, :, :]
            w_val = w_val[None, None, :, :]
            
            # Multiply and accumulate
            acc += tl.where(valid, x_val * w_val, 0.0).sum(axis=2).sum(axis=2)
    
    # Add bias if enabled
    if HAS_BIAS:
        bias_val = tl.load(b_ptr + channel_id)
        acc += bias_val
    
    # Convert to output dtype and store
    acc = acc.to(y_ptr.dtype.element_ty)
    
    # Compute output pointer offset
    y_offset = batch_id * (n_channels * out_h * out_w) + \
               channel_id * (out_h * out_w) + \
               out_h_id * BLOCK_H * out_w + out_w_id * BLOCK_W
    
    # Create mask for output
    out_mask = out_h_mask[:, None] & out_w_mask[None, :]
    
    # Store result
    tl.store(y_ptr + y_offset, acc, mask=out_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor (batch_size, in_channels, height, width)
        weight: Weight tensor (in_channels, 1, kernel_height, kernel_width)
        bias: Optional bias tensor (in_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
    
    Returns:
        Output tensor (batch_size, in_channels, height_out, width_out)
    """
    batch_size, n_channels, in_h, in_w = x.shape
    _, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - kH) // stride + 1
    out_w = (in_w + 2 * padding - kW) // stride + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    y = torch.empty((batch_size, n_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions for kernel launch
    # Grid: (batch, channels, (out_h + BLOCK_H - 1) // BLOCK_H * (out_w + BLOCK_W - 1) // BLOCK_W)
    BLOCK_H = 8
    BLOCK_W = 8
    
    grid = (
        batch_size,
        n_channels,
        (out_h + BLOCK_H - 1) // BLOCK_H * (out_w + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, n_channels, in_h, in_w,
        out_h, out_w,
        kH, kW,
        stride=stride, padding=padding,
        HAS_BIAS=bias is not None,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return y


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel,
    optimized with Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weight and bias as in the original Conv2d
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)