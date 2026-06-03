import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_depthwise_pointwise_conv2d_kernel(
    # Pointers to input, depthwise weights, pointwise weights, and output
    x_ptr, 
    depthwise_weight_ptr,
    pointwise_weight_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    height, width,
    kernel_size,
    stride,
    padding,
    dilation,
    # Strides
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    depthwise_out_channel_stride, depthwise_kernel_height_stride, depthwise_kernel_width_stride,
    pointwise_in_channel_stride, pointwise_out_channel_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    # Block sizes for tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,  # For depthwise kernel
    BLOCK_OC: tl.constexpr,  # For output channels
    HAS_BIAS: tl.constexpr,
):
    # Compute output spatial position
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1) // (out_channels // BLOCK_OC)  # Group output channels
    pid_h = tl.program_id(1) // (out_channels // BLOCK_OC) % (height // stride)
    pid_w = tl.program_id(2)
    
    # Adjust pid_oc for the actual channel block
    pid_oc = tl.program_id(1) % (out_channels // BLOCK_OC)
    pid_h = tl.program_id(1) // (out_channels // BLOCK_OC)
    
    # Calculate actual output coordinates
    out_h = pid_h * stride
    out_w = pid_w * stride
    
    # Offset for this batch and output channel block
    out_batch_offset = pid_b * out_batch_stride
    out_channel_offset = pid_oc * BLOCK_OC * out_channel_stride
    
    # Initialize accumulator
    outputs = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    
    # Process each input channel that contributes to the output channels
    for ic in range(in_channels):
        # Depthwise convolution: process kernel window
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input position
                in_h = out_h + kh * dilation - padding
                in_w = out_w + kw * dilation - padding
                
                # Check bounds
                if 0 <= in_h < height and 0 <= in_w < width:
                    # Load input value
                    x_offset = (pid_b * x_batch_stride + 
                               ic * x_channel_stride + 
                               in_h * x_height_stride + 
                               in_w * x_width_stride)
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load depthwise weight
                    dw_offset = (ic * depthwise_out_channel_stride + 
                                0 * depthwise_kernel_height_stride + 
                                kh * depthwise_kernel_width_stride + 
                                kw)
                    dw_val = tl.load(depthwise_weight_ptr + dw_offset)
                    
                    # Accumulate depthwise result
                    depthwise_result = x_val * dw_val
                    
                    # Pointwise convolution: multiply with pointwise weights for each output channel
                    for oc in range(BLOCK_OC):
                        pw_offset = (ic * pointwise_in_channel_stride + 
                                    (pid_oc * BLOCK_OC + oc) * pointwise_out_channel_stride)
                        pw_val = tl.load(pointwise_weight_ptr + pw_offset)
                        outputs = tl.where(
                            (pid_oc * BLOCK_OC + oc) < out_channels,
                            outputs + depthwise_result * pw_val,
                            outputs
                        )
    
    # Add bias if present
    if HAS_BIAS:
        bias_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
        bias_mask = bias_offsets < out_channels
        bias_val = tl.load(bias_ptr + bias_offsets, mask=bias_mask)
        outputs = outputs + bias_val
    
    # Store results
    for oc in range(BLOCK_OC):
        oc_idx = pid_oc * BLOCK_OC + oc
        if oc_idx < out_channels:
            out_offset = (out_batch_offset + 
                         oc_idx * out_channel_stride + 
                         pid_h * out_height_stride + 
                         pid_w * out_width_stride)
            tl.store(out_ptr + out_offset, outputs[oc])


def fused_depthwise_pointwise_conv2d(x, depthwise_weight, pointwise_weight, bias, 
                                    stride, padding, dilation):
    """
    Fused implementation of depthwise-separable convolution using Triton.
    """
    batch_size, in_channels, height, width = x.shape
    _, out_channels, _, _ = pointwise_weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (depthwise_weight.shape[2] - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (depthwise_weight.shape[3] - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    BLOCK_H = 1
    BLOCK_W = 16
    BLOCK_OC = 16
    BLOCK_K = 1
    
    # Grid configuration
    grid = (batch_size, 
            (out_channels + BLOCK_OC - 1) // BLOCK_OC * ((out_height + BLOCK_H - 1) // BLOCK_H),
            (out_width + BLOCK_W - 1) // BLOCK_W)
    
    # Strides
    x_batch_stride = x.stride(0)
    x_channel_stride = x.stride(1)
    x_height_stride = x.stride(2)
    x_width_stride = x.stride(3)
    
    depthwise_out_channel_stride = depthwise_weight.stride(0)
    depthwise_kernel_height_stride = depthwise_weight.stride(2)
    depthwise_kernel_width_stride = depthwise_weight.stride(3)
    
    pointwise_in_channel_stride = pointwise_weight.stride(0)
    pointwise_out_channel_stride = pointwise_weight.stride(1)
    
    out_batch_stride = out.stride(0)
    out_channel_stride = out.stride(1)
    out_height_stride = out.stride(2)
    out_width_stride = out.stride(3)
    
    # Launch kernel
    fused_depthwise_pointwise_conv2d_kernel[grid](
        x, depthwise_weight, pointwise_weight, 
        bias if bias is not None else torch.empty(0, device=x.device),
        out,
        batch_size, in_channels, out_channels,
        height, width,
        depthwise_weight.shape[2],  # kernel_size
        stride, padding, dilation,
        x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
        depthwise_out_channel_stride, depthwise_kernel_height_stride, depthwise_kernel_width_stride,
        pointwise_in_channel_stride, pointwise_out_channel_stride,
        out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC,
        HAS_BIAS=bias is not None
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using optimized Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store original parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create parameters with correct shapes
        # Depthwise: (in_channels, 1, kernel_size, kernel_size)
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        # Pointwise: (out_channels, in_channels, 1, 1)
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights (using Kaiming initialization style)
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return fused_depthwise_pointwise_conv2d(
            x, self.depthwise_weight, self.pointwise_weight, self.bias,
            self.stride, self.padding, self.dilation
        )


import math