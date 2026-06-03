import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    x_ptr,           # Input tensor: (batch, in_channels, H, W)
    w_ptr,           # Weight tensor: (in_channels, out_channels, kH, kW)
    b_ptr,           # Bias tensor: (out_channels,)
    out_ptr,         # Output tensor: (batch, out_channels, out_H, out_W)
    batch_size, 
    in_channels,
    out_channels,
    in_h, in_w,
    out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    n_groups,
    # Strides for each tensor
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_i, w_stride_o, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Program IDs for batch and output channel
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_c = pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels in groups
    for ic_group in range(0, in_channels, n_groups):
        # Process each group
        for group_idx in range(n_groups):
            ic = ic_group + group_idx
            
            # Calculate input position corresponding to this output
            # For transposed convolution: in_h = (out_h - pad_h + k_h - 1 + dil_h * (k_h - 1)) // stride_h
            # But we need to check which input positions contribute to this output
            for kh in range(k_h):
                for kw in range(k_w):
                    # Calculate input position for this kernel position
                    in_h_pos = (out_h + pad_h - kh * dil_h) // stride_h
                    in_w_pos = (out_w + pad_w - kw * dil_w) // stride_w
                    
                    # Check if input position is valid
                    if in_h_pos >= 0 and in_h_pos < in_h and \
                       in_w_pos >= 0 and in_w_pos < in_w and \
                       (out_h + pad_h - kh * dil_h) % stride_h == 0 and \
                       (out_w + pad_w - kw * dil_w) % stride_w == 0:
                        
                        # Load input value
                        x_offset = pid_b * x_stride_b + ic * x_stride_c + in_h_pos * x_stride_h + in_w_pos * x_stride_w
                        x_val = tl.load(x_ptr + x_offset, mask=pid_b < batch_size)
                        
                        # Load weight value (note: transposed conv uses the kernel as-is, not flipped)
                        w_offset = ic * w_stride_i + out_c * w_stride_o + kh * w_stride_kh + kw * w_stride_kw
                        w_val = tl.load(w_ptr + w_offset, mask=(pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < out_channels)
                        
                        # Accumulate
                        accumulator += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = out_c * (1 if out_channels == 1 else 0)
        bias = tl.load(b_ptr + out_c, mask=out_c < out_channels)
        accumulator += bias
    
    # Store result
    out_offset = pid_b * out_stride_b + out_c * out_stride_c + out_h * out_stride_h + out_w * out_stride_w
    tl.store(out_ptr + out_offset, accumulator, mask=(pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < out_channels)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Parse parameters
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weight.shape
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        
        # Calculate output dimensions
        out_h = (in_h - 1) * stride_h - 2 * pad_h + dil_h * (k_h - 1) + 1
        out_w = (in_w - 1) * stride_w - 2 * pad_w + dil_w * (k_w - 1) + 1
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        
        # Check if bias is provided
        has_bias = bias is not None
        
        # Configure grid
        BLOCK_SIZE_M = 8  # Channels per block
        BLOCK_SIZE_N = 4  # Batch per block (not used in this implementation)
        
        # Grid dimensions: (batch, channels_per_block_groups, height, width)
        grid = (
            batch_size,
            (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            out_h,
            out_w
        )
        
        # Launch kernel
        transposed_conv2d_kernel[grid](
            x, weight, bias if has_bias else None, out,
            batch_size, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            k_h, k_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=32,  # Not used in this implementation
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        return out


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated using optimized Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)