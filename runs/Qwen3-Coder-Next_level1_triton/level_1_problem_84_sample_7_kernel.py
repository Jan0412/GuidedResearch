import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,  # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    bias_ptr,  # Bias tensor pointer (optional)
    output_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    height_in,  # Input height
    width_in,  # Input width
    kernel_size,  # Kernel size (square)
    stride,  # Stride
    padding,  # Padding
    height_out,  # Output height
    width_out,  # Output width
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_K: tl.constexpr = 8,  # Block size for kernel dimension
):
    # Get batch, channel, and spatial indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Calculate output spatial position
    h_out_start = h_block * BLOCK_SIZE_H
    w_out_start = w_block * BLOCK_SIZE_W
    
    # Create offsets for output spatial positions
    h_offsets = tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    h_out = h_out_start + h_offsets
    w_out = w_out_start + w_offsets
    
    # Create mask for valid output positions
    h_mask = h_out < height_out
    w_mask = w_out < width_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Compute depthwise convolution
    # For each kernel position
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input position
            h_in = h_out * stride - padding + kh
            w_in = w_out * stride - padding + kw
            
            # Create masks for valid input positions
            h_in_mask = (h_in >= 0) & (h_in < height_in)
            w_in_mask = (w_in >= 0) & (w_in < width_in)
            input_mask = h_in_mask[:, None] & w_in_mask[None, :]
            
            # Calculate input pointer offset
            # Input layout: (batch, channel, height, width)
            input_offset = (
                batch_idx * in_channels * height_in * width_in +
                channel_idx * height_in * width_in +
                h_in[:, None] * width_in + w_in[None, :]
            )
            
            # Load input values
            input_val = tl.load(
                input_ptr + input_offset,
                mask=input_mask,
                other=0.0
            )
            
            # Load weight value (weight layout: out_channels, 1, kernel_h, kernel_w)
            weight_offset = channel_idx * kernel_size * kernel_size + kh * kernel_size + kw
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate
            output += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
        output += bias
    
    # Store output
    output_offset = (
        batch_idx * in_channels * height_out * width_out +
        channel_idx * height_out * width_out +
        h_out[:, None] * width_out + w_out[None, :]
    )
    tl.store(output_ptr + output_offset, output.to(input_ptr.dtype.element_ty), mask=mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        batch_size, in_channels, height_in, width_in = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        # Calculate output dimensions
        height_out = (height_in + 2 * padding - kernel_h) // stride + 1
        width_out = (width_in + 2 * padding - kernel_w) // stride + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, in_channels, height_out, width_out, 
                           dtype=x.dtype, device=x.device)
        
        # Configure grid
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        grid = (
            batch_size,  # batch
            in_channels,  # channels
            (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # h blocks
            (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,   # w blocks
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, output,
            batch_size, in_channels, height_in, width_in,
            kernel_h, stride, padding, height_out, width_out,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.height_in = height_in
        ctx.width_in = width_in
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - full backward would be more complex
        # For now, fall back to PyTorch for backward pass
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Use PyTorch's implementation for gradient computation
            grad_input = nn.functional.conv_transpose2d(
                grad_output, weight, stride=ctx.stride, padding=ctx.padding,
                output_padding=0, groups=x.size(1), dilation=1
            )
            # Crop to original size
            if ctx.padding > 0:
                grad_input = grad_input[:, :, ctx.padding:ctx.padding+ctx.height_in, 
                                       ctx.padding:ctx.padding+ctx.width_in]
        
        return grad_input, grad_weight, grad_bias, None, None, None


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    return TritonDepthwiseConv2d.apply(x, weight, bias, stride, padding)


class ModelNew(nn.Module):
    """
    Optimized version of depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers for weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      self.stride, self.padding)