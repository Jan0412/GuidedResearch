import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,            # Input tensor pointer (B, C, H, W)
    w_ptr,            # Weight tensor pointer (C, kH, kW)
    b_ptr,            # Bias tensor pointer (C,) or nullptr
    out_ptr,          # Output tensor pointer (B, C, H_out, W_out)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Blocks over batch and channel (B*C)
    BLOCK_SIZE_N: tl.constexpr,  # Blocks over output spatial positions
    BLOCK_SIZE_KH: tl.constexpr,  # Blocks over kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Blocks over kernel width
):
    # Compute batch and channel index
    bc_id = tl.program_id(0)
    batch_idx = bc_id // C
    channel_idx = bc_id % C
    
    # Compute output spatial position
    out_idx = tl.program_id(1)
    h_out = out_idx // W_out
    w_out = out_idx % W_out
    
    # Compute input starting position for this output position
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding
    
    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE_M,), tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kH):
        h_in = h_in_start + kh
        h_in_valid = (h_in >= 0) & (h_in < H)
        
        for kw in range(kW):
            w_in = w_in_start + kw
            w_in_valid = (w_in >= 0) & (w_in < W)
            
            # Load input values for all batch-channel pairs
            if h_in_valid and w_in_valid:
                # Calculate input pointer offset for this (h_in, w_in)
                input_offset = batch_idx * (C * H * W) + channel_idx * (H * W) + h_in * W + w_in
                x_vals = tl.load(x_ptr + input_offset)
            else:
                x_vals = tl.zeros((BLOCK_SIZE_M,), tl.float32)
            
            # Load weight for this channel
            weight_offset = channel_idx * (kH * kW) + kh * kW + kw
            w_val = tl.load(w_ptr + weight_offset)
            
            # Accumulate convolution
            acc += x_vals * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = channel_idx
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    out_offset = batch_idx * (C * H_out * W_out) + channel_idx * (H_out * W_out) + h_out * W_out + w_out
    tl.store(out_ptr + out_offset, acc)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        # Ensure inputs are contiguous and on GPU
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Get dimensions
        B, C, H, W = x.shape
        kH, kW = weight.shape[-2], weight.shape[-1]
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        
        # Create output tensor
        out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Check if we're using FP32
        assert x.dtype == torch.float32, "This implementation is optimized for FP32"
        
        # Grid configuration
        # First dimension: batch * channels
        # Second dimension: output spatial positions
        bc_blocks = B * C
        spatial_blocks = H_out * W_out
        
        # Use reasonable block sizes for FP32
        BLOCK_SIZE_M = 1  # Since each program processes one batch-channel pair
        BLOCK_SIZE_N = 128  # Process multiple output positions per program
        BLOCK_SIZE_KH = 1
        BLOCK_SIZE_KW = 1
        
        # Grid definition
        grid = lambda meta: (
            bc_blocks,
            (spatial_blocks + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, out,
            B=B, C=C, H=H, W=W,
            kH=kH, kW=kW,
            stride=stride, padding=padding,
            H_out=H_out, W_out=W_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_size = (B, C, H, W)
        ctx.kernel_size = (kH, kW)
        ctx.output_size = (H_out, W_out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implementing backward for this optimization task
        raise NotImplementedError("Backward pass not implemented for custom Triton depthwise conv")


def depthwise_conv2d_triton(x, weight, bias=None, stride=1, padding=0):
    return TritonDepthwiseConv2d.apply(x, weight, bias, stride, padding)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # For depthwise convolution, out_channels must equal in_channels
        # but we keep the parameter for compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight and bias parameters
        # Note: For depthwise convolution, groups=in_channels, so each input channel has its own filter
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization for depthwise conv
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is in the correct format for depthwise convolution
        assert x.shape[1] == self.in_channels, "Input channels mismatch"
        
        # Use custom Triton implementation
        return depthwise_conv2d_triton(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding
        )