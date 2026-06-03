import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, H, W)
    w_ptr,  # Weight tensor: (in_channels, 1, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (in_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    input_h, input_w,
    output_h, output_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_h = tl.program_id(2)  # height block index
    
    # Calculate starting positions
    batch_offset = pid_b * in_channels * input_h * input_w
    channel_offset = pid_c * input_h * input_w
    out_channel_offset = pid_c * output_h * output_w
    
    # Calculate output height position
    out_h_start = pid_h * BLOCK_SIZE_H
    out_h_range = tl.arange(0, BLOCK_SIZE_H)
    out_h_offsets = out_h_start + out_h_range
    out_h_mask = out_h_offsets < output_h
    
    # Process width dimension in chunks
    for pw_start in range(0, output_w, BLOCK_SIZE_W):
        pw_range = tl.arange(0, BLOCK_SIZE_W)
        pw_offsets = pw_start + pw_range
        pw_mask = pw_offsets < output_w
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Compute the convolution
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                in_h = out_h_offsets * stride_h - padding_h + kh * dilation_h
                in_w = pw_offsets * stride_w - padding_w + kw * dilation_w
                
                # Check if input position is valid
                h_valid = (in_h >= 0) & (in_h < input_h)
                w_valid = (in_w >= 0) & (in_w < input_w)
                valid_mask = h_valid & w_valid
                
                # Calculate input pointer offset
                in_h_offsets = in_h * input_w
                in_offsets = batch_offset + channel_offset + in_h_offsets + in_w
                
                # Load input values
                x_vals = tl.load(x_ptr + in_offsets, mask=valid_mask, other=0.0)
                
                # Load weight value
                w_offset = pid_c * (kernel_h * kernel_w) + kh * kernel_w + kw
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_vals * w_val
        
        # Add bias if provided
        if b_ptr is not None:
            b_offset = pid_c
            bias = tl.load(b_ptr + b_offset)
            acc += bias
        
        # Store output
        out_offsets = (batch_offset + out_channel_offset + 
                      out_h_offsets[:, None] * output_w + 
                      pw_offsets[None, :])
        tl.store(out_ptr + out_offsets, acc, mask=out_h_mask[:, None] & pw_mask[None, :])


class TritonDepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride_h, stride_w, padding_h, padding_w, 
                dilation_h, dilation_w):
        
        # Get dimensions
        batch_size, in_channels, input_h, input_w = x.shape
        _, _, kernel_h, kernel_w = weight.shape
        
        # Calculate output dimensions
        output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Prepare output tensor
        out = torch.empty((batch_size, in_channels, output_h, output_w), 
                         dtype=x.dtype, device=x.device)
        
        # Grid configuration
        grid = lambda meta: (
            batch_size,
            in_channels,
            triton.cdiv(output_h, meta['BLOCK_SIZE_H']),
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, in_channels,  # out_channels == in_channels for depthwise
            input_h, input_w,
            output_h, output_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            BLOCK_SIZE_C=1,  # Fixed for depthwise
            BLOCK_SIZE_H=8,
            BLOCK_SIZE_W=16,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.input_size = (input_h, input_w)
        ctx.output_size = (output_h, output_w)
        ctx.kernel_size = (kernel_h, kernel_w)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (padding_h, padding_w)
        ctx.dilation = (dilation_h, dilation_w)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        input_h, input_w = ctx.input_size
        output_h, output_w = ctx.output_size
        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        padding_h, padding_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        
        # Gradient w.r.t. input
        # This would require implementing backward kernels, but for simplicity
        # we'll fall back to PyTorch for backward pass since implementing
        # full backward for depthwise conv is complex
        
        # For now, use PyTorch's native backward
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            weight_ = weight.detach().requires_grad_(True)
            bias_ = bias.detach().requires_grad_(True) if bias is not None else None
            
            # Manual implementation of forward using PyTorch to get gradients
            out = torch.nn.functional.conv2d(
                x_, weight_, bias_, 
                stride=(stride_h, stride_w),
                padding=(padding_h, padding_w),
                dilation=(dilation_h, dilation_w),
                groups=x_.size(1)
            )
            
            grad_input, grad_weight, grad_bias = torch.autograd.grad(
                outputs=out, inputs=[x_, weight_, bias_],
                grad_outputs=grad_output,
                retain_graph=False,
                allow_unused=True
            )
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def depthwise_conv2d_triton(x, weight, bias=None,
                           stride_h=1, stride_w=1, 
                           padding_h=0, padding_w=0,
                           dilation_h=1, dilation_w=1):
    return TritonDepthwiseConv2dFunction.apply(
        x, weight, bias,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w
    )


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # For depthwise convolution, groups = in_channels
        # Create weight tensor: (in_channels, 1, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = kernel_size_h * kernel_size_w
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        """
        return depthwise_conv2d_triton(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )