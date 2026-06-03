import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor pointer (B, C, H, W)
    w_ptr,              # Weight tensor pointer (C, 1, KH, KW)
    b_ptr,              # Bias pointer (C,) or None
    y_ptr,              # Output tensor pointer (B, C, H_out, W_out)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_KH: tl.constexpr = 3,
    BLOCK_KW: tl.constexpr = 7,
):
    # Get program IDs
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Check bounds for output
    out_h_mask = out_h < H_out
    out_w_mask = out_w < W_out
    out_mask = out_h_mask[:, None] & out_w_mask[None, :]
    
    # Compute input positions for this output
    in_h = out_h * stride_h - padding_h + (tl.arange(0, KH)[None, :, None] * dilation_h)
    in_w = out_w * stride_w - padding_w + (tl.arange(0, KW)[None, None, :] * dilation_w)
    
    # Create masks for input positions
    in_h_mask = (in_h >= 0) & (in_h < H)
    in_w_mask = (in_w >= 0) & (in_w < W)
    in_mask = in_h_mask & in_w_mask
    
    # Compute base pointer offsets for input
    # x_ptr points to B, C, H, W - need to access [b, pid_c, in_h, in_w] for all b
    # We'll process one batch at a time in the kernel for simplicity
    
    # Process each batch
    for b in range(B):
        # Calculate base offset for this batch and channel
        base_offset = b * C * H * W + pid_c * H * W
        
        # Accumulator for convolution
        acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Load weights for this channel
        w_offset = pid_c * KH * KW + tl.arange(0, KH)[:, None] * KW + tl.arange(0, KW)[None, :]
        w = tl.load(w_ptr + w_offset)
        
        # Iterate over kernel positions
        for kh in range(KH):
            for kw in range(KW):
                # Compute input position
                h_pos = out_h * stride_h - padding_h + kh * dilation_h
                w_pos = out_w * stride_w - padding_w + kw * dilation_w
                
                # Create masks
                h_pos_mask = (h_pos >= 0) & (h_pos < H)
                w_pos_mask = (w_pos >= 0) & (w_pos < W)
                mask = h_pos_mask[:, None] & w_pos_mask[None, :]
                
                # Compute input pointer offset
                offset = base_offset + h_pos[:, None] * W + w_pos[None, :]
                
                # Load input values
                x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
                
                # Load weight value
                w_val = w[kh, kw]
                
                # Accumulate
                acc += x_val * w_val
        
        # Store result
        y_offset = b * C * H_out * W_out + pid_c * H_out * W_out + out_h[:, None] * W_out + out_w[None, :]
        
        # Add bias if present
        if b_ptr is not None:
            bias_val = tl.load(b_ptr + pid_c)
            acc += bias_val
        
        tl.store(y_ptr + y_offset, acc.to(tl.float32), mask=out_mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size_h, kernel_size_w, stride_h, stride_w, 
                padding_h, padding_w, dilation_h, dilation_w):
        
        B, C, H, W = x.shape
        KH, KW = kernel_size_h, kernel_size_w
        
        # Calculate output dimensions
        H_out = (H + 2 * padding_h - dilation_h * (KH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * padding_w - dilation_w * (KW - 1) - 1) // stride_w + 1
        
        # Allocate output tensor
        y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Set up kernel launch parameters
        grid = (C, (H_out + 7) // 8, (W_out + 15) // 16)  # One block per channel, and output spatial dimensions
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, y,
            B, C, H, W,
            KH, KW,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            H_out, W_out,
            BLOCK_SIZE_C=16,
            BLOCK_SIZE_H=8,
            BLOCK_SIZE_W=16,
            BLOCK_KH=KH,
            BLOCK_KW=KW
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.input_shape = (B, C, H, W)
        ctx.output_shape = (B, C, H_out, W_out)
        ctx.kernel_params = (kernel_size_h, kernel_size_w, stride_h, stride_w, 
                            padding_h, padding_w, dilation_h, dilation_w)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch for backward pass
        # In a production implementation, you would also implement backward kernels
        x, weight, bias = ctx.saved_tensors
        B, C, H, W = ctx.input_shape
        KH, KW = ctx.kernel_params[0], ctx.kernel_params[1]
        stride_h, stride_w = ctx.kernel_params[2], ctx.kernel_params[3]
        padding_h, padding_w = ctx.kernel_params[4], ctx.kernel_params[5]
        dilation_h, dilation_w = ctx.kernel_params[6], ctx.kernel_params[7]
        
        # Use PyTorch's native backward for simplicity
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, 
                stride=(stride_h, stride_w), 
                padding=(padding_h, padding_w), 
                dilation=(dilation_h, dilation_w),
                groups=C
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, 
                stride=(stride_h, stride_w), 
                padding=(padding_h, padding_w), 
                dilation=(dilation_h, dilation_w),
                groups=C
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
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
        
        # Create weight parameter (depthwise convolution: out_channels = in_channels)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        
        # Create bias if requested
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonDepthwiseConv2d.apply(
            x, self.weight, self.bias,
            self.kernel_size_h, self.kernel_size_w,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )

import math