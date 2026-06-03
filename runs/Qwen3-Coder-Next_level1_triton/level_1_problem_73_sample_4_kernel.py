import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # Input: (B, C_in, D, H, W)
    W_ptr,  # Weight: (C_in, C_out // groups, kD, kH, kW)
    B_ptr,  # Bias: (C_out,) optional
    Y_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    groups,
    # Block sizes
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    # Flags
    HAS_BIAS: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate which output channel block we're processing
    c_out_start = pid_c_out * BLOCK_C_OUT
    c_out_range = tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_range < (out_channels - c_out_start)
    
    # Calculate input position corresponding to this output position
    # For transposed convolution: d_out = stride * d_in + (kD - 1 - d_offset)
    # So d_in = (d_out - (kD - 1 - d_offset)) / stride
    d_offset = pid_d % stride_d
    d_in = (pid_d - d_offset) // stride_d
    h_offset = pid_h % stride_h
    h_in = (pid_h - h_offset) // stride_h
    w_offset = pid_w % stride_w
    w_in = (pid_w - w_offset) // stride_w
    
    # Check if this is a valid input position
    valid_in = (d_in >= 0) & (d_in < in_depth) & (h_in >= 0) & (h_in < in_height) & (w_in >= 0) & (w_in < in_width)
    
    # Calculate the offset for the input tensor
    # Input shape: (B, C_in, D, H, W)
    input_offset = pid_batch * (in_channels * in_depth * in_height * in_width)
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, in_channels, BLOCK_C_IN):
        c_in_range = c_in + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_range < in_channels
        
        # Calculate input pointer offset
        in_ptr = X_ptr + input_offset + c_in_range[:, None] * (in_depth * in_height * in_width)
        
        # Load input values if valid
        if valid_in:
            x_val = tl.load(in_ptr + d_in * (in_height * in_width) + h_in * in_width + w_in, mask=c_in_mask[:, None], other=0.0)
        else:
            x_val = tl.zeros((BLOCK_C_IN, 1), dtype=tl.float32)
        
        # Iterate over kernel dimensions
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate output channel offset for this kernel position
                    # For group convolution: weight shape is (C_in, C_out // groups, kD, kH, kW)
                    # Weight index: [c_in, c_out_group, kd, kh, kw]
                    # c_out_group = (c_out_start + c_out_idx) // (out_channels // groups)
                    
                    # Get weight pointer offset for this kernel position
                    weight_offset = c_in * (out_channels * kD * kH * kW) + \
                                   kd * (out_channels * kH * kW) + \
                                   kh * (out_channels * kW) + \
                                   kw * out_channels + c_out_start
                    
                    w_ptr_pos = W_ptr + weight_offset
                    
                    # Load weights for current kernel position and output channels
                    w_val = tl.load(w_ptr_pos + c_out_range, mask=c_out_mask, other=0.0)
                    
                    # Accumulate: x[c_in] * w[c_in, c_out, kd, kh, kw]
                    # But only if the kernel position matches the stride offset
                    if kd == d_offset and kh == h_offset and kw == w_offset:
                        acc += tl.sum(x_val * w_val[None, :], axis=0)
    
    # Apply bias if present
    if HAS_BIAS:
        bias_ptr = B_ptr + c_out_start + c_out_range
        bias_val = tl.load(bias_ptr, mask=c_out_mask, other=0.0)
        acc += bias_val
    
    # Store result
    output_offset = pid_batch * (out_channels * out_depth * out_height * out_width) + \
                   c_out_start * (out_depth * out_height * out_width) + \
                   pid_d * (out_height * out_width) + \
                   pid_h * out_width + pid_w
    
    # Store output values
    tl.store(Y_ptr + output_offset, acc.to(tl.float32), mask=c_out_mask)


class TritonConvTranspose3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Store parameters for potential backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        # Calculate output dimensions
        # For transposed convolution: 
        # out_depth = (in_depth - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
        # Simplified: out_depth = (in_depth - 1) * stride + kernel_size + output_padding
        # But with padding applied to input first, the standard formula is:
        # out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
        
        B, C_in, D_in, H_in, W_in = x.shape
        C_in2, C_out_per_group, kD, kH, kW = weight.shape
        C_out = C_out_per_group * groups
        
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        out_pad_d, out_pad_h, out_pad_w = output_padding
        
        # Calculate output dimensions
        D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d
        H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w
        
        # Create output tensor
        Y = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
        
        # Grid dimensions
        grid = (B, triton.cdiv(C_out, 32), triton.cdiv(D_out, 4), triton.cdiv(H_out, 4), triton.cdiv(W_out, 4))
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias, Y,
            B, C_in, C_out,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            kD, kH, kW,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            groups,
            BLOCK_C_OUT=32,
            BLOCK_C_IN=16,
            BLOCK_D=4,
            BLOCK_H=4,
            BLOCK_W=4,
            HAS_BIAS=bias is not None
        )
        
        return Y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Simple implementation using PyTorch for backward
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv3d_input(x.shape, weight, grad_output, 
                                                   ctx.stride, ctx.padding, 
                                                   ctx.output_padding, groups=ctx.groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv3d_weight(x, weight.shape, grad_output, 
                                                     ctx.stride, ctx.padding, 
                                                     ctx.output_padding, groups=ctx.groups)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


class TritonConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, 
                                              self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return TritonConvTranspose3dFunction.apply(x, self.weight, self.bias, 
                                                  self.stride, self.padding, 
                                                  self.output_padding, self.groups)


import math

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = TritonConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), 
                                                     stride=stride, padding=padding, 
                                                     output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)