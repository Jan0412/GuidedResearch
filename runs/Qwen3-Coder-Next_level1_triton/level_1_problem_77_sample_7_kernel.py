import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    # Output dimensions
    D_out, H_out, W_out,
    # Strides
    x_batch_stride, x_c_stride, x_d_stride, x_h_stride, x_w_stride,
    w_in_stride, w_out_stride, w_kd_stride, w_kh_stride, w_kw_stride,
    out_batch_stride, out_c_stride, out_d_stride, out_h_stride, out_w_stride,
    # Block sizes
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_d = tl.program_id(2)  # depth index
    pid_h = tl.program_id(3)  # height index
    pid_w = tl.program_id(4)  # width index

    # Calculate output position
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w

    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate input position from output position
                    # For transposed convolution: input_pos = output_pos * stride - padding + kernel_pos * dilation
                    in_d = out_d * stride_d - padding_d + kd * dilation_d
                    in_h = out_h * stride_h - padding_h + kh * dilation_h
                    in_w = out_w * stride_w - padding_w + kw * dilation_w

                    # Check if input position is within bounds
                    mask_in = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                    if mask_in:
                        # Load input value
                        x_offset = (pid_b * x_batch_stride + c_in * x_c_stride + 
                                   in_d * x_d_stride + in_h * x_h_stride + in_w * x_w_stride)
                        x_val = tl.load(x_ptr + x_offset)

                        # Load weight value
                        w_offset = (c_in * w_in_stride + pid_c_out * w_out_stride + 
                                   kd * w_kd_stride + kh * w_kh_stride + kw * w_kw_stride)
                        w_val = tl.load(w_ptr + w_offset)

                        # Accumulate
                        acc += x_val * w_val

    # Apply bias if available
    if b_ptr is not None:
        bias_offset = pid_c_out
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val

    # Store result
    out_offset = (pid_b * out_batch_stride + pid_c_out * out_c_stride + 
                 out_d * out_d_stride + out_h * out_h_stride + out_w * out_w_stride)
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


class TritonConvTranspose3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        # Extract dimensions
        B, C_in, D, H, W = x.shape
        C_in2, C_out, Kd, Kh, Kw = weight.shape
        assert C_in == C_in2, "Input channels must match"

        # Calculate output dimensions
        D_out = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (Kd - 1) + 1
        H_out = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (Kh - 1) + 1
        W_out = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (Kw - 1) + 1

        # Create output tensor
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

        # Configure kernel launch parameters
        grid = (B, C_out, D_out, H_out, W_out)
        
        # Block sizes (tunable parameters)
        BLOCK_SIZE_C_OUT = min(C_out, 32)
        BLOCK_SIZE_C_IN = min(C_in, 16)
        BLOCK_SIZE_KD = Kd
        BLOCK_SIZE_KH = Kh
        BLOCK_SIZE_KW = Kw

        # Compute strides
        x_batch_stride = x.stride(0)
        x_c_stride = x.stride(1)
        x_d_stride = x.stride(2)
        x_h_stride = x.stride(3)
        x_w_stride = x.stride(4)

        w_in_stride = weight.stride(0)
        w_out_stride = weight.stride(1)
        w_kd_stride = weight.stride(2)
        w_kh_stride = weight.stride(3)
        w_kw_stride = weight.stride(4)

        out_batch_stride = out.stride(0)
        out_c_stride = out.stride(1)
        out_d_stride = out.stride(2)
        out_h_stride = out.stride(3)
        out_w_stride = out.stride(4)

        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias, out,
            B, C_in, D, H, W,
            C_out, Kd, Kh, Kw,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            D_out, H_out, W_out,
            x_batch_stride, x_c_stride, x_d_stride, x_h_stride, x_w_stride,
            w_in_stride, w_out_stride, w_kd_stride, w_kh_stride, w_kw_stride,
            out_batch_stride, out_c_stride, out_d_stride, out_h_stride, out_w_stride,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
            BLOCK_SIZE_KD=BLOCK_SIZE_KD,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        )

        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_size = (B, C_in, D, H, W)
        ctx.output_size = (B, C_out, D_out, H_out, W_out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch implementation for backward
        # In a production implementation, you'd want to implement backward kernels too
        x, weight, bias = ctx.saved_tensors
        
        # Use PyTorch's native backward computation
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.grad_input_conv3d(
                x.shape, weight, grad_output, ctx.stride, ctx.padding, 
                ctx.dilation, groups=1
            )
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.grad_weight_conv3d(
                x, weight.shape, grad_output, ctx.stride, ctx.padding,
                ctx.dilation, groups=1
            )
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])

        return grad_input, grad_weight, grad_bias, None, None, None


class TritonConvTranspose3d(nn.Module):
    """3D transposed convolution using Triton kernel"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return TritonConvTranspose3dFunction.apply(x, self.weight, self.bias, 
                                                   self.stride, self.padding, self.dilation)


import math


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = TritonConvTranspose3d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d(x)