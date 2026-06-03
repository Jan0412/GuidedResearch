import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor: (batch, in_channels, H, W)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kH, kW,
    stride_h, stride_w,
    padding_h, padding_w,
    output_pad_h, output_pad_w,
    dilation_h, dilation_w,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_i, w_stride_o, w_stride_kh, w_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Create output tensor offsets
    out_offsets = (
        pid_batch * out_stride_b +
        pid_out_c * out_stride_c +
        pid_h * out_stride_h +
        pid_w * out_stride_w
    )
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        # Calculate the input position that contributes to this output position
        # For transposed convolution: output_pos = input_pos * stride + (kernel_pos - 1) * dilation - padding + output_pad
        # So input_pos = (output_pos + padding - (kernel_pos - 1) * dilation - output_pad) / stride
        
        # We need to iterate over all possible kernel positions that could contribute
        for kh in range(kH):
            for kw in range(kW):
                # Calculate the corresponding input position
                h_in = (pid_h - (kh - 1) * dilation_h + padding_h - output_pad_h) // stride_h
                w_in = (pid_w - (kw - 1) * dilation_w + padding_w - output_pad_w) // stride_w
                
                # Check if the input position is valid
                if h_in >= 0 and h_in < height_in and w_in >= 0 and w_in < width_in:
                    # Calculate input offset
                    x_offset = (
                        pid_batch * x_stride_b +
                        c_in * x_stride_c +
                        h_in * x_stride_h +
                        w_in * x_stride_w
                    )
                    
                    # Load input value
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Calculate weight offset
                    w_offset = (
                        c_in * w_stride_i +
                        pid_out_c * w_stride_o +
                        kh * w_stride_kh +
                        kw * w_kw
                    )
                    
                    # Load weight value
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_out_c
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, dilation, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        batch_size, in_channels, height_in, width_in = x.shape
        out_channels, _, kH, kW = weight.shape
        
        # Calculate output dimensions
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        output_pad_h, output_pad_w = output_padding
        dilation_h, dilation_w = dilation
        
        height_out = (height_in - 1) * stride_h - 2 * padding_h + (dilation_h * (kH - 1) + 1) + output_pad_h
        width_out = (width_in - 1) * stride_w - 2 * padding_w + (dilation_w * (kW - 1) + 1) + output_pad_w
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
        
        # Set up kernel parameters
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
        
        # Grid dimensions: (batch, out_channels, height_out, width_out)
        grid = (batch_size, out_channels, height_out, width_out)
        
        # Strides
        x_strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
        w_strides = (weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3))
        out_strides = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels,
            height_in, width_in,
            height_out, width_out,
            kH, kW,
            stride_h, stride_w,
            padding_h, padding_w,
            output_pad_h, output_pad_w,
            dilation_h, dilation_w,
            *x_strides, *w_strides, *out_strides,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch's implementation for backward
        # A full implementation would require additional kernels for backward pass
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, 
                ctx.stride, ctx.padding, 
                ctx.output_padding, ctx.dilation, ctx.groups
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.output_padding, ctx.dilation, ctx.groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def triton_conv_transpose2d(x, weight, bias, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    return TritonConvTranspose2d.apply(
        x, weight, bias, stride, padding, output_padding, dilation, groups
    )


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    Uses custom Triton kernel for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights similar to nn.ConvTranspose2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, *kernel_size) / 
            (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, 
            self.output_padding, self.dilation, 
            self.groups
        )