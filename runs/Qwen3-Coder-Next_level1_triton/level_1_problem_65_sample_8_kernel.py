import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    in_h, in_w,
    out_h, out_w,
    kernel_h, kernel_w,
    stride, padding, output_padding,
    # Strides for memory access
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_OUT_H: tl.constexpr,
    BLOCK_SIZE_OUT_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    pid_out_w = tl.program_id(3)
    
    # Calculate output tensor coordinates
    out_h_start = pid_out_h * BLOCK_SIZE_OUT_H
    out_w_start = pid_out_w * BLOCK_SIZE_OUT_W
    
    # Check bounds for output height and width
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_OUT_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_OUT_W)
    out_h_mask = out_h_offsets < out_h
    out_w_mask = out_w_offsets < out_w
    
    # Create meshgrid for output coordinates
    out_h_grid, out_w_grid = tl.meshgrid(out_h_offsets, out_w_offsets)
    out_h_grid = out_h_grid.T
    out_w_grid = out_w_grid.T
    out_h_mask = out_h_mask & (out_h_grid < out_h)
    out_w_mask = out_w_mask & (out_w_grid < out_w)
    
    # Accumulator for output values
    acc = tl.zeros((BLOCK_SIZE_OUT_H, BLOCK_SIZE_OUT_W), dtype=tl.float32)
    
    # Loop over input channels
    for cin in range(0, in_channels, BLOCK_SIZE_CIN):
        cin_offsets = cin + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin_offsets < in_channels
        
        # Load input values: x[b, cin, :, :]
        # We need to compute which input positions contribute to each output position
        # For transposed convolution: out_h = (in_h - 1) * stride - 2 * padding + output_padding + kernel_h
        # So in_h = (out_h + 2 * padding - output_padding - kernel_h) / stride + 1
        in_h_offsets = (out_h_grid - kernel_h + 1 + padding - output_padding) // stride + 1
        in_w_offsets = (out_w_grid - kernel_w + 1 + padding - output_padding) // stride + 1
        
        # Check if the computed input positions are valid
        valid_h = (in_h_offsets >= 0) & (in_h_offsets < in_h)
        valid_w = (in_w_offsets >= 0) & (in_w_offsets < in_w)
        valid_mask = valid_h & valid_w & out_h_mask[:, None] & out_w_mask[None, :]
        
        # Only compute for valid positions
        if tl.sum(valid_mask) > 0:
            # Calculate actual input indices
            in_h_idx = tl.where(valid_mask, in_h_offsets, 0)
            in_w_idx = tl.where(valid_mask, in_w_offsets, 0)
            
            # Calculate kernel indices: k_h = out_h - (in_h - 1) * stride + padding - output_padding
            k_h_offsets = out_h_grid - (in_h_idx - 1) * stride - padding + output_padding
            k_w_offsets = out_w_grid - (in_w_idx - 1) * stride - padding + output_padding
            
            # Load input values
            x_offsets = (
                pid_b * x_stride_b +
                cin_offsets[:, None, None] * x_stride_c +
                in_h_idx[None, :, :] * x_stride_h +
                in_w_idx[None, :, :] * x_stride_w
            )
            x = tl.load(x_ptr + x_offsets, mask=cin_mask[:, None, None] & valid_mask[None, :, :], other=0.0)
            
            # Load weight values: w[cin, cout, k_h, k_w]
            # Note: for transposed conv, weight layout is [C_in, C_out, K_h, K_w]
            k_h_mask = (k_h_offsets >= 0) & (k_h_offsets < kernel_h)
            k_w_mask = (k_w_offsets >= 0) & (k_w_offsets < kernel_w)
            k_h_valid = tl.where(k_h_mask, k_h_offsets, 0)
            k_w_valid = tl.where(k_w_mask, k_w_offsets, 0)
            
            # For weight loading, we need to handle the broadcasting across cin
            w_offsets = (
                cin_offsets[:, None, None] * w_stride_cin +
                pid_cout * w_stride_cout +
                k_h_valid[None, :, :] * w_stride_kh +
                k_w_valid[None, :, :] * w_kw
            )
            w = tl.load(w_ptr + w_offsets, mask=cin_mask[:, None, None] & k_h_mask[None, :, :] & k_w_mask[None, :, :], other=0.0)
            
            # Compute accumulation: acc += x * w
            # Reshape for broadcasting: x is [C_in, H_out, W_out], w is [C_in, H_out, W_out]
            # We need to sum over C_in
            x_reshaped = x  # [C_in_block, H_out, W_out]
            w_reshaped = w  # [C_in_block, H_out, W_out]
            
            # Accumulate over C_in dimension
            acc += tl.sum(x_reshaped * w_reshaped, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_cout)
        acc += bias
    
    # Store output
    out_offsets = (
        pid_b * out_stride_b +
        pid_cout * out_stride_c +
        out_h_grid * out_stride_h +
        out_w_grid * out_stride_w
    )
    
    # Apply masks
    out_mask = out_h_mask & out_w_mask
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding):
        # Get dimensions
        batch_size, in_channels, in_h, in_w = x.shape
        _, out_channels, kernel_h, kernel_w = weight.shape
        
        # Calculate output dimensions
        out_h = (in_h - 1) * stride - 2 * padding + output_padding + kernel_h
        out_w = (in_w - 1) * stride - 2 * padding + output_padding + kernel_w
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        
        # Set up kernel parameters
        BLOCK_SIZE_COUT = 16
        BLOCK_SIZE_CIN = 16
        BLOCK_SIZE_KH = 3  # kernel_h is 3 for our case
        BLOCK_SIZE_KW = 7  # kernel_w is 7 for our case
        BLOCK_SIZE_OUT_H = 8
        BLOCK_SIZE_OUT_W = 8
        
        # Grid dimensions
        grid = (
            batch_size,  # B
            triton.cdiv(out_channels, BLOCK_SIZE_COUT),  # C_out blocks
            triton.cdiv(out_h, BLOCK_SIZE_OUT_H),        # H_out blocks
            triton.cdiv(out_w, BLOCK_SIZE_OUT_W),        # W_out blocks
        )
        
        # Strides
        x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
        w_stride_cin, w_stride_cout, w_stride_kh, w_kw = weight.stride()
        out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            kernel_h, kernel_w,
            stride, padding, output_padding,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            w_stride_cin, w_stride_cout, w_stride_kh, w_kw,
            out_stride_b, out_stride_c, out_stride_h, out_stride_w,
            BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
            BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
            BLOCK_SIZE_OUT_H=BLOCK_SIZE_OUT_H,
            BLOCK_SIZE_OUT_W=BLOCK_SIZE_OUT_W,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - for full functionality
        # we would need proper backward kernels, but for inference-only
        # or simple use cases, we can fall back to PyTorch for backward
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        
        # Use PyTorch's built-in transposed convolution backward
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv_transpose2d(
                grad_output, weight, None,
                stride=stride, padding=padding,
                output_padding=output_padding,
                groups=1
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output,
                stride=stride, padding=padding,
                groups=1
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    """Wrapper function for the custom Triton transposed convolution"""
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding)


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton custom kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, 
                                      self.stride, self.padding, 
                                      self.output_padding)