import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    X_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    W_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    B_ptr,  # Bias tensor: (C_out,) or None
    Y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    H_in: tl.constexpr,  # Input height
    W_in: tl.constexpr,  # Input width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    stride: tl.constexpr,  # Stride
    padding: tl.constexpr,  # Padding
    output_padding: tl.constexpr,  # Output padding
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    BLOCK_SIZE_B: tl.constexpr = 1,
    BLOCK_SIZE_C_OUT: tl.constexpr = 16,
    BLOCK_SIZE_H: tl.constexpr = 32,
    BLOCK_SIZE_W: tl.constexpr = 32,
    BLOCK_SIZE_KH: tl.constexpr = 3,
    BLOCK_SIZE_KW: tl.constexpr = 3,
):
    # Program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Compute output coordinates
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W
    
    # Create ranges for output dimensions
    h_offsets = tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    
    # Output coordinates with bounds checking
    h_out = h_start + h_offsets
    w_out = w_start + w_offsets
    
    h_out_mask = h_out < H_out
    w_out_mask = w_out < W_out
    hw_mask = h_out_mask[:, None] & w_out_mask[None, :]
    
    # Accumulate over input channels and kernel dimensions
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_id in range(C_in):
        # Compute corresponding input coordinates
        h_in = (h_out - padding) // stride
        w_in = (w_out - padding) // stride
        
        # Check if input coordinates are valid
        h_in_valid = (h_in >= 0) & (h_in < H_in)
        w_in_valid = (w_in >= 0) & (w_in < W_in)
        valid_mask = h_in_valid & w_in_valid
        
        # Compute input indices
        x_offset = batch_id * (C_in * H_in * W_in) + \
                   c_in_id * (H_in * W_in) + \
                   h_in[:, None] * W_in + \
                   w_in[None, :]
        
        # Load input values
        x_val = tl.load(X_ptr + x_offset, mask=valid_mask[:, :, None, None].squeeze(-1).squeeze(-1), other=0.0)
        
        # Compute kernel coordinates
        kh = (h_out - h_in * stride - padding)
        kw = (w_out - w_in * stride - padding)
        
        # Kernel offsets
        kh_offsets = tl.arange(0, BLOCK_SIZE_KH)
        kw_offsets = tl.arange(0, BLOCK_SIZE_KW)
        
        # Kernel indices for current output position
        w_offset = c_in_id * (C_out * K_h * K_w) + \
                   c_out_block * (K_h * K_w) + \
                   kh[:, None, None] * K_w + \
                   kw[None, :, None]
        
        # Load weights
        w_val = tl.load(W_ptr + w_offset, mask=hw_mask[:, :, None], other=0.0)
        
        # Accumulate
        acc += tl.sum(x_val[:, :, None] * w_val, axis=1)
    
    # Apply bias if present
    if B_ptr is not None:
        bias = tl.load(B_ptr + c_out_block * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT))
        acc += bias[None, :]
    
    # Store result
    y_offset = batch_id * (C_out * H_out * W_out) + \
               c_out_block * (H_out * W_out) + \
               h_out[:, None] * W_out + \
               w_out[None, :]
    
    tl.store(Y_ptr + y_offset, acc.to(tl.float32), mask=hw_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding):
        # Get dimensions
        B, C_in, H_in, W_in = x.shape
        C_in2, C_out, K_h, K_w = weight.shape
        
        # Compute output dimensions
        H_out = (H_in - 1) * stride - 2 * padding + output_padding + (K_h - 1) + 1 + padding
        W_out = (W_in - 1) * stride - 2 * padding + output_padding + (K_w - 1) + 1 + padding
        
        # Prepare output tensor
        y = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
        
        # Configure grid
        grid = lambda meta: (
            B,
            triton.cdiv(C_out, meta["BLOCK_SIZE_C_OUT"]),
            triton.cdiv(H_out, meta["BLOCK_SIZE_H"]),
            triton.cdiv(W_out, meta["BLOCK_SIZE_W"]),
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            B, C_in, C_out, H_in, W_in,
            K_h, K_w, stride, padding, output_padding,
            H_out, W_out,
            BLOCK_SIZE_B=1,
            BLOCK_SIZE_C_OUT=16,
            BLOCK_SIZE_H=32,
            BLOCK_SIZE_W=32,
            BLOCK_SIZE_KH=3,
            BLOCK_SIZE_KW=3,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.input_size = (B, C_in, H_in, W_in)
        ctx.output_size = (B, C_out, H_out, W_out)
        ctx.kernel_size = (K_h, K_w)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - for production use, you'd want
        # proper backward pass implementation with gradient computation
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        
        # Compute gradients using PyTorch for simplicity in this context
        # In a full implementation, you'd have custom backward kernels too
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Use PyTorch's native implementation for gradient computation
            grad_input = torch.nn.grad.conv2d_input(
                ctx.input_size, weight, grad_output, stride=stride,
                padding=padding, output_padding=output_padding, groups=1
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride=stride,
                padding=padding, output_padding=output_padding, groups=1
            )
        
        if ctx.needs_input_grad[2] and ctx.needs_input_grad[2] is not None:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel using Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights similar to nn.ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return TritonConvTranspose2d.apply(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)