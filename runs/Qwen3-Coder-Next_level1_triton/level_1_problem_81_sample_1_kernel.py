import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,              # Input tensor pointer (B, C_in, H_in, W_in)
    w_ptr,              # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,              # Bias tensor pointer (C_out,) or None
    y_ptr,              # Output tensor pointer (B, C_out, H_out, W_out)
    B: tl.constexpr,    # Batch size
    C_in: tl.constexpr, # Input channels
    C_out: tl.constexpr, # Output channels
    H_in: tl.constexpr, # Input height
    W_in: tl.constexpr, # Input width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    H_out: tl.constexpr, # Output height
    W_out: tl.constexpr, # Output width
    stride: tl.constexpr, # Stride
    padding: tl.constexpr, # Padding
    dilation: tl.constexpr, # Dilation
    bias: tl.constexpr,   # Whether bias is used
    BLOCK_SIZE_B: tl.constexpr = 1,
    BLOCK_SIZE_C_OUT: tl.constexpr = 16,
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 32,
):
    # Program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Calculate output indices
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W
    
    # Create ranges for output height and width
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for output bounds
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(C_in):
        # Calculate corresponding input positions for each output position
        # For transposed convolution: out_pos = in_pos * stride - padding + dilation * kernel_pos
        # So in_pos = (out_pos + padding - dilation * kernel_pos) / stride
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input position for this kernel position
                h_in = h_offsets - (kh * dilation - padding)
                w_in = w_offsets - (kw * dilation - padding)
                
                # Check if the input position is valid for this kernel position
                h_valid = (h_in % stride == 0)
                w_valid = (w_in % stride == 0)
                
                h_in = h_in // stride
                w_in = w_in // stride
                
                # Check bounds
                h_in_valid = (h_in >= 0) & (h_in < H_in) & h_valid
                w_in_valid = (w_in >= 0) & (w_in < W_in) & w_valid
                valid_mask = h_in_valid & w_in_valid
                
                # Load input values
                x_offset = batch_id * (C_in * H_in * W_in) + c_in * (H_in * W_in)
                x_h_offsets = h_in * W_in + w_in
                x_vals = tl.load(x_ptr + x_offset + x_h_offsets, mask=valid_mask, other=0.0)
                
                # Load weight values
                w_offset = c_in * (C_out * K_h * K_w) + c_out_block * (K_h * K_w) + kh * K_w + kw
                w_vals = tl.load(w_ptr + w_offset)
                
                # Accumulate
                output += x_vals * w_vals
    
    # Apply bias if available
    if bias:
        b_offset = c_out_block
        b_vals = tl.load(b_ptr + b_offset)
        output += b_vals
    
    # Store result
    y_offset = batch_id * (C_out * H_out * W_out) + c_out_block * (H_out * W_out)
    y_h_offsets = h_offsets[:, None] * W_out + w_offsets[None, :]
    tl.store(y_ptr + y_offset + y_h_offsets, output, mask=hw_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        # Get dimensions
        B, C_in, H_in, W_in = x.shape
        C_in_, C_out, K_h, K_w = weight.shape
        assert C_in == C_in_, f"Input channels mismatch: {C_in} vs {C_in_}"
        
        # Calculate output dimensions
        H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1
        W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
        
        # Allocate output tensor
        y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Grid configuration
        grid = lambda meta: (
            B,
            triton.cdiv(C_out, meta["BLOCK_SIZE_C_OUT"]),
            triton.cdiv(H_out, meta["BLOCK_SIZE_H"]),
            triton.cdiv(W_out, meta["BLOCK_SIZE_W"]),
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            B=B, C_in=C_in, C_out=C_out, 
            H_in=H_in, W_in=W_in,
            K_h=K_h, K_w=K_w,
            H_out=H_out, W_out=W_out,
            stride=stride, padding=padding, dilation=dilation,
            bias=bias is not None,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_size = (H_in, W_in)
        ctx.kernel_size = (K_h, K_w)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified version - in practice you'd want proper backward implementation
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # For now, fall back to PyTorch for backward pass
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, stride=stride, 
                padding=padding, dilation=dilation, groups=1
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride=stride,
                padding=padding, dilation=dilation, groups=1
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Initialize weights and bias from the original layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create parameters with the same shape as the original layer
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return TritonConvTranspose2d.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)