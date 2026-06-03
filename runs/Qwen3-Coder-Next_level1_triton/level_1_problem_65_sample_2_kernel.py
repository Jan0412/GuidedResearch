import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for transposed 2D convolution
@triton.jit
def conv_transpose2d_kernel(
    x_ptr,              # Input tensor pointer (N, C_in, H_in, W_in)
    w_ptr,              # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,              # Bias tensor pointer (C_out,)
    out_ptr,            # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, C_out,     # Batch size, input channels, output channels
    H_in, W_in,         # Input height and width
    H_out, W_out,       # Output height and width
    K_h, K_w,           # Kernel height and width
    stride,             # Stride
    padding,            # Padding
    output_padding,     # Output padding
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
):
    # Calculate output position
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_start = tl.program_id(2) * BLOCK_H
    out_w_start = tl.program_id(3) * BLOCK_W
    
    # Create ranges for output height and width
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_W)
    
    # Create masks for output height and width
    out_h_mask = out_h_offsets < H_out
    out_w_mask = out_w_offsets < W_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for in_c_idx in range(0, C_in, BLOCK_C_in):
        # Create input channel offsets
        in_c_offsets = in_c_idx + tl.arange(0, BLOCK_C_in)
        in_c_mask = in_c_offsets < C_in
        
        # Process kernel positions
        for kh in range(K_h):
            # Calculate corresponding input height
            in_h = (out_h_offsets - kh + padding) // stride
            
            # Check if input height is valid
            h_valid = (in_h >= 0) & (in_h < H_in) & ((out_h_offsets - kh - padding) % stride == 0)
            h_valid = h_valid & out_h_mask[:, None]
            
            # Calculate input height indices
            in_h_indices = in_h * W_in
            
            for kw in range(K_w):
                # Calculate corresponding input width
                in_w = (out_w_offsets - kw + padding) // stride
                
                # Check if input width is valid
                w_valid = (in_w >= 0) & (in_w < W_in) & ((out_w_offsets - kw - padding) % stride == 0)
                w_valid = w_valid & out_w_mask[None, :]
                
                # Combine masks
                valid_mask = h_valid & w_valid
                
                # Calculate input index
                in_w_indices = in_w
                
                # Load input values
                in_indices = batch_idx * (C_in * H_in * W_in) + in_c_offsets[None, None, :] * (H_in * W_in) + \
                            in_h_indices[:, :, None] * W_in + in_w_indices[:, :, None]
                
                # Reshape for proper broadcasting
                in_indices_flat = tl.reshape(in_indices, (BLOCK_H * BLOCK_W, BLOCK_C_in))
                valid_mask_flat = tl.reshape(valid_mask, (BLOCK_H * BLOCK_W, 1))
                
                # Load input with proper padding
                x_vals = tl.load(x_ptr + in_indices_flat, mask=valid_mask_flat, other=0.0)
                x_vals = tl.reshape(x_vals, (BLOCK_H, BLOCK_W, BLOCK_C_in))
                
                # Load weight values
                w_indices = in_c_offsets[None, None, :] * (C_out * K_h * K_w) + \
                           out_c_idx * (K_h * K_w) + (K_h - 1 - kh) * K_w + (K_w - 1 - kw)
                w_vals = tl.load(w_ptr + w_indices, mask=in_c_mask[None, None, :], other=0.0)
                
                # Accumulate product
                acc += tl.sum(x_vals * w_vals[:, :, :], axis=2)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    out_indices = batch_idx * (C_out * H_out * W_out) + out_c_idx * (H_out * W_out) + \
                 out_h_offsets[:, None] * W_out + out_w_offsets[None, :]
    out_mask = (out_h_offsets[:, None] < H_out) & (out_w_offsets[None, :] < W_out)
    
    tl.store(out_ptr + out_indices, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs transposed 2D convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + (K_h - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (K_w - 1) + output_padding + 1
    
    # Prepare output tensor
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tuning
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C_in = 16
    BLOCK_C_out = 16
    
    # Grid dimensions
    grid = (N, C_out, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride, padding, output_padding,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_C_out=BLOCK_C_out,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.has_bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        
        # Initialize bias if needed
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
        # Check if bias is available
        bias = self.bias if self.has_bias else None
        
        # Call the Triton implementation
        return triton_conv_transpose2d(
            x, self.weight, bias,
            self.stride, self.padding, self.output_padding, self.groups
        )