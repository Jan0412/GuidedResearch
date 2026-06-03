import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or NULL
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    # Strides
    stride_x, stride_cin, stride_h, stride_w,
    stride_w_cin, stride_w_cout, stride_w_kh, stride_w_kw,
    stride_b,  # Bias stride (0 if no bias)
    stride_y, stride_y_cout, stride_y_h, stride_y_w,
    # Convolution parameters
    stride: tl.constexpr, 
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    # Block sizes for tiling
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output coordinates
    out_cout = pid_cout * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    cout_mask = out_cout < C_out
    h_mask = out_h < H_out
    w_mask = out_w < W_out
    
    # Create output tensor pointers
    y_offsets = (pid_b * stride_y + 
                 out_cout[:, None, None] * stride_y_cout + 
                 out_h[None, :, None] * stride_y_h + 
                 out_w[None, None, :] * stride_y_w)
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for cin_idx in range(0, C_in, BLOCK_SIZE_CIN):
        cin_range = cin_idx + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin_range < C_in
        
        # Calculate input coordinates
        # For transposed convolution: H_out = (H_in - 1) * stride - 2 * padding + output_padding + K_h
        # So H_in = (H_out - K_h + 2 * padding - output_padding) / stride + 1
        # The relationship is: in_h = out_h - (K_h - 1) + padding - stride * (something)
        # Actually, for transposed conv: out_h = (in_h - 1) * stride - 2 * padding + output_padding + K_h
        # So in_h = (out_h - (K_h - 1) + 2 * padding - output_padding) // stride
        
        # For each output position, we need to find which input positions contribute
        # out_h = (in_h - 1) * stride + (K_h - 1) - 2 * padding + output_padding
        # => in_h = (out_h - (K_h - 1) + 2 * padding - output_padding) // stride + 1
        
        # Let's compute for each output position, the input position and kernel position
        # For simplicity, iterate over kernel positions
        for kh_idx in range(0, K_h, BLOCK_SIZE_KH):
            kh_range = kh_idx + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh_range < K_h
            
            for kw_idx in range(0, K_w, BLOCK_SIZE_KW):
                kw_range = kw_idx + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw_range < K_w
                
                # Calculate input coordinates for this kernel position
                # in_h = out_h - (K_h - 1 - kh) * stride + padding
                # in_w = out_w - (K_w - 1 - kw) * stride + padding
                
                # We need to handle the batch dimension
                for h_offset in range(BLOCK_SIZE_H):
                    for w_offset in range(BLOCK_SIZE_W):
                        if pid_h * BLOCK_SIZE_H + h_offset >= H_out or pid_w * BLOCK_SIZE_W + w_offset >= W_out:
                            continue
                            
                        out_h_val = pid_h * BLOCK_SIZE_H + h_offset
                        out_w_val = pid_w * BLOCK_SIZE_W + w_offset
                        
                        # Calculate corresponding input coordinates
                        # in_h = (out_h_val - (K_h - 1 - kh) + stride - 1) // stride
                        # But more accurately: out_h = (in_h - 1) * stride + (K_h - 1 - kh) - 2*padding + output_padding
                        # => in_h = (out_h_val - (K_h - 1 - kh) + 2*padding - output_padding) // stride + 1
                        
                        # Actually, let's use the standard transposed conv formula:
                        # out = (in - 1) * stride + (K - 1) - 2 * padding + output_padding + 1
                        # So in_h = ((out_h_val - 1) - (K_h - 1 - kh) + 2 * padding - output_padding) // stride + 1
                        
                        in_h_val = (out_h_val - (K_h - 1 - kh_idx) + 2 * padding - output_padding) // stride + 1
                        in_w_val = (out_w_val - (K_w - 1 - kw_idx) + 2 * padding - output_padding) // stride + 1
                        
                        # Check if this input position is valid
                        if in_h_val < 0 or in_h_val >= H_in or in_w_val < 0 or in_w_val >= W_in:
                            continue
                        
                        # Get input data
                        x_offsets = (pid_b * stride_x + 
                                    cin_range[:, None, None] * stride_cin + 
                                    in_h_val * stride_h + 
                                    in_w_val * stride_w)
                        
                        # Get weight data
                        w_offsets = (cin_range[:, None, None] * stride_w_cin + 
                                    out_cout[None, :, None] * stride_w_cout + 
                                    (K_h - 1 - kh_idx) * stride_w_kh + 
                                    (K_w - 1 - kw_idx) * stride_w_kw)
                        
                        # Load data (with proper masking)
                        x_data = tl.load(x_ptr + x_offsets, mask=cin_mask[:, None, None], other=0.0)
                        w_data = tl.load(w_ptr + w_offsets, mask=cin_mask[:, None, None] & cout_mask[None, :, None], other=0.0)
                        
                        # Accumulate: y[b, cout, out_h, out_w] += x[b, cin, in_h, in_w] * w[cin, cout, kh, kw]
                        # But note: in transposed conv, the relationship is different
                        # Actually: y[b, cout, out_h, out_w] += sum_{cin, kh, kw} x[b, cin, in_h, in_w] * w[cin, cout, kh, kw]
                        # where in_h = out_h - (K_h - 1 - kh) + padding - stride * something
                        
                        # Let me recalculate with correct transposed conv formula:
                        # For transposed conv, the output is computed as:
                        # y[b, cout, out_h, out_w] = b[cout] + sum_{cin} sum_{kh, kw} x[b, cin, in_h, in_w] * w[cin, cout, kh, kw]
                        # where in_h = (out_h - (K_h - 1 - kh) - output_padding + 2*padding) // stride + (K_h - 1 - kh - 2*padding + output_padding) % stride == 0
                        # This is getting complex, let's use a simpler approach with explicit loops
                        
                        # Actually, let's use the standard PyTorch transposed conv formula:
                        # out_h = (in_h - 1) * stride - 2 * padding + output_padding + (K_h - 1) + 1
                        # => in_h = (out_h - (K_h - 1) + 2 * padding - output_padding) // stride + 1
                        
                        # For simplicity, let's iterate over all possible input positions and kernel positions
                        # But that would be inefficient. Let me try a different approach.
                        pass  # Placeholder for now
        
    # Simpler approach: iterate over output dimensions and compute contribution from each input position
    # Reset accumulator
    acc = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # For each output position
    for h_offset in range(BLOCK_SIZE_H):
        for w_offset in range(BLOCK_SIZE_W):
            out_h_val = pid_h * BLOCK_SIZE_H + h_offset
            out_w_val = pid_w * BLOCK_SIZE_W + w_offset
            
            # Check if we're within bounds
            if out_h_val >= H_out or out_w_val >= W_out:
                continue
                
            # For each input channel
            for cin in range(C_in):
                # For each kernel position
                for kh in range(K_h):
                    for kw in range(K_w):
                        # Calculate input position
                        # out_h = (in_h - 1) * stride + (K_h - 1 - kh) - 2 * padding + output_padding + 1
                        # => in_h = (out_h - (K_h - 1 - kh) + 2 * padding - output_padding - 1) // stride + 1
                        
                        # Let's use the correct formula:
                        # The transposed convolution formula is:
                        # out = (in - 1) * stride + (K - 1) - 2 * padding + output_padding + 1
                        # So for a specific kernel position (kh, kw):
                        # out_h = (in_h - 1) * stride + (K_h - 1 - kh) - 2 * padding + output_padding + 1
                        # => in_h = (out_h - (K_h - 1 - kh) + 2 * padding - output_padding - 1) // stride + 1
                        
                        in_h_val = (out_h_val - (K_h - 1 - kh) + 2 * padding - output_padding - 1) // stride + 1
                        in_w_val = (out_w_val - (K_w - 1 - kw) + 2 * padding - output_padding - 1) // stride + 1
                        
                        # Check if input position is valid
                        if in_h_val < 0 or in_h_val >= H_in or in_w_val < 0 or in_w_val >= W_in:
                            continue
                        
                        # Check if the calculation is exact (no remainder)
                        if (out_h_val - (K_h - 1 - kh) + 2 * padding - output_padding - 1) % stride != 0:
                            continue
                        if (out_w_val - (K_w - 1 - kw) + 2 * padding - output_padding - 1) % stride != 0:
                            continue
                        
                        # Load input value
                        x_val = tl.load(x_ptr + pid_b * stride_x + cin * stride_cin + in_h_val * stride_h + in_w_val * stride_w)
                        
                        # Load weight value
                        w_val = tl.load(w_ptr + cin * stride_w_cin + out_cout * stride_w_cout + kh * stride_w_kh + kw * stride_w_kw)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_cout, mask=cout_mask, other=0.0)
        acc += bias[:, None, None]
    
    # Store result
    y_offsets = (pid_b * stride_y + 
                 out_cout[:, None, None] * stride_y_cout + 
                 out_h[None, :, None] * stride_y_h + 
                 out_w[None, None, :] * stride_y_w)
    
    tl.store(y_ptr + y_offsets, acc, mask=cout_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of ConvTranspose2d.
    
    Args:
        x: Input tensor of shape (B, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Bias tensor of shape (C_out,) or None
        stride, padding, output_padding, groups: convolution parameters
    
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out)
    """
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + output_padding + (K_h - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + output_padding + (K_w - 1) + 1
    
    # Create output tensor
    y = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    stride_x = x.stride(0)
    stride_cin = x.stride(1)
    stride_h = x.stride(2)
    stride_w = x.stride(3)
    
    stride_w_cin = weight.stride(0)
    stride_w_cout = weight.stride(1)
    stride_w_kh = weight.stride(2)
    stride_w_kw = weight.stride(3)
    
    stride_b = bias.stride(0) if bias is not None else 0
    
    stride_y = y.stride(0)
    stride_y_cout = y.stride(1)
    stride_y_h = y.stride(2)
    stride_y_w = y.stride(3)
    
    # Grid dimensions
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta["BLOCK_SIZE_COUT"]),
        triton.cdiv(H_out, meta["BLOCK_SIZE_H"]),
        triton.cdiv(W_out, meta["BLOCK_SIZE_W"]),
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride_x, stride_cin, stride_h, stride_w,
        stride_w_cin, stride_w_cout, stride_w_kh, stride_w_kw,
        stride_b,
        stride_y, stride_y_cout, stride_y_h, stride_y_w,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        BLOCK_SIZE_CIN=32,
        BLOCK_SIZE_COUT=32,
        BLOCK_SIZE_KH=3,
        BLOCK_SIZE_KW=3,
        BLOCK_SIZE_H=16,
        BLOCK_SIZE_W=16,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using optimized Triton kernel.
        """
        # Extract parameters from the original layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.conv_transpose2d.stride[0],
            padding=self.conv_transpose2d.padding[0],
            output_padding=self.conv_transpose2d.output_padding[0],
            groups=self.conv_transpose2d.groups
        )