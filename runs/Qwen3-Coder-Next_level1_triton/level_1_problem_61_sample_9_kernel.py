import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor (B, C_in, D, H, W)
    w_ptr,  # Weight tensor (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,  # Input dimensions
    C_out,  # Output channels
    Kd, Kh, Kw,  # Kernel dimensions
    D_out, H_out, W_out,  # Output dimensions
    # Stride parameters
    stride_d, stride_h, stride_w,  # Stride
    # Padding parameters
    pad_d, pad_h, pad_w,  # Padding
    # Output padding
    output_pad_d, output_pad_h, output_pad_w,  # Output padding
    # Block sizes for tiling
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_id = tl.program_id(1)
    
    # Compute output position
    out_idx = tl.program_id(2)
    d_out = out_idx // (H_out * W_out)
    h_out = (out_idx % (H_out * W_out)) // W_out
    w_out = out_idx % W_out
    
    # Initialize accumulator for the output
    acc = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Compute corresponding input position
                    d_in = d_out - kd + pad_d
                    h_in = h_out - kh + pad_h
                    w_in = w_out - kw + pad_w
                    
                    # Check if input position is valid
                    if (d_in >= 0 and d_in < D and 
                        h_in >= 0 and h_in < H and 
                        w_in >= 0 and w_in < W):
                        # Compute input index
                        x_idx = batch_id * C_in * D * H * W + \
                                c_in * D * H * W + \
                                d_in * H * W + \
                                h_in * W + \
                                w_in
                        
                        # Compute weight index
                        w_idx = c_in * C_out * Kd * Kh * Kw + \
                                c_out_id * Kd * Kh * Kw + \
                                kd * Kh * Kw + \
                                kh * Kw + \
                                kw
                        
                        # Load values and accumulate
                        x_val = tl.load(x_ptr + x_idx)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out_id)
        acc += b_val
    
    # Store result
    out_idx_final = batch_id * C_out * D_out * H_out * W_out + \
                    c_out_id * D_out * H_out * W_out + \
                    d_out * H_out * W_out + \
                    h_out * W_out + \
                    w_out
    
    tl.store(out_ptr + out_idx_final, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose3d for groups=1.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in_weight, C_out, Kd, Kh, Kw = weight.shape
    
    # Validate groups
    assert groups == 1, "Only groups=1 is supported in this implementation"
    assert C_in == C_in_weight, f"Input channels mismatch: {C_in} != {C_in_weight}"
    
    # Compute output dimensions
    D_out = (D - 1) * stride - 2 * padding + Kd + output_padding
    H_out = (H - 1) * stride - 2 * padding + Kh + output_padding
    W_out = (W - 1) * stride - 2 * padding + Kw + output_padding
    
    # Create output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure grid
    # Grid: (batch_size, out_channels, D_out * H_out * W_out)
    grid = (B, C_out, D_out * H_out * W_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out,
        Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        BLOCK_B=1,
        BLOCK_C_OUT=1,
        BLOCK_C_IN=1,
        BLOCK_KD=1,
        BLOCK_KH=1,
        BLOCK_KW=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.
    Uses custom Triton kernel for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the original layer
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        output_padding = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups
        bias = self.conv_transpose3d.bias
        
        # Convert tuple parameters to integers if needed
        if isinstance(stride, tuple):
            stride = stride[0] if len(stride) > 0 else 1
        if isinstance(padding, tuple):
            padding = padding[0] if len(padding) > 0 else 0
        if isinstance(output_padding, tuple):
            output_padding = output_padding[0] if len(output_padding) > 0 else 0
            
        # Call the Triton implementation
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups
        )