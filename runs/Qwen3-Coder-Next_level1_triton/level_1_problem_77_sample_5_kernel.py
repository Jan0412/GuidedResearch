import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,) or NULL
    y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, D, H, W,  # Input dimensions
    C_out, Kd, Kh, Kw,  # Weight dimensions
    D_out, H_out, W_out,  # Output dimensions
    stride_d, stride_h, stride_w,  # Stride parameters
    pad_d, pad_h, pad_w,  # Padding parameters
    dil_d, dil_h, dil_w,  # Dilation parameters
    n_blocks_d: tl.constexpr, n_blocks_h: tl.constexpr, n_blocks_w: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr,
):
    # Program IDs for output spatial dimensions and output channels
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Output channel offset for this block
    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate the corresponding input position (d, h, w) in the output
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Map to input coordinates considering stride and padding
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in in range(C_in):
        # Calculate input position for this channel
        in_d = in_d_start + c_in * 0  # We'll handle kernel iteration separately
        # Actually, we need to iterate over kernel dimensions
        pass
    
    # Better approach: iterate over kernel dimensions and input channel
    for kd in range(Kd):
        for kh in range(Kh):
            for kw in range(Kw):
                # Calculate input position
                in_d = in_d_start + kd * dil_d
                in_h = in_h_start + kh * dil_h
                in_w = in_w_start + kw * dil_w
                
                # Check if input position is valid
                valid_mask = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                
                if tl.sum(valid_mask) > 0:
                    # Calculate input pointer offset for this position
                    input_offset = pid_b * (C_in * D * H * W) + c_in * (D * H * W) + in_d * (H * W) + in_h * W + in_w
                    
                    # Load input value
                    x_val = tl.load(x_ptr + input_offset, mask=valid_mask, other=0.0)
                    
                    # Calculate weight pointer offset
                    weight_offset = c_in * (C_out * Kd * Kh * Kw) + c_out_offsets * (Kd * Kh * Kw) + \
                                   kd * (Kh * Kw) + kh * Kw + kw
                    
                    # Load weight values
                    w_val = tl.load(w_ptr + weight_offset, mask=c_out_mask, other=0.0)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offset = c_out_offsets
        bias_val = tl.load(b_ptr + bias_offset, mask=c_out_mask, other=0.0)
        acc += bias_val
    
    # Convert accumulator to float16/bfloat16 if needed, but keep as float32 for precision
    acc = acc.to(y_ptr.dtype.element_ty)
    
    # Store result
    output_offset = pid_b * (C_out * D_out * H_out * W_out) + c_out_offsets * (D_out * H_out * W_out) + \
                   out_d * (H_out * W_out) + out_h * W_out + out_w
    tl.store(y_ptr + output_offset, acc, mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Custom Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out, Kd, Kh, Kw)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for the transposed convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (B, C_out, D_out, H_out, W_out)
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + dilation * (Kd - 1) + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (Kh - 1) + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (Kw - 1) + 1
    
    # Prepare output tensor
    y = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_SIZE_C_OUT = 16  # Output channel block size
    BLOCK_SIZE_C = 32      # Input channel block size (not used in current implementation but kept for extensibility)
    
    # Grid dimensions
    grid = (B, triton.cdiv(C_out, BLOCK_SIZE_C_OUT), D_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        n_blocks_d=D_out, n_blocks_h=H_out, n_blocks_w=W_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(x, self.weight, self.bias, 
                                       stride=self.stride, 
                                       padding=self.padding, 
                                       dilation=self.dilation)