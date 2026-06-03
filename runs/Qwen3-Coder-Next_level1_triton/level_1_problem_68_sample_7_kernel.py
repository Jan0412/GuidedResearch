import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to input/output tensors
    x_ptr,          # Input tensor: (B, C_in, D, H, W)
    w_ptr,          # Weight tensor: (C_in, C_out, kD, kH, kW)
    b_ptr,          # Bias tensor: (C_out,) - can be None
    y_ptr,          # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out,
    D, H, W,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Output dimensions
    D_out, H_out, W_out,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_kD: tl.constexpr,
    BLOCK_kH: tl.constexpr,
    BLOCK_kW: tl.constexpr,
):
    # Get output tensor indices
    batch_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)
    
    # Calculate input indices for transposed convolution
    # For transposed conv: input[i] contributes to output[i*stride + ...]
    d_in_start = d_out - (kD - 1) + pad_d
    h_in_start = h_out - (kH - 1) + pad_h
    w_in_start = w_out - (kW - 1) + pad_w
    
    # Check if this output position is valid
    d_in = d_in_start // stride_d
    h_in = h_in_start // stride_h
    w_in = w_in_start // stride_w
    
    if (d_in < 0 or d_in >= D or h_in < 0 or h_in >= H or w_in < 0 or w_in >= W):
        return
    
    # Check stride alignment
    if (d_in_start % stride_d != 0 or h_in_start % stride_h != 0 or w_in_start % stride_w != 0):
        return
    
    # Compute the kernel indices
    k_d = (kD - 1) - (d_out - d_in * stride_d)
    k_h = (kH - 1) - (h_out - h_in * stride_h)
    k_w = (kW - 1) - (w_out - w_in * stride_w)
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_offset in range(0, C_in, BLOCK_C_in):
        c_in_idx = c_in_offset + tl.arange(0, BLOCK_C_in)
        mask_c_in = c_in_idx < C_in
        
        # Load input value
        x_offset = ((batch_idx * C_in * D * H * W + 
                    c_in_idx * D * H * W + 
                    d_in * H * W + 
                    h_in * W + 
                    w_in) * BLOCK_C_in // BLOCK_C_in)  # Simplified indexing
        
        # Actually compute the correct offset for x
        x_ptr_offset = (batch_idx * C_in * D * H * W + 
                       c_in_idx * D * H * W + 
                       d_in * H * W + 
                       h_in * W + 
                       w_in)
        x_val = tl.load(x_ptr + x_ptr_offset, mask=mask_c_in, other=0.0)
        
        # Load weights
        w_ptr_offset = (c_in_idx * C_out * kD * kH * kW + 
                       c_out_idx * kD * kH * kW + 
                       k_d * kH * kW + 
                       k_h * kW + 
                       k_w)
        w_val = tl.load(w_ptr + w_ptr_offset, mask=mask_c_in, other=0.0)
        
        # Accumulate
        acc += x_val * w_val
    
    # Apply bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_idx)
        acc += bias
    
    # Store result
    y_offset = (batch_idx * C_out * D_out * H_out * W_out + 
               c_out_idx * D_out * H_out * W_out + 
               d_out * H_out * W_out + 
               h_out * W_out + 
               w_out)
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    output_padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
):
    """
    Triton implementation of 3D transposed convolution.
    
    Note: This is a simplified implementation that assumes:
    - groups == 1 (no grouped convolutions)
    - dilation == 1 (no dilated convolutions)
    """
    assert groups == 1, "Only groups=1 is supported in this Triton implementation"
    assert dilation == (1, 1, 1) or dilation == 1, "Only dilation=1 is supported"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_out, _, kD, kH, kW = weight.shape
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    output_pad_d, output_pad_h, output_pad_w = output_padding
    
    # Calculate output dimensions
    D_out = (D - 1) * stride_d - 2 * pad_d + (kD - 1) * dilation[0] + output_pad_d + 1
    H_out = (H - 1) * stride_h - 2 * pad_h + (kH - 1) * dilation[1] + output_pad_h + 1
    W_out = (W - 1) * stride_w - 2 * pad_w + (kW - 1) * dilation[2] + output_pad_w + 1
    
    # Create output tensor
    y = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Kernel configuration
    # We'll use a simple grid configuration where each output element is computed by one thread
    # This is not the most efficient but works for the general case
    
    # For better performance, we can use block-based computation
    BLOCK_C_out = 1
    BLOCK_C_in = min(32, C_in)
    BLOCK_kD = kD
    BLOCK_kH = kH
    BLOCK_kW = kW
    
    # Grid dimensions: (batch, C_out, D_out, H_out, W_out)
    grid = (B, C_out, D_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out,
        D, H, W,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w,
        D_out, H_out, W_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_kD=BLOCK_kD,
        BLOCK_kH=BLOCK_kH,
        BLOCK_kW=BLOCK_kW,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=(1, 1, 1),
            groups=self.groups
        )