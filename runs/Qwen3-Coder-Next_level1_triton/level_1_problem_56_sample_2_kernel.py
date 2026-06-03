import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    n, c_in, h, w,  # Input dimensions
    c_out, k_h, k_w,  # Weight dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    n_elements,  # Total output elements
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get output tensor dimensions
    out_h = (h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Program IDs for output tensor
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_idx = pid_batch * (c_out * out_h * out_w) + pid_c_out * (out_h * out_w) + pid_h * out_w + pid_w
    if out_idx >= n_elements:
        return
    
    # Compute output position
    output = 0.0
    
    # Compute input position for this output element
    in_h_start = pid_h * stride_h - pad_h
    in_w_start = pid_w * stride_w - pad_w
    
    # Iterate over input channels and kernel positions
    for c in range(c_in):
        for kh in range(k_h):
            in_h = in_h_start + kh * dil_h
            for kw in range(k_w):
                in_w = in_w_start + kw * dil_w
                
                # Check bounds for input
                if 0 <= in_h < h and 0 <= in_w < w:
                    # Calculate input index: n, c, in_h, in_w
                    in_idx = pid_batch * (c_in * h * w) + c * (h * w) + in_h * w + in_w
                    x_val = tl.load(x_ptr + in_idx)
                    
                    # Calculate weight index: c_out, c, kh, kw
                    w_idx = pid_c_out * (c_in * k_h * k_w) + c * (k_h * k_w) + kh * k_w + kw
                    w_val = tl.load(w_ptr + w_idx)
                    
                    output += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        output += bias
    
    # Store result
    tl.store(out_ptr + out_idx, output)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this implementation."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    n, c_in, h, w = x.shape
    c_out, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    out_h = (h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((n, c_out, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Grid configuration: (batch, out_channels, out_height, out_width)
    # We use a simplified grid for now
    BLOCK_SIZE = 64
    BLOCK_C_OUT = 1
    BLOCK_C_IN = 1
    BLOCK_KH = 1
    BLOCK_KW = 1
    
    # For simplicity, we'll use a 4D grid over output dimensions
    # Grid: (batch, c_out, out_h, out_w)
    grid = (n, c_out, out_h, out_w)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        n, c_in, h, w,
        c_out, k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        out.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the convolution layer with same parameters
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use Triton kernel for convolution instead of PyTorch's native implementation.
        """
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups
        )