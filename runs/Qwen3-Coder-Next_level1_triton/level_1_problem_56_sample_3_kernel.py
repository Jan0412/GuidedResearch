import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to input, weight, bias, and output
    x_ptr, w_ptr, b_ptr, out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    # Strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_out_c, w_stride_in_c, w_stride_kh, w_stride_kw,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch
    pid_oh = tl.program_id(1)  # output height
    pid_ow = tl.program_id(2)  # output width
    pid_oc = tl.program_id(3)  # output channel block

    # Compute base indices
    out_c = pid_oc * BLOCK_SIZE_OC
    out_h = pid_oh * stride_h - pad_h
    out_w = pid_ow * stride_w - pad_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)

    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel height
        for kh in range(k_h):
            in_h_idx = out_h + kh * dil_h
            mask_h = (in_h_idx >= 0) & (in_h_idx < in_h)
            
            # Loop over kernel width
            for kw in range(k_w):
                in_w_idx = out_w + kw * dil_w
                mask_w = (in_w_idx >= 0) & (in_w_idx < in_w)
                mask = mask_h & mask_w
                
                # Load input value (if within bounds)
                if mask:
                    x_idx = pid_n * x_stride_n + ic * x_stride_c + in_h_idx * x_stride_h + in_w_idx * x_stride_w
                    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                else:
                    x_val = 0.0
                
                # Load weight values for this kernel position and all output channels
                w_idx = (out_c * w_stride_out_c + ic * w_stride_in_c + 
                        kh * w_stride_kh + kw * w_stride_kw)
                w_vals = tl.load(w_ptr + w_idx, mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels - out_c * BLOCK_SIZE_OC, other=0.0)
                
                # Accumulate
                acc += x_val * w_vals

    # Add bias if available
    if b_ptr is not None:
        b_idx = out_c + tl.arange(0, BLOCK_SIZE_OC)
        b_vals = tl.load(b_ptr + b_idx, mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels - out_c * BLOCK_SIZE_OC, other=0.0)
        acc += b_vals

    # Store output
    out_idx = (pid_n * out_stride_n + (out_c + tl.arange(0, BLOCK_SIZE_OC)) * out_stride_c + 
              pid_oh * out_stride_h + pid_ow * out_stride_w)
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels - out_c * BLOCK_SIZE_OC)


def triton_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution.
    Supports only groups=1 for simplicity.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Compute output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_OC = 8  # Output channel block size
    BLOCK_SIZE_C = 1   # Since groups=1, we process all input channels
    
    # Compute strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_out_c, w_stride_in_c, w_stride_kh, w_stride_kw = weight.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()
    
    # Grid dimensions: (batch, output_h, output_w, output_channel_blocks)
    grid = (batch_size, out_h, out_w, (out_channels + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_out_c, w_stride_in_c, w_stride_kh, w_stride_kw,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_IC=1,
        BLOCK_SIZE_KH=1,
        BLOCK_SIZE_KW=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.Conv2d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights (same initialization as PyTorch's Conv2d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using our custom Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )


# Import math for initialization
import math