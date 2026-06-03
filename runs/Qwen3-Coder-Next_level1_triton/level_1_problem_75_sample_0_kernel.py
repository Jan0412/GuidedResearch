import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    # Input tensor (batch, in_channels, in_h, in_w)
    x_ptr,
    # Weight tensor (in_channels, out_channels // groups, k_h, k_w)
    w_ptr,
    # Bias tensor (out_channels,)
    b_ptr,
    # Output tensor (batch, out_channels, out_h, out_w)
    out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    # Block sizes for tiling
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_c = pid_out_c * BLOCK_OUT_CH + tl.arange(0, BLOCK_OUT_CH)
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OUT_CH,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for c_in in range(in_channels // groups):
        # Calculate input position for this output
        # For transposed convolution: in_pos = (out_pos - k_pos + pad) // stride
        # But more efficiently, we iterate over kernel positions
        for kh in range(k_h):
            for kw in range(k_w):
                # Calculate corresponding input position
                in_h_pos = out_h - kh * dil_h + pad_h
                in_w_pos = out_w - kw * dil_w + pad_w
                
                # Check if input position is valid
                if in_h_pos >= 0 and in_h_pos < in_h and in_w_pos >= 0 and in_w_pos < in_w:
                    # Calculate input channel index for this group
                    group_idx = c_in // (in_channels // groups)
                    in_c = group_idx * (in_channels // groups) + c_in
                    
                    # Load input value
                    x_offset = pid_b * (in_channels * in_h * in_w) + \
                              in_c * (in_h * in_w) + \
                              in_h_pos * in_w + in_w_pos
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value
                    w_offset = c_in * (out_channels * k_h * k_w) + \
                              pid_out_c * (k_h * k_w) + \
                              kh * k_w + kw
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        b_offset = pid_out_c * BLOCK_OUT_CH + tl.arange(0, BLOCK_OUT_CH)
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = pid_b * (out_channels * out_h * out_w) + \
                pid_out_c * (out_h * out_w) + \
                pid_h * out_w + pid_w
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


def triton_transposed_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton kernel for 2D transposed convolution.
    """
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    out_h = (in_h - 1) * stride_h - 2 * pad_h + dil_h * (k_h - 1) + 1
    out_w = (in_w - 1) * stride_w - 2 * pad_w + dil_w * (k_w - 1) + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["BLOCK_OUT_CH"]),
        out_h,
        out_w
    )
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        BLOCK_BATCH=1,
        BLOCK_OUT_CH=16,
        BLOCK_IN_CH=32,
        BLOCK_KH=3,
        BLOCK_KW=5
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register parameters but use them manually in forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using our custom Triton kernel.
        """
        return triton_transposed_conv2d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )


# Import math for initialization
import math