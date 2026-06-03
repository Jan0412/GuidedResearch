import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,              # Input tensor pointer (N, C_in, H, W)
    w_ptr,              # Weight tensor pointer (C_out, C_in // groups, kH, kW)
    b_ptr,              # Bias tensor pointer (C_out,) - can be None
    out_ptr,            # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, H, W,      # Input dimensions
    C_out,              # Output channels
    kH, kW,             # Kernel height and width
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    groups,             # Number of groups
    n_elements,         # Total output elements
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    batch_id = pid // (C_out * ((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                                (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w))
    out_elem = pid % (C_out * ((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                               (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w))
    
    c_out = out_elem // (((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                          (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w))
    spatial_idx = out_elem % (((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                               (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w))
    
    h_out = spatial_idx // ((W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w)
    w_out = spatial_idx % ((W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w)
    
    # Compute output position
    h_out_start = h_out * stride_h - pad_h
    w_out_start = w_out * stride_w - pad_w
    
    # Accumulator for convolution
    acc = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels (with grouping support)
    for c_in_group_start in range(0, C_in // groups, BLOCK_C_IN):
        for kh in range(kH):
            h_in = h_out_start + kh * dil_h
            # Check if h_in is within valid bounds
            if h_in >= 0 and h_in < H:
                for kw in range(kW):
                    w_in = w_out_start + kw * dil_w
                    # Check if w_in is within valid bounds
                    if w_in >= 0 and w_in < W:
                        # Compute input pointer offset
                        input_offset = (batch_id * C_in * H * W + 
                                       (c_in_group_start + tl.arange(0, BLOCK_C_IN) % (C_in // groups)) * H * W +
                                       h_in * W + w_in)
                        
                        # Load input values
                        x_offsets = input_offset
                        x_vals = tl.load(x_ptr + x_offsets, 
                                        mask=(c_in_group_start + tl.arange(0, BLOCK_C_IN)) < (C_in // groups), 
                                        other=0.0)
                        
                        # Compute weight pointer offset
                        weight_offset = (c_out * (C_in // groups) * kH * kW + 
                                        tl.arange(0, BLOCK_C_IN) * kH * kW +
                                        kh * kW + kw)
                        
                        # Load weights
                        w_offsets = weight_offset
                        w_vals = tl.load(w_ptr + w_offsets, 
                                        mask=(tl.arange(0, BLOCK_C_IN)) < (C_in // groups), 
                                        other=0.0)
                        
                        # Accumulate multiplication
                        acc += tl.sum(x_vals * w_vals, axis=0, keep_dims=True)
    
    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c_out)
        acc += bias_val
    
    # Store result
    out_offset = (batch_id * C_out * ((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                                      (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w) +
                 c_out * ((H + 2*pad_h - dil_h*(kH-1) + stride_h - 1) // stride_h * 
                          (W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w) +
                 h_out * ((W + 2*pad_w - dil_w*(kW-1) + stride_w - 1) // stride_w) +
                 w_out)
    
    tl.store(out_ptr + out_offset, acc[0])


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in // groups, kH, kW)
        bias: Bias tensor of shape (C_out,) or None
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation
        
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) + stride_h - 1) // stride_h
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) + stride_w - 1) // stride_w
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    n_elements = N * C_out * H_out * W_out
    
    # Configure block sizes (tunable parameters for performance)
    BLOCK_SIZE = 128
    BLOCK_C_OUT = 1
    BLOCK_C_IN = min(32, C_in // groups)
    BLOCK_H = 1
    BLOCK_W = 1
    
    # Grid configuration
    grid = (n_elements,)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    Optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as original but will use our custom convolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 
                                               self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=self.dilation, 
                            groups=self.groups)