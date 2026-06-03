import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, D, H, W)
    w_ptr,  # Weight tensor pointer (C_in, C_out // groups, Kd, Kh, Kw)
    bias_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    D: tl.constexpr,  # Input depth
    H: tl.constexpr,  # Input height
    W: tl.constexpr,  # Input width
    D_out: tl.constexpr,  # Output depth
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    Kd: tl.constexpr,  # Kernel depth
    Kh: tl.constexpr,  # Kernel height
    Kw: tl.constexpr,  # Kernel width
    stride_d: tl.constexpr,  # Stride depth
    stride_h: tl.constexpr,  # Stride height
    stride_w: tl.constexpr,  # Stride width
    pad_d: tl.constexpr,  # Padding depth
    pad_h: tl.constexpr,  # Padding height
    pad_w: tl.constexpr,  # Padding width
    output_pad_d: tl.constexpr,  # Output padding depth
    output_pad_h: tl.constexpr,  # Output padding height
    output_pad_w: tl.constexpr,  # Output padding width
    groups: tl.constexpr,  # Number of groups
    BLOCK_B: tl.constexpr = 1,
    BLOCK_C_OUT: tl.constexpr = 32,
    BLOCK_D: tl.constexpr = 8,
    BLOCK_H: tl.constexpr = 8,
    BLOCK_W: tl.constexpr = 8,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for output dimensions
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    
    # Output indices (flattened)
    out_indices = ((pid_b * C_out * D_out * H_out * W_out + 
                   pid_c_out * D_out * H_out * W_out + 
                   out_d[:, None, None] * H_out * W_out + 
                   out_h[None, :, None] * W_out + 
                   out_w[None, None, :]) * mask_d[:, None, None] * mask_h[None, :, None] * mask_w[None, None, :])
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Calculate which group this output channel belongs to
    group_id = pid_c_out // (C_out // groups)
    
    # Loop over input channels in this group
    c_in_start = group_id * (C_in // groups)
    c_in_end = (group_id + 1) * (C_in // groups)
    
    for c_in in range(c_in_start, c_in_end):
        # Loop over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate corresponding input position
                    in_d = (out_d[:, None, None] - kd + pad_d) // stride_d
                    in_h = (out_h[None, :, None] - kh + pad_h) // stride_h
                    in_w = (out_w[None, None, :] - kw + pad_w) // stride_w
                    
                    # Check if input position is valid
                    mask_in_d = (in_d >= 0) & (in_d < D)
                    mask_in_h = (in_h >= 0) & (in_h < H)
                    mask_in_w = (in_w >= 0) & (in_w < W)
                    
                    # Input indices
                    in_indices = ((pid_b * C_in * D * H * W + 
                                  c_in * D * H * W + 
                                  in_d * H * W + 
                                  in_h * W + 
                                  in_w) * mask_in_d * mask_in_h * mask_in_w)
                    
                    # Load input values
                    x_val = tl.load(x_ptr + in_indices, mask=(mask_in_d & mask_in_h & mask_in_w), other=0.0)
                    
                    # Load weight value
                    w_idx = ((c_in * C_out * Kd * Kh * Kw + 
                             pid_c_out * Kd * Kh * Kw + 
                             kd * Kh * Kw + 
                             kh * Kw + 
                             kw) * mask_d[:, None, None] * mask_h[None, :, None] * mask_w[None, None, :])
                    w_val = tl.load(w_ptr + w_idx, mask=(mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]), other=0.0)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_c_out)
        acc += bias_val
    
    # Store output
    tl.store(out_ptr + out_indices, acc.to(x_ptr.dtype.element_ty), mask=(mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]))


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs 3D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out // groups, Kd, Kh, Kw)
        bias: Bias tensor of shape (C_out,) or None
        stride: Tuple (stride_d, stride_h, stride_w)
        padding: Tuple (pad_d, pad_h, pad_w)
        output_padding: Tuple (output_pad_d, output_pad_h, output_pad_w)
        groups: Number of groups
    
    Returns:
        Output tensor of shape (B, C_out, D_out, H_out, W_out)
    """
    B, C_in, D, H, W = x.shape
    _, C_out, Kd, Kh, Kw = weight.shape
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    output_pad_d, output_pad_h, output_pad_w = output_padding
    
    # Calculate output dimensions
    D_out = (D - 1) * stride_d - 2 * pad_d + Kd + output_pad_d
    H_out = (H - 1) * stride_h - 2 * pad_h + Kh + output_pad_h
    W_out = (W - 1) * stride_w - 2 * pad_w + Kw + output_pad_w
    
    # Prepare output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta["BLOCK_C_OUT"]),
        triton.cdiv(D_out, meta["BLOCK_D"]),
        triton.cdiv(H_out, meta["BLOCK_H"]),
        triton.cdiv(W_out, meta["BLOCK_W"]),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B=B, C_in=C_in, C_out=C_out,
        D=D, H=H, W=W,
        D_out=D_out, H_out=H_out, W_out=W_out,
        Kd=Kd, Kh=Kh, Kw=Kw,
        stride_d=stride_d, stride_h=stride_h, stride_w=stride_w,
        pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
        output_pad_d=output_pad_d, output_pad_h=output_pad_h, output_pad_w=output_pad_w,
        groups=groups,
        BLOCK_B=1,
        BLOCK_C_OUT=32,
        BLOCK_D=8,
        BLOCK_H=8,
        BLOCK_W=8,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using custom Triton kernels.
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
        
        # Initialize weights
        Kd, Kh, Kw = kernel_size
        weight = torch.empty(in_channels, out_channels // groups, Kd, Kh, Kw)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)
        
        # Initialize bias if requested
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * Kd * Kh * Kw // groups
            bound = 1 / math.sqrt(fan_in)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )