import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor (N, C_in, H, W)
    w_ptr,  # Weight tensor (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,) - optional
    out_ptr,  # Output tensor (N, C_out, H_out, W_out)
    # Tensor dimensions
    N, C_in, H, W,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    # Block sizes for tiling
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Block size for H dimension
    BLOCK_N: tl.constexpr,  # Block size for W dimension
):
    # Get program IDs
    pid_c_out = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output positions
    out_h = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)
    out_w = pid_w * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for valid output positions
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator for the convolution
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_idx in range(0, C_in, BLOCK_C_in):
        c_in_range = c_in_idx + tl.arange(0, BLOCK_C_in)
        mask_c_in = c_in_range < C_in
        
        for kh in range(0, K_h, BLOCK_Kh):
            kh_range = kh + tl.arange(0, BLOCK_Kh)
            mask_kh = kh_range < K_h
            
            for kw in range(0, K_w, BLOCK_Kw):
                kw_range = kw + tl.arange(0, BLOCK_Kw)
                mask_kw = kw_range < K_w
                
                # Calculate input position for this kernel element
                in_h = out_h[:, None, None] * stride_h - pad_h + kh_range[None, :, None] * dil_h
                in_w = out_w[None, :, :] * stride_w - pad_w + kw_range[None, None, :] * dil_w
                
                # Create masks for valid input positions
                mask_in_h = (in_h >= 0) & (in_h < H)
                mask_in_w = (in_w >= 0) & (in_w < W)
                mask_in = mask_in_h & mask_in_w & mask_c_in[None, :, None] & mask_kw[None, None, :]
                
                # Load input values
                # x_ptr shape: (N, C_in, H, W)
                x_offset = (pid_n * C_in * H * W + 
                           c_in_range[None, :, None, None] * H * W + 
                           in_h[:, None, :, None] * W + 
                           in_w[None, :, None, :])
                x_val = tl.load(x_ptr + x_offset, mask=mask_in, other=0.0)
                
                # Load weight values
                # w_ptr shape: (C_out, C_in, K_h, K_w)
                w_offset = (pid_c_out * C_in * K_h * K_w + 
                           c_in_range[:, None, None] * K_h * K_w + 
                           kh_range[None, :, None] * K_w + 
                           kw_range[None, None, :])
                w_val = tl.load(w_ptr + w_offset, mask=mask_c_in[:, None, None] & mask_kh[None, :, None] & mask_kw[None, None, :], other=0.0)
                
                # Accumulate convolution result
                acc += tl.sum(x_val * w_val[None, :, :, :], axis=(1, 2, 3))
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_offset = (pid_n * C_out * H_out * W_out + 
                 pid_c_out * H_out * W_out + 
                 out_h[:, None] * W_out + 
                 out_w[None, :])
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor:
    """
    Performs 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tiling
    BLOCK_C_out = 16  # Block size for output channels
    BLOCK_C_in = 16   # Block size for input channels
    BLOCK_Kh = 3      # Block size for kernel height
    BLOCK_Kw = 3      # Block size for kernel width
    BLOCK_M = 8       # Block size for height dimension
    BLOCK_N = 32      # Block size for width dimension
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(C_out, BLOCK_C_out),
        N,
        triton.cdiv(H_out, BLOCK_M),
        triton.cdiv(W_out, BLOCK_N),
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out, K_h, K_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        H_out, W_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_Kh=BLOCK_Kh,
        BLOCK_Kw=BLOCK_Kw,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers instead of nn.Parameter to avoid gradient computation issues
        # We'll handle weights manually for the Triton kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias as buffers (not parameters)
        self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )


import math  # Add math import for initialization