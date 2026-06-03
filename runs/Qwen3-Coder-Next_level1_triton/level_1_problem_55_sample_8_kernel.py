import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer: (B, C_in, H, W)
    w_ptr,  # Weight tensor pointer: (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer: (C_out,) or nullptr
    out_ptr,  # Output tensor pointer: (B, C_out, H_out, W_out)
    B, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Output and kernel dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    H_out, W_out,  # Output spatial dimensions
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_c_out = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    c_out_start = pid_c_out * BLOCK_SIZE_C_out
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c_in_idx in range(C_in):
        for k_h in range(K_h):
            for k_w in range(K_w):
                # Calculate input position
                in_h = out_h * stride_h + k_h * dilation_h - pad_h
                in_w = out_w * stride_w + k_w * dilation_w - pad_w
                
                # Load input value
                x_offset = pid_batch * (C_in * H * W) + c_in_idx * (H * W) + in_h * W + in_w
                x_val = tl.load(x_ptr + x_offset, mask=(in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W), other=0.0)
                
                # Load weight value
                w_offset = c_out_start * (C_in * K_h * K_w) + c_in_idx * (K_h * K_w) + k_h * K_w + k_w
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate convolution
                output += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = c_out_start
        bias = tl.load(b_ptr + b_offset)
        output += bias
    
    # Store output
    out_offset = pid_batch * (C_out * H_out * W_out) + c_out_start * (H_out * W_out) + out_h * W_out + out_w
    tl.store(out_ptr + out_offset, output, mask=(c_out_start < C_out))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections (not used in this implementation, assumes groups=1)
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid dimensions
    # Split output channels across blocks
    # Split batch dimension across blocks  
    # Split spatial dimensions across blocks
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid_c_out = (C_out + BLOCK_SIZE_C_out - 1) // BLOCK_SIZE_C_out
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    grid = (grid_c_out, B, grid_h, grid_w)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        H_out, W_out,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=1,  # Not used in current implementation
        BLOCK_SIZE_K=1,     # Not used in current implementation
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernels for convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with the same parameters as the original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )