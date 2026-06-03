import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor (B, C, H, W)
    w_ptr,              # Weight tensor (C, 1, K_h, K_w)
    b_ptr,              # Bias tensor (C,) - can be None
    out_ptr,            # Output tensor (B, C, H_out, W_out)
    batch_size,         # B
    in_channels,        # C
    height,             # H
    width,              # W
    out_height,         # H_out
    out_width,          # W_out
    kernel_size,        # K_h = K_w = K
    stride,             # stride
    padding,            # padding
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the 2D grid
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_hw = tl.program_id(2)  # combined h_out, w_out index
    
    # Decode pid_hw into h_out and w_out
    h_out = pid_hw // out_width
    w_out = pid_hw % out_width
    
    # Compute the corresponding input position
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Calculate the kernel half-size
    kernel_half = kernel_size // 2
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Loop over the kernel
    for kh in range(kernel_size):
        h = h_in + kh
        # Check if h is within valid bounds
        if h >= 0 and h < height:
            for kw in range(kernel_size):
                w = w_in + kw
                # Check if w is within valid bounds
                if w >= 0 and w < width:
                    # Compute input index: (pid_b, pid_c, h, w)
                    x_offset = ((pid_b * in_channels + pid_c) * height + h) * width + w
                    # Compute weight index: (pid_c, 0, kh, kw)
                    w_offset = (pid_c * kernel_size + kh) * kernel_size + kw
                    
                    x_val = tl.load(x_ptr + x_offset)
                    w_val = tl.load(w_ptr + w_offset)
                    acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offset = pid_c
        acc += tl.load(b_ptr + bias_offset)
    
    # Compute output index: (pid_b, pid_c, h_out, w_out)
    out_offset = ((pid_b * in_channels + pid_c) * out_height + h_out) * out_width + w_out
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, 1, kernel_size, kernel_size)
        bias: Optional bias tensor of shape (in_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
    
    Returns:
        Output tensor of shape (batch_size, in_channels, out_height, out_width)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, 
                      dtype=x.dtype, device=x.device)
    
    # Define grid: (batch_size, in_channels, out_height * out_width)
    grid = (batch_size, in_channels, out_height * out_width)
    
    # Launch kernel with reasonable block size (not really needed for this 2D grid, but kept for compatibility)
    BLOCK_SIZE = 1  # Not used in this configuration but required by triton
    
    # Compute constants for kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        out_height, out_width, kernel_size,
        stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Uses optimized Triton kernel instead of PyTorch's native implementation.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (same as nn.Conv2d with groups=in_channels)
        # Weight shape: (in_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      stride=self.stride, padding=self.padding)