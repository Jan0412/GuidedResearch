import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr, y_ptr, out_ptr, bias_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    height, width, depth,
    kernel_size,
    stride, padding, dilation,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d,
    # Block sizes
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Block size for kernel dimension
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * stride + padding
    out_w = pid_w * stride + padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels and depth
    for ic in range(in_channels):
        for d_out in range(BLOCK_D_OUT):
            d_in = d_out * stride + padding
            if d_in >= depth:
                break
                
            # Loop over kernel positions
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Compute input position
                    h_in = out_h + kh * dilation
                    w_in = out_w + kw * dilation
                    
                    # Check bounds
                    mask_h = (h_in >= 0) & (h_in < height)
                    mask_w = (w_in >= 0) & (w_in < width)
                    
                    # Load input values
                    x_offset = (pid_b * x_stride_b + ic * x_stride_c + 
                               h_in * x_stride_h + w_in * x_stride_w + d_in * x_stride_d)
                    x_vals = tl.load(x_ptr + x_offset, 
                                   mask=(mask_h & mask_w), 
                                   other=0.0)
                    
                    # Load kernel values
                    k_offset = (pid_c * (kernel_size * kernel_size * in_channels) + 
                               kh * (kernel_size * in_channels) + 
                               kw * in_channels + ic)
                    k_vals = tl.load(y_ptr + k_offset)
                    
                    # Accumulate
                    acc += x_vals * k_vals
    
    # Add bias if available
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias
    
    # Store result
    out_offset = (pid_b * out_stride_b + pid_c * out_stride_c + 
                 pid_h * out_stride_h + pid_w * out_stride_w)
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Custom Triton implementation of 3D convolution optimized for (k,k,1) kernels.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, height, width, depth = x.shape
    out_channels, _, kernel_size_h, kernel_size_w, kernel_size_d = weight.shape
    
    # For our specific architecture, kernel_size_h == kernel_size_w and kernel_size_d == 1
    kernel_size = kernel_size_h
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_d = (depth + 2 * padding - dilation * (kernel_size_d - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, out_d, 
                     dtype=x.dtype, device=x.device)
    
    # Calculate strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d = x.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d = out.stride()
    
    # Define grid
    # For the kernel, we'll process multiple output channels at once
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_D_OUT = 1  # Since kernel_d = 1, we process one depth slice at a time
    BLOCK_K = 16  # Not used in this simplified version but kept for extensibility
    
    # Grid dimensions
    grid = (batch_size, out_channels, 
            triton.cdiv(out_h, BLOCK_H), triton.cdiv(out_w, BLOCK_W))
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, out, bias,
        batch_size, in_channels, out_channels,
        height, width, depth,
        kernel_size,
        stride, padding, dilation,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_D_OUT=BLOCK_D_OUT, BLOCK_K=BLOCK_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model using custom Triton kernels for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with same parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weights and bias (same as nn.Conv3d)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)


import math