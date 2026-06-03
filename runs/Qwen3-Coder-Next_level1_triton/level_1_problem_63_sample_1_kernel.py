import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, height, width)
    w_ptr,  # Weight tensor (out_channels, in_channels, kernel_size, kernel_size)
    b_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (batch, out_channels, height_out, width_out)
    batch_size, in_channels, out_channels, 
    height, width, 
    height_out, width_out,
    kernel_size, stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, 
    BLOCK_C: tl.constexpr, BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output spatial coordinates
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Calculate input spatial coordinates
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Create ranges for output block
    h_offsets = tl.arange(0, BLOCK_H)
    w_offsets = tl.arange(0, BLOCK_W)
    out_h_indices = out_h + h_offsets
    out_w_indices = out_w + w_offsets
    
    # Create masks for output block
    h_mask = out_h_indices < height_out
    w_mask = out_w_indices < width_out
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution loop over input channels and kernel positions
    for c in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position
                in_h = in_h_start + kh * dilation
                in_w = in_w_start + kw * dilation
                
                # Create masks for input position
                in_h_indices = in_h + h_offsets
                in_w_indices = in_w + w_offsets
                in_h_mask = (in_h_indices >= 0) & (in_h_indices < height)
                in_w_mask = (in_w_indices >= 0) & (in_w_indices < width)
                in_hw_mask = in_h_mask[:, None] & in_w_mask[None, :]
                
                # Load input values
                x_offset = (pid_batch * in_channels * height * width + 
                           c * height * width + 
                           in_h_indices[:, None] * width + 
                           in_w_indices[None, :])
                x_vals = tl.load(x_ptr + x_offset, mask=in_hw_mask, other=0.0)
                
                # Load weight values
                w_offset = (pid_out_c * in_channels * kernel_size * kernel_size + 
                           c * kernel_size * kernel_size + 
                           kh * kernel_size + 
                           kw)
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate convolution
                acc += x_vals * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_out_c
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store output
    out_offset = (pid_batch * out_channels * height_out * width_out + 
                 pid_out_c * height_out * width_out + 
                 out_h_indices[:, None] * width_out + 
                 out_w_indices[None, :])
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor (batch, in_channels, height, width)
        weight: Weight tensor (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias tensor (out_channels,) or None
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    # Calculate output dimensions
    height_out = (height + 2 * padding - dilation * (kernel_size_h - 1) - 1) // stride + 1
    width_out = (width + 2 * padding - dilation * (kernel_size_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, 
                     dtype=x.dtype, device=x.device)
    
    # Kernel configuration
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C = 4  # Process multiple input channels at once
    BLOCK_KH = 3
    BLOCK_KW = 3
    
    # Grid dimensions: (batch, out_channels, height_blocks, width_blocks)
    grid = (
        batch_size,
        out_channels,
        (height_out + BLOCK_H - 1) // BLOCK_H,
        (width_out + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width,
        height_out, width_out,
        kernel_size_h, stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C, BLOCK_KH=BLOCK_KH, BLOCK_KW=BLOCK_KW
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel,
    optimized with Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights with Kaiming initialization
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 
            (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )