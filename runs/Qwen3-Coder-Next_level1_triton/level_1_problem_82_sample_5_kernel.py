import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, K, K)
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (B, C, H_out, W_out)
    batch_size,  # B
    in_channels,  # C
    height,  # H
    width,  # W
    kernel_size,  # K
    stride,  # S
    padding,  # P
    out_height,  # H_out
    out_width,  # W_out
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel (depthwise convolution)
    channel_id = tl.program_id(0)
    # Batch and spatial position
    batch_id = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    # Calculate input coordinates corresponding to this output position
    h_start = out_h * stride - padding
    w_start = out_w * stride - padding

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over kernel dimensions
    for kh in range(kernel_size):
        h_idx = h_start + kh
        for kw in range(kernel_size):
            w_idx = w_start + kw
            
            # Check bounds for input coordinates
            valid_h = (h_idx >= 0) & (h_idx < height)
            valid_w = (w_idx >= 0) & (w_idx < width)
            valid = valid_h & valid_w
            
            # Calculate input pointer offset
            input_offset = (batch_id * in_channels * height * width + 
                           channel_id * height * width + 
                           h_idx * width + w_idx)
            
            # Load input value if valid
            x_val = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
            
            # Calculate weight pointer offset
            weight_offset = (channel_id * kernel_size * kernel_size + 
                           kh * kernel_size + kw)
            
            # Load weight value
            w_val = tl.load(w_ptr + weight_offset)
            
            # Accumulate
            acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_id)
        acc += bias

    # Store result to output
    output_offset = (batch_id * in_channels * out_height * out_width + 
                    channel_id * out_height * out_width + 
                    out_h * out_width + out_w)
    
    tl.store(out_ptr + output_offset, acc.to(x_ptr.dtype.element_ty))


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Triton implementation of depthwise 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, 1, kernel_size, kernel_size)
        bias: Optional bias tensor of shape (in_channels,)
        stride: Stride for convolution
        padding: Padding applied to input
        
    Returns:
        Output tensor of shape (batch_size, in_channels, out_height, out_width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, 
                     dtype=x.dtype, device=x.device)
    
    # Configure grid dimensions
    # Grid: (num_channels, batch_size, out_height, out_width)
    # We'll use a grid that processes multiple output positions per program for better occupancy
    BLOCK_SIZE = 1
    
    grid = (in_channels, batch_size, out_height, out_width)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_size, stride, padding, out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for depthwise convolution.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as original Model
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create the weight and bias parameters to match original Conv2d behavior
        # Note: We'll use the same parameter names for compatibility but implement forward manually
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      stride=self.stride, padding=self.padding)