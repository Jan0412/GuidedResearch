import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    w_ptr,  # Weight tensor pointer (C, K_h, K_w) - note: K_h=kernel_size, K_w=1 for our case
    b_ptr,  # Bias pointer (C,) - optional
    out_ptr,  # Output tensor pointer (B, C, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs: we'll parallelize over (batch, channel, height_out, width_out)
    # Use 2D grid: one dimension for batch*channel combinations, another for output spatial positions
    
    # Compute batch_idx, channel_idx, and output spatial indices
    bc_idx = tl.program_id(0)
    out_w_idx = tl.program_id(1)
    
    batch_idx = bc_idx // in_channels
    channel_idx = bc_idx % in_channels
    
    # Calculate output height index from the grid dimension
    out_h_idx = tl.program_id(2) if tl.num_programs(2) > 1 else 0
    
    # Compute input spatial indices based on stride, padding, dilation
    in_h_start = out_h_idx * stride - padding
    in_w_start = out_w_idx * stride - padding
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Loop over kernel height (only kernel_size elements, since K_w=1)
    for kh in range(kernel_size):
        in_h = in_h_start + kh * dilation
        # Check if in_h is within bounds
        if 0 <= in_h < height:
            # Compute input index for this channel and spatial location
            in_idx = batch_idx * in_channels * height * width + \
                     channel_idx * height * width + \
                     in_h * width + \
                     in_w_start
                     
            # Load weight for this kernel position
            w_idx = channel_idx * kernel_size + kh
            w_val = tl.load(w_ptr + w_idx)
            
            # Load input value
            mask = (0 <= in_h < height) & (0 <= in_w_start < width)
            x_val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_val = tl.load(b_ptr + channel_idx)
        acc += b_val
    
    # Compute output index
    out_idx = batch_idx * in_channels * output_height * output_width + \
              channel_idx * output_height * output_width + \
              out_h_idx * output_width + \
              out_w_idx
    
    # Store result
    tl.store(out_ptr + out_idx, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> torch.Tensor:
    """
    Performs depthwise 2D convolution using Triton kernel.
    Assumes 1D kernel in height dimension (kernel_size, 1), but we handle general case.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    kernel_size_h, kernel_size_w = weight.shape[1], weight.shape[2]
    # For our specific case, kernel_size_w should be 1, but let's be general
    
    # Compute output dimensions
    output_height = (height + 2 * padding - dilation * (kernel_size_h - 1) - 1) // stride + 1
    output_width = (width + 2 * padding - dilation * (kernel_size_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_height, output_width, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # We'll parallelize over batch*channel combinations and output width positions
    grid = (batch_size * in_channels, output_width, output_height)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_size_h, stride, padding, dilation,
        output_height, output_width,
        BLOCK_SIZE=128
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for depthwise 2D convolution.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the weight and bias as in the original Conv2d
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.register_buffer('weight', torch.empty(in_channels, kernel_size, 1))
        if bias:
            self.register_buffer('bias', torch.empty(in_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights using kaiming uniform initialization (similar to PyTorch default)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in = kernel_size * 1  # kernel_size * 1 since kernel is (kernel_size, 1)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )