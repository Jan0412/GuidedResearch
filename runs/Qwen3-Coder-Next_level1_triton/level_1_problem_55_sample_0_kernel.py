import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Input tensor pointer (N, C_in, H, W)
    weight_ptr, # Weight tensor pointer (C_out, C_in, K_h, K_w)
    bias_ptr,   # Bias tensor pointer (C_out,)
    output_ptr, # Output tensor pointer (N, C_out, H_out, W_out)
    batch_size, in_channels, out_channels,
    input_height, input_width,
    kernel_size, stride, padding, dilation,
    output_height, output_width,
    BLOCK_M: tl.constexpr,  # Block size for output channels
    BLOCK_N: tl.constexpr,  # Block size for batch
    BLOCK_K: tl.constexpr,  # Block size for reduction (C_in * K_h * K_w)
):
    # Compute output position
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Batch block
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute output channel indices
    out_channels_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_channels_mask = out_channels_offsets < out_channels
    
    # Compute batch indices
    batch_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_mask = batch_offsets < batch_size
    
    # Create weight pointer offset for this output channel block
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(out_channels, in_channels * kernel_size * kernel_size),
        strides=(in_channels * kernel_size * kernel_size, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    # Create input pointer offset for this batch block
    # We'll compute for each spatial location in the output
    for oh in range(output_height):
        for ow in range(output_width):
            # Compute input position for this output location
            ih = oh * stride - padding
            iw = ow * stride - padding
            
            # Process kernel positions
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_h = ih + kh * dilation
                    in_w = iw + kw * dilation
                    
                    # Check bounds for input position
                    valid_h = (in_h >= 0) & (in_h < input_height)
                    valid_w = (in_w >= 0) & (in_w < input_width)
                    valid = valid_h & valid_w
                    
                    # Create input block pointer for current position
                    # We need to handle the case where input position is out of bounds
                    if tl.constexpr(valid):
                        input_block_ptr = tl.make_block_ptr(
                            base=input_ptr,
                            shape=(batch_size, in_channels * input_height * input_width),
                            strides=(in_channels * input_height * input_width, 1),
                            offsets=(0, 0),
                            block_shape=(BLOCK_N, BLOCK_K),
                            order=(1, 0)
                        )
                        
                        # Compute input offset for this position
                        input_offset = in_h * input_width + in_w
                        
                        # Load input and weight blocks
                        input_block = tl.load(input_block_ptr, boundary_check=(1,))
                        weight_block = tl.load(weight_block_ptr, boundary_check=(1,))
                        
                        # Accumulate multiplication
                        acc += tl.dot(input_block, weight_block.T)
                    else:
                        # Zero padding for out-of-bounds positions
                        # This is handled by loading zero values
                        pass
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        bias_mask = bias_offsets < out_channels
        bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias[:, None]
    
    # Store result
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(batch_size, out_channels * output_height * output_width),
        strides=(out_channels * output_height * output_width, 1),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M * output_height * output_width + oh * output_width + ow),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )
    
    tl.store(output_block_ptr, acc.to(tl.float32), boundary_check=(0, 1))


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton-based 2D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, height_out, width_out)
    """
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    assert kernel_size_h == kernel_size_w, "Only square kernels supported"
    kernel_size = kernel_size_h
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), 
                        dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    BLOCK_M = 32
    BLOCK_N = 8
    BLOCK_K = 32
    
    # Grid configuration
    grid = lambda meta: (
        triton.cdiv(out_channels, meta["BLOCK_M"]),
        triton.cdiv(batch_size, meta["BLOCK_N"])
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding, dilation,
        output_height, output_width,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create parameters manually to match nn.Conv2d behavior
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, 
                           stride=self.stride, padding=self.padding, 
                           dilation=self.dilation, groups=self.groups)


# Import math for initialization
import math