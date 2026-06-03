import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, length_out)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding, input_length, output_length,
    BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs: batch_idx, out_channel_idx, output_position
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_pos = tl.program_id(2)
    
    # Output pointer offset for this thread
    out_offset = batch_idx * out_channels * output_length + out_c_idx * output_length + out_pos
    
    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    
    # Compute the input position this output element depends on
    # For transposed conv: out_pos = i * stride + j - padding - output_padding (simplified)
    # More precisely: input_pos = (out_pos - (kernel_size - 1 - j) + padding + output_padding) / stride
    # We need to iterate over all kernel positions and input positions that contribute
    
    # Compute the range of kernel positions and input positions that affect this output
    # input_pos = (out_pos - j + padding + output_padding) // stride
    # We need input_pos in [0, input_length)
    
    # Iterate over kernel positions
    kernel_start = 0
    kernel_end = kernel_size
    
    # Compute valid kernel range for this output position
    # input_pos = (out_pos - kernel_pos + padding + output_padding) // stride
    # We require: 0 <= input_pos < input_length
    # => 0 <= (out_pos - kernel_pos + padding + output_padding) // stride < input_length
    # => 0 <= out_pos - kernel_pos + padding + output_padding < stride * input_length
    
    # So kernel_pos must satisfy: out_pos + padding + output_padding - stride * input_length < kernel_pos <= out_pos + padding + output_padding
    # And kernel_pos in [0, kernel_size)
    
    # Simplified approach: iterate through all kernel positions and check validity
    for kernel_pos in range(kernel_start, kernel_end):
        # Compute corresponding input position
        input_pos = (out_pos - kernel_pos + padding + output_padding) // stride
        
        # Check if this is a valid input position
        valid = (input_pos >= 0) & (input_pos < input_length) & ((out_pos - kernel_pos + padding + output_padding) % stride == 0)
        
        if valid:
            # Compute input pointer offset
            # Input shape: (batch_size, in_channels, length)
            in_offset = batch_idx * in_channels * input_length + tl.arange(0, BLOCK_K) * input_length + input_pos
            
            # Load input values (we'll process multiple input channels in parallel with BLOCK_K)
            # But we need to handle this carefully - let's restructure to process in_channels sequentially
            
    # Let's implement a cleaner version with separate loops for input channels
    pass  # Placeholder for the actual implementation below


@triton.jit
def conv_transpose1d_kernel_v2(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, length_out)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding, input_length, output_length,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs: batch_idx, out_channel_idx
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # We'll process output positions in blocks
    out_block_start = tl.program_id(2) * BLOCK_SIZE
    out_offsets = out_block_start + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < output_length
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over input channels
    for in_c in range(in_channels):
        # Iterate over kernel positions
        for k in range(kernel_size):
            # Compute corresponding input position for each output position
            # For transposed conv: out_pos = in_pos * stride + k - padding - output_padding
            # So in_pos = (out_pos - k + padding + output_padding) / stride
            in_pos = (out_offsets - k + padding + output_padding) // stride
            
            # Check validity: in_pos must be in [0, input_length) and divisible condition
            valid = (in_pos >= 0) & (in_pos < input_length) & ((out_offsets - k + padding + output_padding) % stride == 0)
            
            # Compute pointers
            x_offset = batch_idx * in_channels * input_length + in_c * input_length + in_pos
            w_offset = in_c * out_channels * kernel_size + out_c_idx * kernel_size + k
            
            # Load input and weight values
            # For input, we need to gather from in_pos which varies per output position
            # We'll use tl.where to handle invalid positions
            x_val = tl.load(x_ptr + x_offset, mask=mask & valid, other=0.0)
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += tl.where(mask & valid, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    tl.store(out_ptr + batch_idx * out_channels * output_length + out_c_idx * output_length + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


# Optimized kernel using tiling for better performance
@triton.jit
def conv_transpose1d_kernel_opt(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, length_out)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding, input_length, output_length,
    BLOCK_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,  # Block size for kernel dimension
):
    # Program IDs: batch_idx, out_channel_idx
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Process output positions in blocks
    out_block_start = tl.program_id(2) * BLOCK_SIZE
    out_offsets = out_block_start + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < output_length
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process input channels in tiles
    for in_c in range(in_channels):
        # Process kernel positions
        for k in range(kernel_size):
            # Compute corresponding input position
            in_pos = (out_offsets - k + padding + output_padding) // stride
            
            # Check validity
            valid = (in_pos >= 0) & (in_pos < input_length) & ((out_offsets - k + padding + output_padding) % stride == 0)
            
            # Compute input offset
            # Input layout: batch_size * in_channels * input_length + in_c * input_length + in_pos
            # Since in_pos varies per output position, we need to compute it carefully
            x_base_offset = batch_idx * in_channels * input_length + in_c * input_length
            # We'll use tl.where for the indexing since in_pos is dynamic
            x_val = tl.load(x_ptr + x_base_offset + in_pos, mask=mask & valid, other=0.0)
            
            # Weight is constant for this kernel position
            w_offset = in_c * out_channels * kernel_size + out_c_idx * kernel_size + k
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += tl.where(mask & valid, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    tl.store(out_ptr + batch_idx * out_channels * output_length + out_c_idx * output_length + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d forward pass.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, output_padding, groups: ConvTranspose parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_length)
    """
    batch_size, in_channels, input_length = x.shape
    kernel_size = weight.shape[2]
    out_channels = weight.shape[1]
    
    # Calculate output length
    # Formula: output_length = (input_length - 1) * stride - 2 * padding + output_padding + kernel_size
    output_length = (input_length - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 128
    
    # Grid: (batch_size, out_channels, ceil(output_length / BLOCK_SIZE))
    grid = (batch_size, out_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_transpose1d_kernel_opt[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        stride, padding, output_padding, input_length, output_length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose1d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters similar to nn.ConvTranspose1d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Ensure x is on the same device as weight
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        
        # Call our Triton implementation
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups
        )
    
    def _apply(self, fn):
        """Override _apply to handle device changes properly."""
        super()._apply(fn)
        # Re-initialize parameters after device change if needed
        if hasattr(self, 'weight') and self.weight.device != fn(torch.empty(1)).device:
            self.weight = nn.Parameter(self.weight.to(fn(torch.empty(1)).device))
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.to(fn(torch.empty(1)).device))


# Add missing import
import math