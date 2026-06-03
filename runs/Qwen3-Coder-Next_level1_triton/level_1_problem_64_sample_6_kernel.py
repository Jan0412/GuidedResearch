import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, L_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, L_out)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding, input_length, output_length,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to output positions
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # output channel index
    pid_l = tl.program_id(2) * BLOCK_SIZE  # output position index
    
    # Output position
    out_pos = pid_l + tl.arange(0, BLOCK_SIZE)
    out_mask = out_pos < output_length
    
    # Calculate input position range that contributes to this output position
    # For transposed convolution: out_pos = in_pos * stride + (kernel_pos - padding)
    # => in_pos = (out_pos - (kernel_pos - padding)) / stride
    
    # Accumulator for the output
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Compute corresponding input position
            # out_pos = in_pos * stride + k - padding + output_padding_offset
            # Let's derive: standard transposed conv formula
            # L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
            # For position mapping: out_pos = in_pos * stride + k - padding
            in_pos = (out_pos - k + padding) // stride
            
            # Check if in_pos is within valid range
            in_pos_mask = (in_pos >= 0) & (in_pos < input_length) & ((out_pos - k + padding) % stride == 0)
            
            # Get input value
            x_offset = pid_b * (in_channels * input_length) + ic * input_length + in_pos
            x_val = tl.load(x_ptr + x_offset, mask=in_pos_mask & out_mask, other=0.0)
            
            # Get weight value
            w_offset = ic * (out_channels * kernel_size) + pid_c * kernel_size + k
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_c
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = pid_b * (out_channels * output_length) + pid_c * output_length + out_pos
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, output_padding):
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, output_length), dtype=x.dtype, device=x.device)
    
    # Set kernel parameters
    BLOCK_SIZE = 128
    
    # Grid: (batch_size, out_channels, ceil(output_length / BLOCK_SIZE))
    grid = (batch_size, out_channels, triton.cdiv(output_length, BLOCK_SIZE))
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        stride, padding, output_padding, input_length, output_length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights using the same logic as ConvTranspose1d
        # Note: PyTorch uses (in_channels, out_channels/groups, kernel_size) for weights
        # but our kernel expects (in_channels, out_channels, kernel_size)
        # Since groups=1 is standard for transposed conv, we'll handle groups=1 case primarily
        
        # Create weight parameter with proper initialization
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Same initialization as PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Handle groups - for simplicity, assuming groups=1 (standard case)
        # In a full implementation, we'd need to handle grouped transposed convolution
        if self.groups != 1:
            raise NotImplementedError("Groups > 1 not implemented in this Triton kernel")
        
        # Call our optimized Triton kernel
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding
        )


# Import math for initialization
import math