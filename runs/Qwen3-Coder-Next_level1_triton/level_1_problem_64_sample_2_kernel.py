import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, L_in)
    w_ptr,  # Weight tensor (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor (out_channels,) - can be None
    y_ptr,  # Output tensor (batch, out_channels, L_out)
    batch_size, 
    in_channels, 
    out_channels, 
    kernel_size,
    stride,
    padding,
    output_padding,
    L_in,  # Input length
    L_out,  # Output length
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch size dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for out_channels dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels dimension
    BLOCK_SIZE_L: tl.constexpr,  # Block size for length dimension
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_length = tl.program_id(2)
    
    # Calculate output position
    out_start = pid_length * BLOCK_SIZE_L
    if out_start >= L_out:
        return
    
    # Calculate input position based on stride and padding
    # For transposed convolution: out_pos = in_pos * stride + (kernel_pos - padding)
    # So for a given out_pos, we need to find contributing in_pos values
    # in_pos = (out_pos - (kernel_pos - padding)) / stride
    
    # Compute bias offset
    bias_ptr = b_ptr + pid_out_c
    has_bias = b_ptr is not None
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Iterate over kernel size and input channels
    for k in range(kernel_size):
        for ic in range(in_channels):
            # Calculate input position for this kernel position
            # out_pos = in_pos * stride + k - padding + output_padding_offset
            # Rearranging: in_pos = (out_pos - k + padding - output_padding) / stride
            
            # For each output position in our block
            offsets_l = tl.arange(0, BLOCK_SIZE_L)
            out_pos = out_start + offsets_l
            
            # Calculate corresponding input position
            # We need to handle the case where multiple input positions contribute to the same output
            
            # Compute base input position for this kernel position
            base_in_pos = (out_pos - k + padding) // stride
            
            # Check if this input position is valid
            valid_mask = (base_in_pos >= 0) & (base_in_pos < L_in)
            
            # Check if there's an exact match (no fractional position)
            remainder = (out_pos - k + padding) % stride
            exact_match = (remainder == 0)
            
            # Only process positions where exact match exists
            effective_mask = valid_mask & exact_match
            
            if tl.sum(effective_mask) > 0:
                # Get the actual input positions
                in_pos = base_in_pos
                
                # Compute indices for input tensor
                # x_ptr shape: (batch, in_channels, L_in)
                x_offset = pid_batch * (in_channels * L_in) + ic * L_in + in_pos
                
                # Compute indices for weight tensor
                # w_ptr shape: (in_channels, out_channels, kernel_size)
                w_offset = ic * (out_channels * kernel_size) + pid_out_c * kernel_size + k
                
                # Load input values
                x_val = tl.load(x_ptr + x_offset, mask=effective_mask, other=0.0)
                
                # Load weight value
                w_val = tl.load(w_ptr + w_offset)
                
                # Multiply and accumulate
                acc += tl.where(effective_mask, x_val * w_val, 0.0)
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr)
        acc += tl.where(tl.arange(0, BLOCK_SIZE_L) < BLOCK_SIZE_L, bias_val, 0.0)
    
    # Store result
    out_offsets = pid_batch * (out_channels * L_out) + pid_out_c * L_out + out_start + tl.arange(0, BLOCK_SIZE_L)
    out_mask = out_offsets < batch_size * out_channels * L_out
    tl.store(y_ptr + out_offsets, acc.to(tl.float32), mask=out_mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose1d.
    Note: This implementation assumes groups=1 for simplicity.
    For groups > 1, the weight tensor layout needs to be handled differently.
    """
    batch_size, in_channels, L_in = x.shape
    # Weight shape: (in_channels, out_channels, kernel_size) for non-grouped
    # But PyTorch uses (in_channels, out_channels // groups, kernel_size)
    
    kernel_size = weight.shape[2]
    out_channels = weight.shape[1] * groups if groups > 1 else weight.shape[1]
    
    # Calculate output length
    # L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    y = torch.empty(batch_size, out_channels, L_out, dtype=x.dtype, device=x.device)
    
    # Check if bias is provided
    has_bias = bias is not None
    bias_ptr = bias.data_ptr() if has_bias else None
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 1  # batch size block (single batch per program for simplicity)
    BLOCK_SIZE_N = 16  # out_channels block
    BLOCK_SIZE_K = 1  # in_channels block
    BLOCK_SIZE_L = 64  # length block
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        (L_out + meta['BLOCK_SIZE_L'] - 1) // meta['BLOCK_SIZE_L']
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias_ptr, y,
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding, output_padding, L_in, L_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized Model with Triton kernel for transposed 1D convolution.
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
        self.bias_flag = bias
        
        # Create weight and bias parameters
        if groups == 1:
            self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        else:
            # For grouped convolutions: (in_channels, out_channels // groups, kernel_size)
            self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size))
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        # Call the Triton kernel
        return triton_conv_transpose1d(x, weight, bias, 
                                       self.stride, self.padding, 
                                       self.output_padding, self.groups)


# Import math for calculations
import math