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
    batch_size, in_channels, out_channels, 
    kernel_size, stride, padding, output_padding,
    L_in, L_out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs: 
    # pid_b: batch index
    # pid_c_out: output channel index
    # pid_l: position in output sequence
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Calculate output position
    out_pos = pid_l
    
    # Calculate corresponding input position range
    # For transposed conv: out_pos = i * stride + k - padding + output_padding_offset
    # where i is input position, k is kernel position
    # So i = (out_pos + padding - k + output_padding_offset) / stride
    
    # Accumulator for the output
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate corresponding input position
            # For each kernel position k, the output position out_pos receives contribution from input position i
            # where out_pos = i * stride + k - padding + (optional output_padding)
            # So i = (out_pos + padding - k + output_padding) / stride
            
            # The effective position in input that contributes to out_pos
            i = (out_pos + padding - k) // stride
            
            # Check if i is within valid input range
            if i >= 0 and i < L_in:
                # Calculate the exact offset in input
                input_offset = pid_b * (in_channels * L_in) + c_in * L_in + i
                
                # Load input value
                x_offset = input_offset
                x_val = tl.load(x_ptr + x_offset)
                
                # Load corresponding weight
                # Weight layout: (in_channels, out_channels, kernel_size)
                weight_offset = c_in * (out_channels * kernel_size) + pid_c_out * kernel_size + k
                w_val = tl.load(w_ptr + weight_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_c_out
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    out_offset = pid_b * (out_channels * L_out) + pid_c_out * L_out + out_pos
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose1d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    assert groups == 1, "Only groups=1 supported in this implementation"
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, L_in = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    # L_out = (L_in - 1) * stride - 2 * padding + output_padding + kernel_size
    L_out = (L_in - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, L_out, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (batch_size, out_channels, L_out)
    
    # Launch kernel with reasonable block size
    BLOCK_SIZE = 1
    BLOCK_K = 1
    
    # Note: We use a simple 3D grid where each thread computes one output element
    # This is not the most efficient but correct implementation
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding, output_padding,
        L_in, L_out,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with same parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters (similar to PyTorch default initialization)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our Triton-based transposed convolution
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding,
            groups=self.groups
        )