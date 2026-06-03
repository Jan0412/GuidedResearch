import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,          # Input tensor (B, C_in, L_in)
    w_ptr,          # Weight tensor (C_in, C_out, K)
    b_ptr,          # Bias tensor (C_out,) or None
    y_ptr,          # Output tensor (B, C_out, L_out)
    B, C_in, C_out, L_in, L_out, K,
    stride, padding, output_padding,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Calculate output position range
    start_pos = tl.program_id(2) * BLOCK_SIZE
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    mask = offsets < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute contributions from all input positions and input channels
    # For transposed conv: out[i] = sum_{k} in[(i - k + padding) // stride] * w[k] (when valid)
    for k in range(K):
        # For each kernel position, compute corresponding input positions
        # In transposed conv: output position i gets contribution from input position (i - k + padding) // stride
        # but only if (i - k + padding) is divisible by stride and within bounds
        
        # Calculate input positions that contribute to output positions
        in_pos = (offsets - k + padding) // stride
        valid = ((offsets - k + padding) % stride == 0) & (in_pos >= 0) & (in_pos < L_in)
        
        # Load input values
        x_offset = batch_idx * C_in * L_in + in_pos * C_in
        x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
        
        # Load weight for this kernel position and channel combination
        w_offset = k * C_out + out_channel_idx
        w_val = tl.load(w_ptr + w_offset)
        
        # Accumulate
        acc += tl.where(valid, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx)
        acc += bias
    
    # Store result
    y_offset = batch_idx * C_out * L_out + out_channel_idx * L_out + offsets
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, output_padding, groups: Convolution parameters
    """
    assert groups == 1, "Only groups=1 is supported in this implementation"
    
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + output_padding + K
    
    # Prepare output tensor
    y = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Grid dimensions: (batch, out_channels, output_length_blocks)
    BLOCK_SIZE = 256
    grid = (B, C_out, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our optimized Triton implementation
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            output_padding=self.output_padding, groups=self.groups
        )