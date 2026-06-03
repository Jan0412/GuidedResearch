import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (B, C_in, L_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K)
    b_ptr,  # Bias tensor: (C_out,) or nullptr
    out_ptr,  # Output tensor: (B, C_out, L_out)
    B, C_in, C_out, K, L_in, L_out,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element: out[batch, out_channel, out_pos]
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l_out = tl.program_id(2)

    # Compute the output position
    out_pos = pid_l_out

    # Offset in output tensor
    out_offset = pid_b * (C_out * L_out) + pid_c_out * L_out + out_pos

    # Accumulator
    acc = 0.0

    # Loop over input channels and kernel positions
    for c_in in range(C_in):
        for k in range(K):
            # Compute corresponding input position
            # l_out = stride * l_in + dilation * (k - padding)
            # => l_in = (l_out - dilation * (k - padding)) / stride
            l_in = (out_pos - dilation * (k - padding)) // stride
            
            # Check if l_in is in valid range and the division was exact
            if (out_pos - dilation * (k - padding)) % stride == 0 and 0 <= l_in < L_in:
                # Compute offsets
                x_offset = pid_b * (C_in * L_in) + c_in * L_in + l_in
                w_offset = c_in * (C_out * K) + pid_c_out * K + k
                
                # Load values
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_c_out
        acc += tl.load(b_ptr + b_offset)

    # Store result
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 1D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + out_padding + 1
    # Since out_padding is 0 in default ConvTranspose1d and we assume no out_padding
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Set block size for Triton kernel
    BLOCK_SIZE = 128
    
    # Grid: [batch_size, out_channels, L_out]
    grid = (B, C_out, L_out)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, K, L_in, L_out,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register weights and bias as buffers/parameters similar to nn.ConvTranspose1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights (matching nn.ConvTranspose1d initialization)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights and bias similar to nn.ConvTranspose1d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )