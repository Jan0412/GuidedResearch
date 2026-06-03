import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,              # Input tensor (B, C_in, L_in)
    w_ptr,              # Weight tensor (C_in, C_out, K)
    b_ptr,              # Bias tensor (C_out,) or nullptr
    y_ptr,              # Output tensor (B, C_out, L_out)
    B: tl.constexpr,    # Batch size
    C_in: tl.constexpr, # Input channels
    C_out: tl.constexpr, # Output channels
    L_in: tl.constexpr, # Input length
    L_out: tl.constexpr, # Output length
    K: tl.constexpr,    # Kernel size
    stride: tl.constexpr, # Stride
    padding: tl.constexpr, # Padding
    dilation: tl.constexpr, # Dilation
    BLOCK_SIZE_M: tl.constexpr, # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr, # Block size for batch
    BLOCK_SIZE_L: tl.constexpr, # Block size for output sequence length
):
    # Program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_b = tl.program_id(1)  # Batch block
    pid_l = tl.program_id(2)  # Output sequence position block
    
    # Calculate output channel range
    c_start = pid_c * BLOCK_SIZE_M
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_offsets < C_out
    
    # Calculate batch index
    batch_idx = pid_b
    
    # Calculate output sequence position range
    l_start = pid_l * BLOCK_SIZE_L
    l_offsets = l_start + tl.arange(0, BLOCK_SIZE_L)
    l_mask = l_offsets < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in_idx in range(C_in):
        for k in range(K):
            # Calculate corresponding input position
            # For transposed convolution: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
            # The relationship is: l_out = l_in * stride - padding + k * dilation
            # So l_in = (l_out + padding - k * dilation) / stride
            l_out_offsets = l_offsets
            l_in_offsets = (l_out_offsets + padding - k * dilation) // stride
            
            # Check if l_in is valid
            valid_mask = (l_in_offsets >= 0) & (l_in_offsets < L_in) & ((l_out_offsets + padding - k * dilation) % stride == 0)
            
            # Get input values
            x_batch_offset = batch_idx * C_in * L_in
            x_channel_offset = c_in_idx * L_in
            x_pos = x_batch_offset + x_channel_offset + l_in_offsets
            
            # Load x values with mask
            x_val = tl.load(x_ptr + x_pos, mask=valid_mask, other=0.0)
            
            # Get weight values
            w_channel_in_offset = c_in_idx * C_out * K
            w_channel_out_offset = c_out_idx * K if 'c_out_idx' in locals() else 0
            w_pos = w_channel_in_offset + c_offsets * K + k
            w_val = tl.load(w_ptr + w_pos, mask=c_mask, other=0.0)
            
            # Accumulate
            acc += x_val[:, None] * w_val[None, :] * valid_mask[:, None]
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_offsets, mask=c_mask, other=0.0)
        acc += bias[None, :]
    
    # Store result
    y_batch_offset = batch_idx * C_out * L_out
    y_channel_offset = c_offsets * L_out
    y_pos = y_batch_offset + y_channel_offset[None, :] + l_offsets[None, :]
    tl.store(y_ptr + y_pos, acc, mask=c_mask[None, :] & l_mask[None, :])


# More efficient implementation using a different tiling strategy
@triton.jit
def conv_transpose1d_kernel_v2(
    x_ptr,              # Input tensor (B, C_in, L_in)
    w_ptr,              # Weight tensor (C_in, C_out, K)
    b_ptr,              # Bias tensor (C_out,) or nullptr
    y_ptr,              # Output tensor (B, C_out, L_out)
    B: tl.constexpr,    # Batch size
    C_in: tl.constexpr, # Input channels
    C_out: tl.constexpr, # Output channels
    L_in: tl.constexpr, # Input length
    L_out: tl.constexpr, # Output length
    K: tl.constexpr,    # Kernel size
    stride: tl.constexpr, # Stride
    padding: tl.constexpr, # Padding
    dilation: tl.constexpr, # Dilation
    BLOCK_SIZE_CIN: tl.constexpr, # Block size for input channels
    BLOCK_SIZE_LIN: tl.constexpr, # Block size for input sequence
):
    # Calculate output position and channel
    l_out_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    # Calculate input position range for this output position
    # For transposed conv: output position l_out comes from input positions
    # where l_out = l_in * stride - padding + k * dilation
    # So l_in = (l_out + padding - k * dilation) / stride
    
    # Accumulator
    acc = 0.0
    
    # Iterate over input channels
    for c_in_idx in range(C_in):
        # Iterate over kernel positions
        for k in range(K):
            # Calculate corresponding input position
            l_in = (l_out_idx + padding - k * dilation) // stride
            remainder = (l_out_idx + padding - k * dilation) % stride
            
            # Check if valid
            if remainder == 0 and l_in >= 0 and l_in < L_in:
                # Get input value
                x_offset = batch_idx * C_in * L_in + c_in_idx * L_in + l_in
                x_val = tl.load(x_ptr + x_offset)
                
                # Get weight value
                w_offset = c_in_idx * C_out * K + c_out_idx * K + k
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        b_offset = c_out_idx
        acc += tl.load(b_ptr + b_offset)
    
    # Store result
    y_offset = batch_idx * C_out * L_out + c_out_idx * L_out + l_out_idx
    tl.store(y_ptr + y_offset, acc)


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 1D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Create output tensor
    y = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Launch kernel with grid: (L_out, C_out, B)
    grid = (L_out, C_out, B)
    
    # Calculate kernel parameters as compile-time constants
    BLOCK_SIZE_CIN = 1
    BLOCK_SIZE_LIN = 1
    
    # Launch kernel
    conv_transpose1d_kernel_v2[grid](
        x, weight, bias, y,
        B=B, C_in=C_in, C_out=C_out, L_in=L_in, L_out=L_out,
        K=K, stride=stride, padding=padding, dilation=dilation,
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN, BLOCK_SIZE_LIN=BLOCK_SIZE_LIN
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters (same as nn.ConvTranspose1d)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Ensure input is on GPU and contiguous
        x = x.contiguous()
        
        # Call the Triton implementation
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )


# Create a wrapper that initializes weights properly like nn.ConvTranspose1d
def create_model_from_config(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
    model = ModelNew(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    # Initialize weights similar to nn.ConvTranspose1d
    nn.init.kaiming_uniform_(model.weight, a=math.sqrt(5))
    if model.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(model.bias, -bound, bound)
    return model


# Import math for initialization
import math