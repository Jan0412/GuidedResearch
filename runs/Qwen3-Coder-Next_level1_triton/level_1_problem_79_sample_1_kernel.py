import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (B, C_in, L_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, L_out)
    B, C_in, C_out, K, L_in, L_out,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Get the batch index, output channel, and position in output sequence
    batch_idx = tl.program_id(0)
    out_c = tl.program_id(1)
    pos_out = tl.program_id(2)
    
    # Calculate the output position for this thread
    start_pos = pos_out * BLOCK_SIZE
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    mask = offsets < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(C_in):
        # Calculate the corresponding input positions for this output position
        # For transposed convolution: out_pos = in_pos * stride + (kernel_pos - 1) * dilation - padding
        # => in_pos = (out_pos + padding - (kernel_pos - 1) * dilation) / stride
        
        # We need to iterate over kernel positions
        for kernel_pos in range(K):
            # Calculate input position
            in_pos = (offsets + padding - kernel_pos * dilation) // stride
            
            # Check if the input position is valid
            valid_mask = (in_pos >= 0) & (in_pos < L_in) & (mask)
            
            # Only process if there are valid positions
            if tl.sum(valid_mask) > 0:
                # Get the valid input positions
                in_offsets = in_pos * valid_mask
                in_offsets = tl.where(valid_mask, in_offsets, 0)
                
                # Load input values
                x_offsets = batch_idx * C_in * L_in + in_c * L_in + in_offsets
                x_vals = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
                
                # Load corresponding weight
                w_offset = in_c * C_out * K + out_c * K + kernel_pos
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc = tl.where(valid_mask, acc + x_vals * w_val, acc)
    
    # Apply bias if present
    if HAS_BIAS:
        b_val = tl.load(b_ptr + out_c)
        acc = acc + b_val
    
    # Store output
    out_offsets = batch_idx * C_out * L_out + out_c * L_out + offsets
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Triton implementation of transposed 1D convolution.
    
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
    C_out = weight.shape[1]
    K = weight.shape[2]
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + out_padding + 1
    # For standard transposed convolution with out_padding=0: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, L_out, dtype=x.dtype, device=x.device)
    
    # Determine block size and grid
    BLOCK_SIZE = 128
    grid = (B, C_out, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    triton_conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, K, L_in, L_out,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_BIAS=bias is not None
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.
    Uses optimized Triton kernel for inference.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters (same as PyTorch default initialization)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Initialize weight using kaiming_uniform_ (same as PyTorch)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Check if we need to use the Triton kernel or fall back to PyTorch
        # For now, always use Triton kernel
        if self.bias is not None:
            return triton_conv_transpose1d(x, self.weight, self.bias, 
                                          self.stride, self.padding, self.dilation)
        else:
            return triton_conv_transpose1d(x, self.weight, None, 
                                          self.stride, self.padding, self.dilation)


# Import math for initialization
import math