import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,           # Input tensor: (batch, in_channels, length)
    w_ptr,           # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,           # Bias tensor: (out_channels,) or None
    out_ptr,         # Output tensor: (batch, out_channels, out_length)
    batch_size, 
    in_channels, 
    out_channels, 
    in_length,
    out_length,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and output channel indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Output position
    out_pos = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_pos < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute the effective kernel size considering dilation
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    
    # For each input position that contributes to this output position
    for k in range(kernel_size):
        # Calculate corresponding input position
        # For transposed convolution: out_pos = in_pos * stride + k * dilation - padding
        # So in_pos = (out_pos + padding - k * dilation) / stride
        in_pos = (out_pos + padding - k * dilation) // stride
        
        # Check if in_pos is valid and if the division was exact
        in_pos_valid = (in_pos >= 0) & (in_pos < in_length)
        in_pos_valid = in_pos_valid & ((out_pos + padding - k * dilation) % stride == 0)
        
        # Load input values
        x_offsets = batch_idx * in_channels * in_length + tl.arange(0, in_channels) * in_length + in_pos
        x_vals = tl.load(x_ptr + x_offsets[:, None], mask=(tl.arange(0, in_channels) < in_channels) & in_pos_valid[:, None], other=0.0)
        
        # Load weight values
        w_offsets = k * out_channels + out_c_idx * kernel_size + tl.arange(0, in_channels) * kernel_size
        w_vals = tl.load(w_ptr + w_offsets, mask=tl.arange(0, in_channels) < in_channels, other=0.0)
        
        # Accumulate: sum over in_channels
        acc += tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + out_c_idx)
        acc += b_val
    
    # Store result
    out_offsets = batch_idx * out_channels * out_length + out_c_idx * out_length + out_pos
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, in_channels, in_length = x.shape
    in_channels_w, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Grid dimensions: (batch_size, out_channels, blocks_over_out_length)
    BLOCK_SIZE = 128
    grid = (batch_size, out_channels, (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_length, out_length,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation. Uses optimized Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv_transpose1d(x, self.weight, self.bias, 
                                       self.stride, self.padding, self.dilation)