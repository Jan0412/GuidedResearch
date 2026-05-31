import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,       # Input tensor: (batch, in_channels, length)
    w_ptr,       # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,       # Bias tensor: (out_channels,)
    out_ptr,     # Output tensor: (batch, out_channels, length_out)
    batch_size,
    in_channels,
    out_channels,
    length,
    length_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Grid: (batch, out_channels, length_out)
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_l_id = tl.program_id(2)

    # Calculate the corresponding input start position
    # output_l = (input_l - 1) * stride + 1 - padding + dilation * (kernel_size - 1)
    # Solving for input_l: input_l = (output_l + padding - dilation * (kernel_size - 1) - 1) / stride + 1
    # But easier: iterate over kernel positions and accumulate
    # For each output position, we need to gather contributions from input positions
    # input_l = (out_l_id + padding - dilation * (k * dilation)) // stride + 1 ? 
    # Standard formula: out_l = (in_l - 1) * stride + 1 - padding + dilation * (kernel_size - 1)
    # So in_l = (out_l + padding - dilation * (kernel_size - 1) - 1) // stride + 1
    # Let's compute the base input index for this output position
    base_in_l = (out_l_id + padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_K), dtype=tl.float32) if BLOCK_C > 0 and BLOCK_K > 0 else 0.0
    
    # We'll accumulate over input channels and kernel positions
    # To optimize, we can tile over in_channels and kernel_size
    for c_offset in range(0, in_channels, BLOCK_C):
        c_block = tl.arange(0, BLOCK_C) + c_offset
        c_mask = c_block < in_channels
        
        for k_offset in range(0, kernel_size, BLOCK_K):
            k_block = tl.arange(0, BLOCK_K) + k_offset
            k_mask = k_block < kernel_size
            
            # Calculate input position for this kernel offset
            # input_l = base_in_l + k_offset * dilation
            in_l = base_in_l + k_offset * dilation
            
            # Check if in_l is within bounds
            in_l_mask = (in_l >= 0) & (in_l < length)
            
            # Load input tile
            x_ptr_base = x_ptr + (batch_id * in_channels + c_block) * length + in_l
            x = tl.load(x_ptr_base, mask=in_l_mask & c_mask[:, None], other=0.0)
            
            # Load weight tile
            w_ptr_base = w_ptr + (out_ch_id * in_channels + c_block) * kernel_size + k_block
            w = tl.load(w_ptr_base, mask=c_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Multiply and accumulate
            acc += tl.dot(x, w)
    
    # Store result
    out_ptr_base = out_ptr + (batch_id * out_channels + out_ch_id) * length_out + out_l_id
    tl.store(out_ptr_base, acc)


def triton_conv_transpose1d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                            stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Custom Triton implementation of 1D transposed convolution.
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = w.shape
    
    # Calculate output length
    length_out = (length - 1) * stride + 1 - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Ensure contiguous tensors
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, length_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_B = 1
    BLOCK_C = 16
    BLOCK_K = 8
    BLOCK_L = 1
    
    # Grid dimensions
    grid = (batch_size, out_channels, length_out)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, w, b, out,
        batch_size, in_channels, out_channels, length, length_out,
        kernel_size, stride, padding, dilation,
        BLOCK_B, BLOCK_C, BLOCK_K, BLOCK_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        if self.bias is not None:
            out = triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        else:
            out = triton_conv_transpose1d(x, self.weight, torch.empty(0), self.stride, self.padding, self.dilation)
        return out