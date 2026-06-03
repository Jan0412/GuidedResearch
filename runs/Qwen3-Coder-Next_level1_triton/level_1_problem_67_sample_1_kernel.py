import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Batch index
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Calculate starting position for output
    # Each block computes multiple output positions along the sequence dimension
    out_pos_start = tl.program_id(2) * BLOCK_SIZE_N
    
    # Create offsets for output positions
    out_offsets = out_pos_start + tl.arange(0, BLOCK_SIZE_N)
    out_mask = out_offsets < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over in_channels and kernel positions
    for ic in range(in_channels):
        for k in range(kernel_size):
            # Calculate input position with dilation and padding
            input_pos = out_offsets * stride + k * dilation - padding
            
            # Create mask for valid input positions
            in_mask = (input_pos >= 0) & (input_pos < length)
            
            # Calculate input pointer offset for this batch, channel, and position
            x_offset = batch_idx * in_channels * length + ic * length + input_pos
            x_offset = tl.where(in_mask, x_offset, 0)
            
            # Load input values (with masking)
            x_val = tl.load(x_ptr + x_offset, mask=in_mask, other=0.0)
            
            # Calculate weight pointer offset
            w_offset = out_channel_idx * in_channels * kernel_size + ic * kernel_size + k
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = out_channel_idx
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = batch_idx * out_channels * out_length + out_channel_idx * out_length + out_offsets
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


class TritonConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        batch_size, in_channels, length = x.shape
        out_channels, _, kernel_size = weight.shape
        
        # Calculate output length
        out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE_M = 1  # batch size dimension (1 per block)
        BLOCK_SIZE_N = 256  # output sequence length dimension
        BLOCK_SIZE_K = 32   # not used directly but kept for consistency
        
        # Grid dimensions: (batch_size, out_channels, out_length // BLOCK_SIZE_N + 1)
        grid = (batch_size, out_channels, triton.cdiv(out_length, BLOCK_SIZE_N))
        
        # Launch kernel
        conv1d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels, length, out_length, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Save for backward pass (not implemented in this example, but needed for autograd)
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.out_length = out_length
        
        return out


class TritonConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return TritonConv1d.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with Triton optimization.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Note: groups parameter is not fully implemented in this Triton kernel
        # For simplicity, we assume groups=1 (standard convolution)
        if groups != 1:
            raise ValueError("Triton Conv1d implementation only supports groups=1")
            
        self.conv1d = TritonConv1dLayer(in_channels, out_channels, kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation, 
                                        bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d(x)