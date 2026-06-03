import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor (batch_size, in_channels, length)
    w_ptr,  # Weight tensor (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Calculate output position
    out_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < out_length
    
    # Compute the starting position in input for each output position
    # input_start = out_idx * stride - padding
    input_start = out_idx * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel positions
    for k in range(kernel_size):
        # Calculate input position: input_start + k * dilation
        input_idx = input_start + k * dilation
        input_mask = (input_idx >= 0) & (input_idx < length) & mask
        
        # Load kernel weight for this position
        w_offset = out_channel_id * in_channels * kernel_size + k
        w_val = tl.load(w_ptr + w_offset)
        
        # Loop over input channels
        for c in range(in_channels):
            # Load input value
            x_offset = batch_id * in_channels * length + c * length + input_idx
            x_val = tl.load(x_ptr + x_offset, mask=input_mask, other=0.0)
            
            # Accumulate
            acc += x_val * w_val
    
    # Apply bias if provided
    if b_ptr is not None:
        b_offset = out_channel_id
        b_val = tl.load(b_ptr + b_offset)
        acc += b_val
    
    # Convert to output dtype and store
    out_val = acc.to(x_ptr.dtype.element_ty)
    out_offset = batch_id * out_channels * out_length + out_channel_id * out_length + out_idx
    tl.store(out_ptr + out_offset, out_val, mask=mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Performs 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    assert groups == 1, "Only groups=1 is supported in this implementation"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Set up grid
    BLOCK_SIZE = 256  # Tunable parameter
    grid = (batch_size, out_channels, (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights using kaiming_uniform initialization (similar to PyTorch)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in * kernel_size)**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups)