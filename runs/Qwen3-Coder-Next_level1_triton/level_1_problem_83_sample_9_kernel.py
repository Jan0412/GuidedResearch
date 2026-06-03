import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, K) where K is kernel_size
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (N, C, H, W_out)
    n, c, h, w,  # Input dimensions
    kernel_size,  # Size of the 1D kernel
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    n_out,  # Output width dimension
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs: batch (n_id), channel (c_id), height (h_id)
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    height_id = tl.program_id(2)
    
    # Calculate input and output pointers for this batch, channel, and height
    input_offset = batch_id * c * h * w + channel_id * h * w + height_id * w
    output_offset = batch_id * c * h * n_out + channel_id * h * n_out + height_id * n_out
    
    # Load bias if available
    bias_val = 0.0
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + channel_id)
    
    # Process output width positions in blocks
    out_width_start = 0
    while out_width_start < n_out:
        # Compute output width positions for this block
        out_width_offsets = out_width_start + tl.arange(0, BLOCK_SIZE_W)
        mask_out = out_width_offsets < n_out
        
        # Calculate corresponding input positions for each output position
        # For depthwise conv: each output position depends on kernel_size input positions
        input_offsets = tl.zeros((BLOCK_SIZE_W,), dtype=tl.int32)
        
        for k in range(kernel_size):
            # Calculate input position: out_pos * stride - padding + k * dilation
            in_pos = out_width_offsets * stride - padding + k * dilation
            # Create mask for valid input positions
            mask_valid = (in_pos >= 0) & (in_pos < w)
            # For invalid positions, we'll use offset 0 and mask will be 0
            input_offsets = tl.where(
                mask_valid, 
                input_offsets + in_pos * mask_valid,
                input_offsets
            )
        
        # Load input values: (BLOCK_SIZE_W, kernel_size)
        x_block = tl.zeros((BLOCK_SIZE_W, kernel_size), dtype=tl.float32)
        for k in range(kernel_size):
            in_pos_k = out_width_offsets * stride - padding + k * dilation
            mask_k = (in_pos_k >= 0) & (in_pos_k < w)
            x_k = tl.load(x_ptr + input_offset + in_pos_k, mask=mask_k, other=0.0)
            x_block = tl.where(
                tl.arange(0, BLOCK_SIZE_W) < n_out,
                x_block + tl.expand_dims(x_k, 1) * tl.cast(mask_k, tl.float32),
                x_block
            )
        
        # Load kernel weights: (kernel_size,)
        kernel_offsets = k_offsets = tl.arange(0, kernel_size)
        w_vals = tl.load(w_ptr + channel_id * kernel_size + k_offsets, mask=kernel_offsets < kernel_size, other=0.0)
        
        # Compute convolution: sum over kernel dimension
        conv_result = tl.sum(x_block * tl.expand_dims(w_vals, 0), axis=1)
        
        # Add bias
        conv_result = conv_result + bias_val
        
        # Store output
        tl.store(out_ptr + output_offset + out_width_offsets, conv_result, mask=mask_out)
        
        out_width_start += BLOCK_SIZE_W


def triton_depthwise_conv1d(x, weight, bias, kernel_size, stride, padding, dilation):
    """
    Performs depthwise 1D convolution using Triton.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, 1, kernel_size)
        bias: Bias tensor of shape (in_channels,) or None
        kernel_size, stride, padding, dilation: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, in_channels, height, width_out)
    """
    n, c, h, w = x.shape
    kernel_size = weight.shape[2]
    
    # Calculate output width
    n_out = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((n, c, h, n_out), dtype=x.dtype, device=x.device)
    
    # Block sizes - tuned for FP32
    BLOCK_SIZE_W = 64
    BLOCK_SIZE_K = 16  # kernel_size is typically small, so this might be larger than needed
    
    # Grid: (batch_size, in_channels, height)
    grid = (n, c, h)
    
    # Launch kernel
    depthwise_conv1d_kernel[grid](
        x, weight, bias, out,
        n, c, h, w,
        kernel_size, stride, padding, dilation,
        n_out,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel using Triton.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        
        # Initialize weights for depthwise conv: (in_channels, 1, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias = self.bias if self.bias is not None else None
        
        # Call our Triton implementation
        return triton_depthwise_conv1d(x, weight, bias, 
                                       self.kernel_size, self.stride, 
                                       self.padding, self.dilation)