import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    in_height,  # Input height
    in_width,   # Input width
    out_height,  # Output height
    out_width,   # Output width
    kernel_size,  # Kernel size
    stride,       # Stride
    padding,      # Padding
    dilation,     # Dilation
    BLOCK_SIZE: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    # Compute channel block index
    c_block_idx = tl.program_id(1)
    # Compute output height index
    oh_idx = tl.program_id(2)
    # Compute output width index
    ow_idx = tl.program_id(3)
    
    # Calculate actual channel index from block
    c_start = c_block_idx * C_BLOCK
    c_indices = c_start + tl.arange(0, C_BLOCK)
    mask_channels = c_indices < channels
    
    # Calculate input starting positions for this output position
    h_start = oh_idx * stride - padding
    w_start = ow_idx * stride - padding
    
    # Initialize output with -inf for max operation
    max_vals = tl.zeros([C_BLOCK], dtype=tl.float32) - float("inf")
    
    # Iterate over the kernel window
    for kh in range(kernel_size):
        h = h_start + kh * dilation
        # Check if h is within input bounds
        h_valid = (h >= 0) & (h < in_height)
        
        for kw in range(kernel_size):
            w = w_start + kw * dilation
            # Check if w is within input bounds
            w_valid = (w >= 0) & (w < in_width)
            valid = h_valid & w_valid
            
            if valid:
                # Calculate input pointer offset
                input_offset = batch_idx * channels * in_height * in_width + \
                              c_indices * in_height * in_width + \
                              h * in_width + w
                
                # Load values (with masking for channels outside range)
                x_vals = tl.load(x_ptr + input_offset, mask=mask_channels, other=-float("inf"))
                
                # Update max values
                max_vals = tl.maximum(max_vals, x_vals)
    
    # Calculate output pointer offset
    output_offset = batch_idx * channels * out_height * out_width + \
                   c_indices * out_height * out_width + \
                   oh_idx * out_width + ow_idx
    
    # Store results
    tl.store(y_ptr + output_offset, max_vals, mask=mask_channels)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    """
    Applies 2D max pooling using a Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size, stride, padding, dilation: Pooling parameters
    
    Returns:
        Output tensor after max pooling
    """
    batch_size, channels, in_height, in_width = x.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Configure kernel parameters
    BLOCK_SIZE = 128  # Not used directly but kept for structure
    C_BLOCK = min(8, channels)  # Process multiple channels per block
    
    # Grid dimensions: (batch_size, num_channel_blocks, out_height, out_width)
    grid = (
        batch_size,
        (channels + C_BLOCK - 1) // C_BLOCK,
        out_height,
        out_width,
    )
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels, in_height, in_width,
        out_height, out_width,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        C_BLOCK=C_BLOCK,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the optimized Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)