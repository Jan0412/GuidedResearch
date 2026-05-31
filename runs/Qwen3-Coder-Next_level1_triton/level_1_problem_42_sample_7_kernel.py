import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    y_ptr,  # Output tensor pointer (N, C, out_h, out_w)
    batch_size,  # N
    channels,    # C
    in_h,        # Input height
    in_w,        # Input width
    out_h,       # Output height
    out_w,       # Output width
    kernel_h,    # Kernel height
    kernel_w,    # Kernel width
    stride_h,    # Stride height
    stride_w,    # Stride width
    pad_h,       # Padding height
    pad_w,       # Padding width
    dil_h,       # Dilation height
    dil_w,       # Dilation width
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID encodes output spatial position and batch/channel
    # We'll use a 1D grid and compute 3D indices (batch, channel, output position)
    pid = tl.program_id(0)
    
    # Calculate batch, channel, and output position from pid
    # pid = batch * (channels * out_h * out_w) + channel * (out_h * out_w) + out_h_idx * out_w + out_w_idx
    total_out_elements = batch_size * channels * out_h * out_w
    if pid >= total_out_elements:
        return
    
    # Decode the 3D indices
    temp = pid
    out_w_idx = temp % out_w
    temp //= out_w
    out_h_idx = temp % out_h
    temp //= out_h
    channel_idx = temp % channels
    batch_idx = temp // channels
    
    # Calculate the top-left corner of the pooling window in input space
    # With padding and dilation, the actual positions are:
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Initialize max value to -inf
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate actual input position with dilation
            h_pos = in_h_start + kh * dil_h
            w_pos = in_w_start + kw * dil_w
            
            # Check if position is within valid input bounds
            if (h_pos >= 0 and h_pos < in_h and 
                w_pos >= 0 and w_pos < in_w):
                # Calculate input index
                input_offset = (batch_idx * channels * in_h * in_w + 
                               channel_idx * in_h * in_w + 
                               h_pos * in_w + w_pos)
                val = tl.load(x_ptr + input_offset)
                max_val = tl.maximum(max_val, val)
    
    # Store result
    y_offset = pid
    tl.store(y_ptr + y_offset, max_val)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Triton implementation of MaxPool2d.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_elements = batch_size * channels * out_h * out_w
    
    # Grid size
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out, batch_size, channels, in_h, in_w, out_h, out_w,
        kernel_size, kernel_size, stride, stride, padding, padding,
        dilation, dilation, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

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