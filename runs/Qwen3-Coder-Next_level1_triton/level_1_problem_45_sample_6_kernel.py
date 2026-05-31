import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    in_h, in_w,  # Input height and width
    out_h, out_w,  # Output height and width
    kernel_h, kernel_w,  # Kernel height and width
    stride_h, stride_w,  # Stride height and width
    pad_h, pad_w,  # Padding height and width
    BLOCK_SIZE: tl.constexpr,
    TILE_C: tl.constexpr,
):
    # Get batch and channel indices
    bc_id = tl.program_id(0)
    c_id = bc_id % channels
    b_id = bc_id // channels
    
    # Output spatial indices
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate the top-left corner of the pooling window in input space
    h_start = out_h_idx * stride_h - pad_h
    w_start = out_w_idx * stride_w - pad_w
    
    # Accumulator for the sum
    sum_val = 0.0
    count = 0
    
    # Iterate over the pooling window
    for kh in range(kernel_h):
        h = h_start + kh
        h_valid = (h >= 0) & (h < in_h)
        for kw in range(kernel_w):
            w = w_start + kw
            w_valid = (w >= 0) & (w < in_w)
            valid = h_valid & w_valid
            
            if valid:
                # Calculate input index
                in_idx = b_id * (channels * in_h * in_w) + \
                         c_id * (in_h * in_w) + \
                         h * in_w + w
                # Load the value
                x_val = tl.load(x_ptr + in_idx)
                sum_val += x_val
                count += 1
    
    # Compute average (avoid division by zero)
    if count > 0:
        avg = sum_val / count
    else:
        avg = 0.0
    
    # Write output
    out_idx = b_id * (channels * out_h * out_w) + \
              c_id * (out_h * out_w) + \
              out_h_idx * out_w + out_w_idx
    tl.store(out_ptr + out_idx, avg)


class TritonAvgPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding):
        # Convert to tuples if needed
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
            
        if stride is None:
            stride_h = stride_w = kernel_h
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
            
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding
            
        # Extract dimensions
        batch_size, channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
        
        # Create output tensor
        out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        if out.numel() == 0:
            return out
            
        # Configure grid
        # Each block handles one (batch, channel) pair
        # Grid dimensions: [batch * channels, out_h, out_w]
        grid = (batch_size * channels, out_h, out_w)
        
        # Launch kernel
        avg_pool2d_kernel[grid](
            x, out,
            batch_size, channels,
            in_h, in_w, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            BLOCK_SIZE=1,  # Not used in this implementation but required
            TILE_C=1,      # Not used in this implementation but required
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For average pooling, gradient is distributed evenly across pooling window
        # This is a placeholder - in practice, you'd need to implement backward pass
        # But for simplicity, we'll return None for non-tensor args
        return grad_output, None, None, None


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return TritonAvgPool2d.apply(x, self.kernel_size, self.stride, self.padding)