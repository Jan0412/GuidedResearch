import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    y_ptr,  # Output tensor pointer (N, C, H_out, W_out)
    n_elements,  # Total number of output elements
    # Strides for each dimension
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    # Dimensions
    batch_size, channels, in_height, in_width,
    out_height, out_width,
    kernel_h, kernel_w,
    stride_h_pool, stride_w_pool,
    pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID corresponds to output element index
    pid = tl.program_id(0)
    
    # Calculate output indices (n, c, h_out, w_out)
    # Flatten output tensor for easier indexing
    temp = pid
    w_out = temp % out_width
    temp //= out_width
    h_out = temp % out_height
    temp //= out_height
    c = temp % channels
    n = temp // channels
    
    # Calculate the top-left corner of the pooling window in input
    h_start = h_out * stride_h_pool - pad_h
    w_start = w_out * stride_w_pool - pad_w
    
    # Calculate the actual pooling window boundaries (clamped to input bounds)
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end_clamped = tl.minimum(h_start + kernel_h, in_height)
    w_end_clamped = tl.minimum(w_start + kernel_w, in_width)
    
    # Calculate actual pooling region dimensions (handling padding)
    pool_h = h_end_clamped - h_start_clamped
    pool_w = w_end_clamped - w_start_clamped
    
    # Total number of valid elements in the pooling region
    pool_size = pool_h * pool_w
    
    # Compute the average
    sum_val = 0.0
    for h in range(h_start_clamped, h_end_clamped):
        for w in range(w_start_clamped, w_end_clamped):
            # Calculate input pointer offset
            offset = (n * stride_n + 
                     c * stride_c + 
                     h * stride_h + 
                     w * stride_w)
            x_val = tl.load(x_ptr + offset)
            sum_val += x_val
    
    # Avoid division by zero (shouldn't happen with valid inputs, but safe)
    if pool_size > 0:
        avg_val = sum_val / pool_size
    else:
        avg_val = 0.0
    
    # Store result
    out_offset = (n * out_stride_n + 
                 c * out_stride_c + 
                 h_out * out_stride_h + 
                 w_out * out_stride_w)
    tl.store(y_ptr + out_offset, avg_val)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Triton implementation of 2D average pooling.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation (defaults to kernel_size)
        padding: Padding applied to the input tensor
        
    Returns:
        Output tensor after average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, in_height, in_width = x.shape
    
    # Calculate output dimensions
    if stride is None:
        stride = kernel_size
    
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_n = x.stride(0)
    stride_c = x.stride(1)
    stride_h = x.stride(2)
    stride_w = x.stride(3)
    
    out_stride_n = out.stride(0)
    out_stride_c = out.stride(1)
    out_stride_h = out.stride(2)
    out_stride_w = out.stride(3)
    
    # Total number of output elements
    n_elements = batch_size * channels * out_height * out_width
    
    # Triton kernel parameters
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per output element (could be optimized with tiling)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out, n_elements,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        batch_size, channels, in_height, in_width,
        out_height, out_width,
        kernel_size, kernel_size,  # kernel_h, kernel_w
        stride, stride,            # stride_h_pool, stride_w_pool
        padding, padding           # pad_h, pad_w
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with Triton optimization.
        
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
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)