import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    B: tl.constexpr,  # Batch size
    C: tl.constexpr,  # Number of channels
    D_in: tl.constexpr,  # Input depth
    H_in: tl.constexpr,  # Input height
    W_in: tl.constexpr,  # Input width
    D_out: tl.constexpr,  # Output depth
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    K: tl.constexpr,  # Kernel size
    S: tl.constexpr,  # Stride
    P: tl.constexpr,  # Padding
    BLOCK_B: tl.constexpr = 1,
    BLOCK_C: tl.constexpr = 1,
    BLOCK_D: tl.constexpr = 4,
    BLOCK_H: tl.constexpr = 8,
    BLOCK_W: tl.constexpr = 8,
):
    # Compute output indices
    b = tl.program_id(0)
    c = tl.program_id(1)
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Calculate input region bounds
    d_start = d_out * S - P
    h_start = h_out * S - P
    w_start = w_out * S - P
    
    # Calculate kernel region bounds (clamped to input bounds)
    d0 = tl.maximum(d_start, 0)
    d1 = tl.minimum(d_start + K, D_in)
    h0 = tl.maximum(h_start, 0)
    h1 = tl.minimum(h_start + K, H_in)
    w0 = tl.maximum(w_start, 0)
    w1 = tl.minimum(w_start + K, W_in)
    
    # Compute actual kernel dimensions (handling edge cases near boundaries)
    kernel_d = d1 - d0
    kernel_h = h1 - h0
    kernel_w = w1 - w0
    
    # Compute sum
    sum_val = 0.0
    count = 0
    
    # Iterate over kernel region with optimized loops
    for kd in range(kernel_d):
        d = d0 + kd
        for kh in range(kernel_h):
            h = h0 + kh
                # Vectorized load for width dimension
                w = w0
                while w < w1:
                    # Compute input pointer offset
                    offset = ((b * C + c) * D_in * H_in * W_in + 
                             d * H_in * W_in + 
                             h * W_in + 
                             w)
                    # Load multiple elements at once (up to 8 elements)
                    remaining = w1 - w
                    n_load = tl.minimum(remaining, 8)
                    
                    # Create mask for valid indices
                    mask = tl.arange(0, 8) < n_load
                    
                    # Load values
                    x_vals = tl.load(x_ptr + offset + tl.arange(0, 8), mask=mask, other=0.0)
                    sum_val += tl.sum(x_vals)
                    count += n_load
                    
                    w += 8

    # Compute average and store result
    avg_val = sum_val / count if count > 0 else 0.0
    
    # Compute output pointer offset
    out_offset = ((b * C + c) * D_out * H_out * W_out + 
                 d_out * H_out * W_out + 
                 h_out * W_out + 
                 w_out)
    
    tl.store(out_ptr + out_offset, avg_val)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Applies 3D average pooling using custom Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling operation (default: kernel_size)
        padding: Padding to apply before pooling (default: 0)
        
    Returns:
        Output tensor after average pooling
    """
    # Ensure input is contiguous and on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Extract input dimensions
    B, C, D_in, H_in, W_in = x.shape
    
    # Set default stride if None
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    D_out = (D_in + 2 * padding - kernel_size) // stride + 1
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, C, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define grid dimensions
    # For large tensors, we use a 5D grid: (batch, channels, depth_out, height_out, width_out)
    # But we'll use a more efficient approach by processing multiple outputs per block
    
    # Use smaller block sizes for better occupancy
    BLOCK_B = 1
    BLOCK_C = 1
    BLOCK_D = 1
    BLOCK_H = 1
    BLOCK_W = 1
    
    # Grid dimensions
    grid = (B, C, D_out, H_out, W_out)
    
    # Launch kernel with fixed block sizes
    avg_pool3d_kernel[grid](
        x, out,
        B, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)