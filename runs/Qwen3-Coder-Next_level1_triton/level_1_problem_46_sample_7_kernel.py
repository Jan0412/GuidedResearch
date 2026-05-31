import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool3d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    n_batches,  # Number of batches
    n_channels,  # Number of channels
    input_depth, input_height, input_width,  # Input dimensions
    output_depth, output_height, output_width,  # Output dimensions
    kernel_size, stride, padding,  # Pooling parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_y = tl.program_id(3)
    pid_x = tl.program_id(4)
    
    # Early exit if out of bounds
    if pid_batch >= n_batches or pid_channel >= n_channels or \
       pid_z >= output_depth or pid_y >= output_height or pid_x >= output_width:
        return
    
    # Calculate input region for this output position
    input_z_start = pid_z * stride - padding
    input_y_start = pid_y * stride - padding
    input_x_start = pid_x * stride - padding
    
    # Calculate kernel boundaries
    kernel_z_end = min(input_z_start + kernel_size, input_depth)
    kernel_y_end = min(input_y_start + kernel_height, input_height)
    kernel_x_end = min(input_x_start + kernel_width, input_width)
    
    input_z_start = max(0, input_z_start)
    input_y_start = max(0, input_y_start)
    input_x_start = max(0, input_x_start)
    
    # Compute valid region dimensions
    kernel_z_size = kernel_z_end - input_z_start
    kernel_y_size = kernel_y_end - input_y_start
    kernel_x_size = kernel_x_end - input_x_start
    
    # Calculate total valid elements for averaging
    valid_count = kernel_z_size * kernel_y_size * kernel_x_size
    
    # Compute pointer offset for this batch and channel
    x_offset = (pid_batch * n_channels * input_depth * input_height * input_width +
                pid_channel * input_depth * input_height * input_width)
    
    # Accumulate sum over the pooling window
    sum_val = 0.0
    for dz in range(kernel_z_size):
        z_idx = input_z_start + dz
        for dy in range(kernel_y_size):
            y_idx = input_y_start + dy
            for dx in range(kernel_x_size):
                x_idx = input_x_start + dx
                offset = x_offset + z_idx * input_height * input_width + y_idx * input_width + x_idx
                sum_val += tl.load(x_ptr + offset)
    
    # Compute average and store
    avg = sum_val / valid_count
    tl.store(y_ptr + pid_batch * n_channels * output_depth * output_height * output_width +
             pid_channel * output_depth * output_height * output_width +
             pid_z * output_height * output_width +
             pid_y * output_width +
             pid_x, avg)


# Alternative optimized version with better memory access patterns
@triton.jit
def avgpool3d_kernel_optimized(
    x_ptr,  # Input tensor pointer (BCDHW format)
    y_ptr,  # Output tensor pointer
    n_batches, n_channels,
    input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global indices for output
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Check bounds
    if pid_b >= n_batches or pid_c >= n_channels or \
       pid_z >= output_depth or pid_h >= output_height or pid_w >= output_width:
        return
    
    # Calculate input region
    z_start = pid_z * stride - padding
    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding
    
    # Calculate valid pooling region
    z_start_clamped = tl.maximum(z_start, 0)
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    
    z_end = tl.minimum(z_start + kernel_size, input_depth)
    h_end = tl.minimum(h_start + kernel_size, input_height)
    w_end = tl.minimum(w_start + kernel_size, input_width)
    
    # Calculate kernel dimensions
    k_z = z_end - z_start_clamped
    k_h = h_end - h_start_clamped
    k_w = w_end - w_start_clamped
    
    # Total valid elements
    valid_count = k_z * k_h * k_w
    
    # Compute base offset for this batch and channel
    base_offset = (pid_b * n_channels * input_depth * input_height * input_width +
                   pid_c * input_depth * input_height * input_width)
    
    # Accumulate sum
    sum_val = 0.0
    for dz in range(k_z):
        z_idx = z_start_clamped + dz
        for dh in range(k_h):
            h_idx = h_start_clamped + dh
            for dw in range(k_w):
                w_idx = w_start_clamped + dw
                offset = base_offset + z_idx * input_height * input_width + h_idx * input_width + w_idx
                sum_val += tl.load(x_ptr + offset)
    
    # Store average
    output_offset = (pid_b * n_channels * output_depth * output_height * output_width +
                     pid_c * output_depth * output_height * output_width +
                     pid_z * output_height * output_width +
                     pid_h * output_width +
                     pid_w)
    tl.store(y_ptr + output_offset, sum_val / valid_count)


def triton_avgpool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of pooling kernel
        stride: Stride of pooling operation
        padding: Padding applied before pooling
    
    Returns:
        Output tensor after 3D average pooling
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Get input dimensions
    batch_size, channels, depth, height, width = x.shape
    
    # Calculate output dimensions
    out_depth = (depth + 2 * padding - kernel_size) // stride + 1
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, channels, out_depth, out_height, out_width, 
                     dtype=x.dtype, device=x.device)
    
    # Define grid dimensions
    grid = (batch_size, channels, out_depth, out_height, out_width)
    
    # Launch kernel
    avgpool3d_kernel_optimized[grid](
        x, out,
        batch_size, channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        BLOCK_SIZE=32
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
            stride (int, optional): Stride of the pooling operation. Defaults to kernel_size if None.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)