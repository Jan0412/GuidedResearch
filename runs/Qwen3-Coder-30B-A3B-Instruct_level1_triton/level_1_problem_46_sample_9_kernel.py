import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate global thread index
    idx = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total number of output elements
    total_output_elements = batch_size * channels * output_depth * output_height * output_width
    
    # Check bounds
    mask = idx < total_output_elements
    
    # Calculate which output element this thread handles
    if not mask[0]:
        return
        
    # Decompose linear index into 5D coordinates
    temp_idx = idx[0] if mask[0] else 0
    out_w = temp_idx % output_width
    temp_idx //= output_width
    out_h = temp_idx % output_height
    temp_idx //= output_height
    out_d = temp_idx % output_depth
    temp_idx //= output_depth
    ch = temp_idx % channels
    batch = temp_idx // channels
    
    # Calculate input start positions
    input_d_start = out_d * stride - padding
    input_h_start = out_h * stride - padding
    input_w_start = out_w * stride - padding
    
    # Initialize sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_d = input_d_start + kd
                input_h = input_h_start + kh
                input_w = input_w_start + kw
                
                # Check if within bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (
                        batch * (channels * input_depth * input_height * input_width) +
                        ch * (input_depth * input_height * input_width) +
                        input_d * (input_height * input_width) +
                        input_h * input_width +
                        input_w
                    )
                    
                    # Load value and accumulate
                    val = tl.load(input_ptr + input_idx, mask=True)
                    sum_val += val
                    count += 1
    
    # Compute average
    if count > 0:
        avg = sum_val / count
    else:
        avg = 0.0
    
    # Calculate output index
    out_idx = (
        batch * (channels * output_depth * output_height * output_width) +
        ch * (output_depth * output_height * output_width) +
        out_d * (output_height * output_width) +
        out_h * output_width +
        out_w
    )
    
    # Store result
    tl.store(output_ptr + out_idx, avg, mask=True)

def triton_avg_pool3d(input_tensor, kernel_size, stride, padding):
    """
    Triton implementation of 3D Average Pooling
    """
    # Ensure input is on GPU
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, channels, depth, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_depth, output_height, output_width, 
                         dtype=torch.float32, device=input_tensor.device)
    
    # Flatten tensors for easier indexing
    input_flat = input_tensor.view(-1)
    output_flat = output.view(-1)
    
    # Set up kernel launch parameters
    total_elements = output.numel()
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    avg_pool3d_kernel[grid_size](
        input_flat,
        output_flat,
        batch_size,
        channels,
        depth,
        height,
        width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 3D Average Pooling.
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
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)