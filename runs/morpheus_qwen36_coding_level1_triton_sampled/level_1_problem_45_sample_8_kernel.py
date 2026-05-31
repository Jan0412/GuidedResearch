import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    x_strides,
    out_strides,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
):
    # Each program handles one output element
    n, c, i, j = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    
    # Boundary check
    if n >= batch_size or c >= channels or i >= H_out or j >= W_out:
        return
    
    # Compute start indices for the pooling window in the input tensor
    h_start = i * stride + padding
    w_start = j * stride + padding
    
    # Compute base offset for the top-left element of the window
    base_offset = n * x_strides[0] + c * x_strides[1] + h_start * x_strides[2] + w_start * x_strides[3]
    
    # Accumulate sum over the pooling window
    total_sum = 0.0
    for di in range(kernel_size):
        for dj in range(kernel_size):
            offset = base_offset + di * x_strides[2] + dj * x_strides[3]
            val = tl.load(x_ptr + offset)
            total_sum += val
            
    # Compute output offset and store the average
    out_offset = n * out_strides[0] + c * out_strides[1] + i * out_strides[2] + j * out_strides[3]
    out_val = total_sum / (kernel_size * kernel_size)
    tl.store(out_ptr + out_offset, out_val)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Wrapper function to launch the Triton average pooling kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    # Compute output spatial dimensions
    H_out = (height + 2 * padding - kernel_size) // stride + 1
    W_out = (width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty((batch_size, channels, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Get strides for memory access optimization
    x_strides = x.stride()
    out_strides = out.stride()
    
    # Define grid: one program per output element
    grid = (batch_size, channels, H_out, W_out)
    
    # Launch kernel
    avg_pool_kernel[grid](
        x, out,
        x_strides, out_strides,
        batch_size, channels, H_out, W_out, kernel_size, stride, padding
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling using a custom Triton kernel.
        """
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)