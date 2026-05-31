import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer (batch, channels, height, width)
    out_ptr,  # Output tensor pointer
    batch_size, channels, 
    in_h, in_w,
    out_h, out_w,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate output spatial position
    out_h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = tl.arange(0, BLOCK_SIZE_W)
    out_w_idx = tl.program_id(3) * BLOCK_SIZE_W + out_w
    
    # Compute input position for each output position
    h_start = out_h_idx * stride - padding
    w_start = out_w_idx * stride - padding
    
    # Compute max over the kernel window
    max_val = tl.full([BLOCK_SIZE_H, BLOCK_SIZE_W], -float('inf'), dtype=tl.float32)
    
    # Iterate over kernel height
    for kh in range(kernel_size):
        h_idx = h_start + kh * dilation
        # Check bounds for height
        h_valid = (h_idx >= 0) & (h_idx < in_h)
        
        # Iterate over kernel width
        for kw in range(kernel_size):
            w_idx = w_start + kw * dilation
            # Check bounds for width
            w_valid = (w_idx >= 0) & (w_idx < in_w)
            valid = h_valid & w_valid
            
            # Calculate input pointer offset for this position
            offset = pid_b * (channels * in_h * in_w) + pid_c * (in_h * in_w) + h_idx[:, None] * in_w + w_idx[None, :]
            
            # Load values (masked to avoid out-of-bounds memory access)
            val = tl.load(x_ptr + offset, mask=valid, other=-float('inf'))
            
            # Update max
            max_val = tl.maximum(max_val, val)
    
    # Store result
    out_offset = pid_b * (channels * out_h * out_w) + pid_c * (out_h * out_w) + out_h_idx[:, None] * out_w + out_w_idx[None, :]
    tl.store(out_ptr + out_offset, max_val, mask=(out_h_idx[:, None] < out_h) & (out_w_idx[None, :] < out_w))


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Triton implementation of MaxPool2d.
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure grid for kernel launch
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid = (
        batch_size,  # batch dimension
        channels,    # channel dimension
        (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # height blocks
        (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W   # width blocks
    )
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels,
        in_h, in_w,
        out_h, out_w,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
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
        Applies optimized Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)