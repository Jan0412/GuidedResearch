import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,
    out_ptr,
    B, C, D_in, H_in, W_in,
    D_out, H_out, W_out,
    kernel_size, stride, padding,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute output indices
    bd = tl.program_id(0)
    bh = tl.program_id(1)
    bw = tl.program_id(2)
    bc = tl.program_id(3)
    
    # Compute input indices for the pooling region
    d_in_start = bd * stride - padding
    h_in_start = bh * stride - padding
    w_in_start = bw * stride - padding
    
    # Initialize accumulator for average
    acc = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    count = 0
    
    # Iterate over the pooling kernel region
    for kd in range(kernel_size):
        d_in = d_in_start + kd
        d_mask = (d_in >= 0) & (d_in < D_in)
        
        for kh in range(kernel_size):
            h_in = h_in_start + kh
            h_mask = (h_in >= 0) & (h_in < H_in)
            
            for kw in range(kernel_size):
                w_in = w_in_start + kw
                w_mask = (w_in >= 0) & (w_in < W_out)
                
                # Check if this position is valid
                valid_mask = d_mask & h_mask & w_mask
                
                if tl.sum(valid_mask) > 0:
                    # Compute input pointer offset
                    offset = (
                        bc * (D_in * H_in * W_in) +
                        d_in * (H_in * W_in) +
                        h_in * W_in +
                        w_in
                    )
                    
                    # Load input values
                    x_val = tl.load(x_ptr + offset, mask=valid_mask, other=0.0)
                    
                    # Accumulate
                    acc += x_val.to(tl.float32)
                    count += 1
    
    # Compute average
    if count > 0:
        avg = acc / count
    else:
        avg = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    
    # Store output
    out_offset = (
        bc * (D_out * H_out * W_out) +
        bd * (H_out * W_out) +
        bh * W_out +
        bw
    )
    
    tl.store(out_ptr + out_offset, avg)


def triton_avg_pool3d(x, kernel_size, stride, padding):
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling operation
        padding: Padding applied before pooling
    
    Returns:
        Output tensor after 3D average pooling
    """
    # Ensure input is contiguous and on CUDA
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Get input dimensions
    B, C, D_in, H_in, W_in = x.shape
    
    # Calculate output dimensions
    D_out = (D_in + 2 * padding - kernel_size) // stride + 1
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty((B, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_D = 1
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 1
    
    # Define grid dimensions
    grid = (D_out, H_out, W_out, C)
    
    # Launch kernel
    avg_pool3d_kernel[grid](
        x,
        out,
        B, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with optimized Triton implementation.

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
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)