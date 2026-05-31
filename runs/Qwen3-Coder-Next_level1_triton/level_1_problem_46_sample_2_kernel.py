import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # Input tensor pointer (N, C, D, H, W)
    out_ptr,  # Output tensor pointer
    N, C, D_in, H_in, W_in,  # Input dimensions
    D_out, H_out, W_out,  # Output dimensions
    k_d, k_h, k_w,  # Kernel dimensions
    s_d, s_h, s_w,  # Strides
    pad_d, pad_h, pad_w,  # Padding
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs for output dimensions
    bid = tl.program_id(0)  # Batch index
    cid = tl.program_id(1)  # Channel index
    od = tl.program_id(2)  # Output depth index
    oh = tl.program_id(3)  # Output height index
    ow = tl.program_id(4)  # Output width index
    
    # Calculate input starting positions for this output position
    id_start = od * s_d - pad_d
    ih_start = oh * s_h - pad_h
    iw_start = ow * s_w - pad_w
    
    # Accumulator for the sum
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    count = 0
    
    # Loop over kernel dimensions
    for kd in range(k_d):
        id = id_start + kd
        # Check if depth index is valid
        depth_valid = (id >= 0) & (id < D_in)
        
        for kh in range(k_h):
            ih = ih_start + kh
            # Check if height index is valid
            height_valid = (ih >= 0) & (ih < H_in)
            
            for kw in range(k_w):
                iw = iw_start + kw
                # Check if width index is valid
                width_valid = (iw >= 0) & (iw < W_in)
                
                # Combined valid mask
                valid = depth_valid & height_valid & width_valid
                
                # Calculate input pointer offset
                # Input layout: N, C, D, H, W
                offset = ((bid * C * D_in * H_in * W_in) + 
                         (cid * D_in * H_in * W_in) + 
                         (id * H_in * W_in) + 
                         (ih * W_in) + 
                         iw)
                
                # Load input values (supporting vectorized load for multiple channels)
                mask = tl.full([BLOCK_C], valid, dtype=tl.int1)
                x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
                
                # Accumulate
                acc += x_val.to(tl.float32)
                count += valid
    
    # Compute average
    if count > 0:
        avg = acc / tl.full([BLOCK_C], count, dtype=tl.float32)
    else:
        avg = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    # Write output
    out_offset = ((bid * C * D_out * H_out * W_out) + 
                 (cid * D_out * H_out * W_out) + 
                 (od * H_out * W_out) + 
                 (oh * W_out) + 
                 ow)
    
    # Store result (broadcast to all channels in block if needed)
    if BLOCK_C == 1:
        tl.store(out_ptr + out_offset, avg[0])
    else:
        tl.store(out_ptr + out_offset, avg[0])


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply before pooling
    
    Returns:
        Output tensor after 3D average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Parse parameters
    N, C, D_in, H_in, W_in = x.shape
    
    if stride is None:
        stride = kernel_size
        
    # Calculate output dimensions
    D_out = (D_in + 2 * padding - kernel_size) // stride + 1
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(N, C, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set kernel parameters
    k_d = k_h = k_w = kernel_size
    s_d = s_h = s_w = stride
    pad_d = pad_h = pad_w = padding
    
    # Define block sizes (tunable)
    BLOCK_D = 1
    BLOCK_H = 1
    BLOCK_W = 1
    BLOCK_C = 1  # We process one channel at a time for simplicity
    
    # Grid dimensions: (batch, channel, output_depth, output_height, output_width)
    grid = (N, C, D_out, H_out, W_out)
    
    # Launch kernel
    avg_pool3d_kernel[grid](
        x, out,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        pad_d, pad_h, pad_w,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using Triton kernel.
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
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)