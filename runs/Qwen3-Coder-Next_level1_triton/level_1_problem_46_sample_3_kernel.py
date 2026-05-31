import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    B, C, D_in, H_in, W_in,  # Input dimensions
    D_out, H_out, W_out,  # Output dimensions
    kernel_d, kernel_h, kernel_w,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride dimensions
    pad_d, pad_h, pad_w,  # Padding dimensions
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get output coordinates
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute output position in D, H, W dimensions
    out_d_start = tl.program_id(2) * BLOCK_SIZE_D
    out_h_start = tl.program_id(3) * BLOCK_SIZE_H
    out_w = tl.program_id(4) * BLOCK_SIZE_W
    
    # Load output position for W dimension (single value per iteration for simplicity)
    out_w_idx = out_w + tl.arange(0, BLOCK_SIZE_W)
    mask_w = out_w_idx < W_out
    
    # Calculate the input region this thread block processes
    d_idx = out_d_start + tl.arange(0, BLOCK_SIZE_D)
    h_idx = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    
    # Create masks for valid indices
    mask_d = d_idx < D_out
    mask_h = h_idx < H_out
    mask = mask_d[:, None, None] & mask_h[:, None] & mask_w[None, None]
    
    # Initialize accumulator for average
    sum_val = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    count_val = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.int32)
    
    # Iterate over kernel window
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                in_d = d_idx * stride_d - pad_d + kd
                in_h = h_idx[:, None] * stride_h - pad_h + kh
                in_w = out_w_idx[None, None] * stride_w - pad_w + kw
                
                # Check if within input bounds
                valid_mask = (in_d >= 0) & (in_d < D_in) & \
                            (in_h >= 0) & (in_h < H_in) & \
                            (in_w >= 0) & (in_w < W_in)
                
                # Calculate linear offset for valid positions
                offsets = batch_idx * (C * D_in * H_in * W_in) + \
                         channel_idx * (D_in * H_in * W_in) + \
                         in_d[:, None, None] * (H_in * W_in) + \
                         in_h[:, None, None] * W_in + \
                         in_w
                
                # Load values and accumulate
                x_val = tl.load(x_ptr + offsets, mask=valid_mask & mask, other=0.0).to(tl.float32)
                sum_val += x_val
                count_val += valid_mask & mask
                
    # Compute average
    avg_val = sum_val / (count_val.to(tl.float32) + 1e-6)
    
    # Store result
    out_offsets = batch_idx * (C * D_out * H_out * W_out) + \
                 channel_idx * (D_out * H_out * W_out) + \
                 d_idx[:, None, None] * (H_out * W_out) + \
                 h_idx[:, None, None] * W_out + \
                 out_w_idx[None, None]
    
    tl.store(out_ptr + out_offsets, avg_val, mask=mask)


def triton_avg_pool3d(x, kernel_size, stride, padding):
    """
    Custom Triton implementation of 3D average pooling.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride for the pooling operation
        padding: Padding to apply before pooling
    
    Returns:
        Output tensor after applying 3D average pooling
    """
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    
    # Extract dimensions
    B, C, D_in, H_in, W_in = x.shape
    
    # Calculate output dimensions
    if isinstance(kernel_size, int):
        kernel_d = kernel_h = kernel_w = kernel_size
    else:
        kernel_d, kernel_h, kernel_w = kernel_size
        
    if stride is None:
        stride_d = stride_h = stride_w = kernel_d
    elif isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
        
    # Calculate output spatial dimensions
    D_out = (D_in + 2 * pad_d - kernel_d) // stride_d + 1
    H_out = (H_in + 2 * pad_h - kernel_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Create output tensor
    out = torch.empty((B, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set up kernel launch parameters
    # Use reasonable block sizes for the dimensions
    BLOCK_SIZE_D = min(4, D_out)
    BLOCK_SIZE_H = min(4, H_out)
    BLOCK_SIZE_W = min(8, W_out)
    
    # For kernel blocks, we can use the kernel dimensions
    BLOCK_KD = kernel_d
    BLOCK_KH = kernel_h
    BLOCK_KW = kernel_w
    
    # Grid dimensions: (batch, channel, D_blocks, H_blocks, W_blocks)
    grid = (B, C, (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D, 
            (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, 
            (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    avg_pool3d_kernel[grid](
        x, out,
        B, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernel for 3D Average Pooling.
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
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)