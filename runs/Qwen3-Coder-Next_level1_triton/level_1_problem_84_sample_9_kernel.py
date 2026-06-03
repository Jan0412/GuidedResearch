import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Weight tensor (C, 1, K, K)
    b_ptr,  # Bias tensor (C,) or None
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    batch_size, in_channels, height_in, width_in,
    height_out, width_out,
    kernel_size, stride, padding,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index
    
    # Calculate output position
    h_out = pid_h
    w_out = pid_w
    
    # Calculate input position (top-left corner of the kernel)
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel height
    for kh in range(kernel_size):
        h = h_in + kh
        # Check if kernel position is within input bounds
        h_valid = (h >= 0) & (h < height_in)
        
        # Loop over kernel width
        for kw in range(kernel_size):
            w = w_in + kw
            # Check if kernel position is within input bounds
            w_valid = (w >= 0) & (w < width_in)
            
            # Get input offsets
            offsets_h = tl.arange(0, BLOCK_SIZE_H)
            offsets_w = tl.arange(0, BLOCK_SIZE_W)
            h_offsets = h * width_in + offsets_w
            w_offsets = tl.arange(0, BLOCK_SIZE_W)
            
            # Calculate input pointer offset
            input_offset = pid_b * (in_channels * height_in * width_in) + \
                          pid_c * (height_in * width_in) + \
                          h * width_in + w_in
            
            # Load input values with masking
            x_offsets = input_offset + offsets_h[:, None] * width_in + offsets_w[None, :]
            mask = (h_offsets[:, None] < height_in) & (offsets_w[None, :] < width_in)
            
            # Only load if valid
            x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            
            # Load kernel weight
            w_offset = pid_c * (kernel_size * kernel_size) + kh * kernel_size + kw
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_c
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store output
    out_offset = pid_b * (in_channels * height_out * width_out) + \
                pid_c * (height_out * width_out) + \
                h_out * width_out + w_out
    out_offsets = out_offset + tl.arange(0, BLOCK_SIZE_H)[:, None] * width_out + tl.arange(0, BLOCK_SIZE_W)[None, :]
    mask_out = (tl.arange(0, BLOCK_SIZE_H)[:, None] < height_out) & (tl.arange(0, BLOCK_SIZE_W)[None, :] < width_out)
    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height_in, width_in = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, height_out, width_out, dtype=x.dtype, device=x.device)
    
    # Determine block sizes for optimal performance
    BLOCK_SIZE_H = min(8, height_out)
    BLOCK_SIZE_W = min(8, width_out)
    BLOCK_SIZE_K = kernel_size
    
    # Grid dimensions: (batch, channel, output_height, output_width)
    grid = (batch_size, in_channels, 
            triton.cdiv(height_out, BLOCK_SIZE_H), 
            triton.cdiv(width_out, BLOCK_SIZE_W))
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height_in, width_in,
        height_out, width_out,
        kernel_size, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (depthwise convolution: groups=in_channels)
        # For depthwise: weight shape is (out_channels, 1, kernel_size, kernel_size)
        # But we need to handle the case where out_channels might differ from in_channels
        # However, the original implementation uses groups=in_channels, so out_channels must be multiple of in_channels
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)