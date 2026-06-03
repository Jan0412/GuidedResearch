import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, K, K)
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (B, C, H_out, W_out)
    batch_size,  # B
    channels,  # C
    height,  # H
    width,  # W
    kernel_size,  # K
    stride,  # S
    padding,  # P
    out_height,  # H_out
    out_width,  # W_out
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    c_id = tl.program_id(0)  # Channel index
    bh_id = tl.program_id(1)  # Batch and height block index
    w_id = tl.program_id(2)  # Width block index
    
    # Calculate batch and height from bh_id
    b_id = bh_id // out_height
    h_out = bh_id % out_height
    
    # Calculate input height position
    h_in = h_out * stride - padding
    
    # Calculate width range for this block
    w_start = w_id * BLOCK_SIZE_W
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    w_mask = w_offsets < out_width
    
    # Calculate input width range
    w_in_start = w_start * stride - padding
    
    # Prepare output pointers
    out_ptr_base = out_ptr + b_id * channels * out_height * out_width + c_id * out_height * out_width
    out_offsets = h_out * out_width + w_offsets
    out_mask = w_mask
    
    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over kernel height
    for kh in range(kernel_size):
        h_k = h_in + kh
        # Check if kernel position is within input bounds
        if h_k >= 0 and h_k < height:
            # Calculate input pointer for this channel and kernel height
            x_ptr_base = x_ptr + b_id * channels * height * width + c_id * height * width
            x_offsets_h = h_k * width
            # Iterate over kernel width
            for kw in range(kernel_size):
                w_k = w_in_start + kw
                # Check if kernel position is within input bounds
                if w_k >= 0 and w_k < width:
                    # Load weights for this kernel position
                    w_offset = kh * kernel_size + kw
                    w_val = tl.load(w_ptr + c_id * kernel_size * kernel_size + w_offset)
                    
                    # Load input values
                    x_offsets = x_offsets_h + w_k
                    x_val = tl.load(x_ptr_base + x_offsets, mask=w_mask, other=0.0)
                    
                    # Accumulate convolution
                    acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_id)
        acc += bias
    
    # Store output
    tl.store(out_ptr_base + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, 1, kernel_size, kernel_size)
        bias: Optional bias tensor of shape (in_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        
    Returns:
        Output tensor of shape (batch_size, in_channels, height_out, width_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous memory
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, channels, height, width = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Define grid dimensions
    # We use channel as the first dimension for parallelism
    grid_c = channels
    grid_bh = batch_size * out_height  # Combined batch and height dimension
    grid_w = (out_width + 127) // 128  # Width blocks with 128 width block size
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_C = 1  # Process one channel per program
    BLOCK_SIZE_H = 1  # Process one height position per program
    BLOCK_SIZE_W = 128  # Process 128 width positions per program
    BLOCK_KH = 3  # Kernel height block size
    BLOCK_KW = 3  # Kernel width block size
    
    # Launch kernel
    depthwise_conv2d_kernel[grid_c, grid_bh, grid_w](
        x, weight, bias, out,
        batch_size, channels, height, width,
        kernel_size, stride, padding, out_height, out_width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights and bias similar to nn.Conv2d
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight parameter with proper initialization
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        # Initialize with kaiming uniform as in PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
            # Initialize bias
            fan_in = kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      self.stride, self.padding)


# Import math for initialization
import math