import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, K, K)
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (N, C, H_out, W_out)
    n, c, h, w,  # Input dimensions
    h_out, w_out,  # Output dimensions
    kernel_size, stride, padding,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for batch, channel, and spatial dimensions
    batch_id = tl.program_id(0)
    c_id = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Compute starting positions for the block
    start_h = block_h * BLOCK_SIZE_H
    start_w = block_w * BLOCK_SIZE_W
    
    # Compute the range of output positions this block will process
    h_offsets = start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = start_w + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output positions
    h_mask = h_offsets < h_out
    w_mask = w_mask = w_offsets < w_out
    
    # Calculate the corresponding input positions (accounting for padding and stride)
    # For depthwise conv, each output position corresponds to a kernel-sized region in input
    h_kernel_start = start_h * stride - padding
    w_kernel_start = start_w * stride - padding
    
    # Initialize accumulator for this block
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over the kernel dimensions
    for kh in range(kernel_size):
        h_in = h_kernel_start + kh
        h_in_mask = (h_in >= 0) & (h_in < h)
        
        for kw in range(kernel_size):
            w_in = w_kernel_start + kw
            w_in_mask = (w_in >= 0) & (w_in < w)
            
            # Load weight for this kernel position
            w_offset = kh * kernel_size + kw
            weight = tl.load(w_ptr + c_id * kernel_size * kernel_size + w_offset)
            
            # Load input values for this kernel position
            # We need to broadcast across batch dimension for the same channel
            # Since we're processing per-channel in the c_id dimension, we need to handle batch separately
            
            # For each output position in the block
            for bh in range(BLOCK_SIZE_H):
                h_pos = h_offsets[bh]
                h_in = h_kernel_start + kh
                h_in_mask_bh = (h_in >= 0) & (h_in < h) & (h_pos < h_out)
                
                if h_in_mask_bh:
                    for bw in range(BLOCK_SIZE_W):
                        w_pos = w_offsets[bw]
                        w_in = w_kernel_start + kw
                        w_in_mask_bw = (w_in >= 0) & (w_in < w) & (w_pos < w_out)
                        
                        if w_in_mask_bw:
                            # Calculate input index
                            input_idx = batch_id * (c * h * w) + c_id * (h * w) + h_in * w + w_in
                            x_val = tl.load(x_ptr + input_idx)
                            
                            # Accumulate
                            output = tl.where(
                                (h_pos < h_out) & (w_pos < w_out),
                                output + x_val * weight,
                                output
                            )
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_id)
        output = output + bias
    
    # Store results
    for bh in range(BLOCK_SIZE_H):
        h_pos = h_offsets[bh]
        for bw in range(BLOCK_SIZE_W):
            w_pos = w_offsets[bw]
            out_idx = batch_id * (c * h_out * w_out) + c_id * (h_out * w_out) + h_pos * w_out + w_pos
            tl.store(out_ptr + out_idx, output[bh, bw], mask=(h_pos < h_out) & (w_pos < w_out))


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
        Output tensor of shape (batch_size, in_channels, output_height, output_width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Determine block sizes
    BLOCK_SIZE_N = 1  # batch dimension
    BLOCK_SIZE_C = min(32, in_channels)  # channel dimension
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        in_channels,  # channel dimension  
        (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # height blocks
        (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,   # width blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        out_height, out_width,
        kernel_size, stride, padding,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    Uses optimized Triton kernel instead of PyTorch's native implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # For depthwise convolution, groups=in_channels, so each input channel has its own filter
        # The weight shape should be (out_channels, 1, kernel_size, kernel_size) for depthwise
        # But since we're doing depthwise with in_channels == out_channels in the example,
        # we'll keep the standard PyTorch convention where weight has shape (in_channels, 1, k, k)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )


import math