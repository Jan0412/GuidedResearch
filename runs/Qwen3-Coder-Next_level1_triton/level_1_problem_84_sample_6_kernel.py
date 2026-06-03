import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (N, C, H, W)
    w_ptr,  # Weight tensor: (C, 1, K, K)
    b_ptr,  # Bias tensor: (C,) or None
    out_ptr,  # Output tensor: (N, C, H_out, W_out)
    batch_size,  # N
    in_channels,  # C
    height_in,  # H
    width_in,  # W
    kernel_size,  # K
    stride,  # stride
    padding,  # padding
    height_out,  # H_out
    width_out,  # W_out
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    KERNEL_HALF: tl.constexpr,
):
    # Program IDs
    c_id = tl.program_id(0)  # Channel block ID
    h_id = tl.program_id(1)  # Height block ID
    w_id = tl.program_id(2)  # Width block ID
    
    # Compute actual channel range
    c_start = c_id * BLOCK_SIZE_C
    c_end = tl.minimum(c_start + BLOCK_SIZE_C, in_channels)
    
    # Compute output spatial coordinates
    h_start = h_id * BLOCK_SIZE_H
    w_start = w_id * BLOCK_SIZE_W
    
    # Check bounds
    if c_start >= in_channels:
        return
    
    # Create channel offsets [c_start, c_start+1, ..., c_end-1]
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < in_channels
    
    # Process each output position in the block
    for h in range(BLOCK_SIZE_H):
        # Compute input height coordinate
        h_in = h_start + h * stride - padding
        if h_in < 0 or h_in >= height_in:
            continue  # Outside input bounds
            
        for w in range(BLOCK_SIZE_W):
            # Compute input width coordinate
            w_in = w_start + w * stride - padding
            if w_in < 0 or w_in >= width_in:
                continue  # Outside input bounds
            
            # Compute output indices
            out_n = tl.program_id(3)  # Batch index (4th dimension)
            out_h = h_start + h
            out_w = w_start + w
            
            # Only process if within output bounds
            if out_h >= height_out or out_w >= width_out:
                continue
                
            # Accumulate convolution result
            acc = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
            
            # Loop over kernel dimensions
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Compute input position
                    h_in_k = h_in + kh
                    w_in_k = w_in + kw
                    
                    # Check bounds
                    if h_in_k >= 0 and h_in_k < height_in and w_in_k >= 0 and w_in_k < width_in:
                        # Compute input pointer offset: N * (C*H*W) + c * (H*W) + h_in_k * W + w_in_k
                        input_offset = (
                            out_n * (in_channels * height_in * width_in) +
                            c_offsets * (height_in * width_in) +
                            h_in_k * width_in +
                            w_in_k
                        )
                        
                        # Compute weight pointer offset: c * (K*K) + kh * K + kw
                        weight_offset = (
                            c_offsets * (kernel_size * kernel_size) +
                            kh * kernel_size +
                            kw
                        )
                        
                        # Load input and weight values
                        x_val = tl.load(x_ptr + input_offset, mask=c_mask, other=0.0)
                        w_val = tl.load(w_ptr + weight_offset, mask=c_mask, other=0.0)
                        
                        # Accumulate
                        acc += x_val * w_val
            
            # Add bias if present
            if b_ptr is not None:
                bias_val = tl.load(b_ptr + c_offsets, mask=c_mask, other=0.0)
                acc += bias_val
            
            # Store result
            # Compute output pointer offset: out_n * (C*H_out*W_out) + c * (H_out*W_out) + out_h * W_out + out_w
            out_offset = (
                out_n * (in_channels * height_out * width_out) +
                c_offsets * (height_out * width_out) +
                out_h * width_out +
                out_w
            )
            tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=c_mask)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (C, 1, K, K)
        bias: Optional bias tensor of shape (C,)
        stride: Stride of convolution
        padding: Padding applied to input
        
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    # Kernel parameters
    kernel_half = kernel_size // 2
    BLOCK_SIZE_C = 16  # Channels per block
    BLOCK_SIZE_H = 4   # Height block size
    BLOCK_SIZE_W = 16  # Width block size
    
    # Grid dimensions: (num_channel_blocks, num_height_blocks, num_width_blocks, batch_size)
    grid = (
        (in_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
        (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        batch_size
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height_in, width_in,
        kernel_size, stride, padding, height_out, width_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        KERNEL_HALF=kernel_half,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    Optimized with Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for inference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create depthwise convolution weight
        # Note: out_channels must equal in_channels for depthwise convolution
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding
        )