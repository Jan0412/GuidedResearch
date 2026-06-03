import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Depthwise filter (C, 1, KH, KW)
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    n_batches, n_channels, in_height, in_width,
    out_height, out_width,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_KH: tl.constexpr, BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs for output spatial dimensions
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Calculate output spatial position
    out_h_start = block_h * BLOCK_SIZE_H
    out_w_start = block_w * BLOCK_SIZE_W
    
    # Offset to the current batch and channel in input
    x_batch_offset = batch_id * n_channels * in_height * in_width
    x_channel_offset = channel_id * in_height * in_width
    
    # Offset to the current channel in weights
    w_channel_offset = channel_id * kernel_size * kernel_size
    
    # Output offset base
    out_batch_offset = batch_id * n_channels * out_height * out_width
    out_channel_offset = channel_id * out_height * out_width
    
    # Iterate over output height block
    for oh in range(BLOCK_SIZE_H):
        out_h = out_h_start + oh
        if out_h >= out_height:
            break
            
        # Calculate input height position
        in_h = out_h * stride - padding
        
        # Iterate over output width block
        for ow in range(BLOCK_SIZE_W):
            out_w = out_w_start + ow
            if out_w >= out_width:
                break
                
            # Calculate input width position
            in_w = out_w * stride - padding
            
            # Accumulator for the convolution
            acc = 0.0
            
            # Iterate over kernel
            for kh in range(kernel_size):
                in_kh = in_h + kh * dilation
                if in_kh < 0 or in_kh >= in_height:
                    continue
                    
                for kw in range(kernel_size):
                    in_kw = in_w + kw * dilation
                    if in_kw < 0 or in_kw >= in_width:
                        continue
                        
                    # Calculate indices
                    x_idx = x_batch_offset + x_channel_offset + in_kh * in_width + in_kw
                    w_idx = w_channel_offset + kh * kernel_size + kw
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    
                    # Accumulate
                    acc += x_val * w_val
            
            # Store result
            out_idx = out_batch_offset + out_channel_offset + out_h * out_width + out_w
            tl.store(out_ptr + out_idx, acc)

@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Pointwise filter (OC, IC, 1, 1)
    bias_ptr,  # Bias tensor (OC,) or None
    out_ptr,  # Output tensor (B, OC, H, W)
    n_batches, in_channels, out_channels, height, width,
    has_bias: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_OC: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Calculate output spatial position
    out_h_start = block_h * BLOCK_SIZE_H
    out_w_start = block_w * BLOCK_SIZE_W
    
    # Output batch offset
    out_batch_offset = batch_id * out_channels * height * width
    out_channel_offset = out_channel_id * height * width
    
    # Iterate over output height block
    for oh in range(BLOCK_SIZE_H):
        out_h = out_h_start + oh
        if out_h >= height:
            break
            
        # Iterate over output width block
        for ow in range(BLOCK_SIZE_W):
            out_w = out_w_start + ow
            if out_w >= width:
                break
                
            # Accumulator for pointwise convolution
            acc = 0.0
            
            # Iterate over input channels
            for ic in range(in_channels):
                # Input index
                x_idx = batch_id * in_channels * height * width + ic * height * width + out_h * width + out_w
                # Weight index (pointwise: just maps IC to OC)
                w_idx = out_channel_id * in_channels + ic
                
                # Load values
                x_val = tl.load(x_ptr + x_idx)
                w_val = tl.load(w_ptr + w_idx)
                
                # Accumulate
                acc += x_val * w_val
            
            # Add bias if available
            if has_bias:
                bias_val = tl.load(bias_ptr + out_channel_id)
                acc += bias_val
            
            # Store result
            out_idx = out_batch_offset + out_channel_offset + out_h * width + out_w
            tl.store(out_ptr + out_idx, acc)

def triton_depthwise_conv2d(x, weight, stride=1, padding=0, dilation=1):
    """Perform depthwise convolution using Triton kernel"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_height, in_width = x.shape
    _, _, kernel_size_h, kernel_size_w = weight.shape
    kernel_size = kernel_size_h  # Assume square kernel for simplicity
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 3
    
    grid = (batch_size, in_channels, 
            (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, in_height, in_width,
        out_height, out_width,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH, BLOCK_SIZE_KW=BLOCK_SIZE_KW
    )
    
    return out

def triton_pointwise_conv2d(x, weight, bias=None):
    """Perform pointwise convolution using Triton kernel"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height, width, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_OC = 16
    BLOCK_SIZE_IC = 16
    
    grid = (batch_size, out_channels,
            (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, bias if bias is not None else None,
        out, batch_size, in_channels, out_channels, height, width,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC, BLOCK_SIZE_IC=BLOCK_SIZE_IC
    )
    
    return out

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation with optimized Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution with Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Depthwise convolution
        x = triton_depthwise_conv2d(x, self.depthwise.weight, 
                                   stride=self.depthwise.stride[0],
                                   padding=self.depthwise.padding[0],
                                   dilation=self.depthwise.dilation[0])
        
        # Add depthwise bias if it exists
        if self.depthwise.bias is not None:
            x = x + self.depthwise.bias.view(1, -1, 1, 1)
        
        # Pointwise convolution
        x = triton_pointwise_conv2d(x, self.pointwise.weight, 
                                   bias=self.pointwise.bias if self.pointwise.bias is not None else None)
        
        return x