import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,   # Pointer to input tensor (N, C, H, W)
    weight_ptr,  # Pointer to weight tensor (OC, IC, KH, KW)
    output_ptr,  # Pointer to output tensor (N, OC, OH, OW)
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    oc_idx = tl.program_id(1)
    oh_idx = tl.program_id(2)
    ow_idx = tl.program_id(3)
    
    # Calculate output dimensions
    output_size = output_height * output_width
    
    # Shared memory for tiles
    tile_input = tl.shared.tensor([TILE_H, TILE_W], tl.float32)
    tile_weight = tl.shared.tensor([TILE_C, TILE_H, TILE_W], tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over input channels
    for ic_start in range(0, in_channels, TILE_C):
        # Load weight tile
        ic_offset = ic_start + tl.arange(0, TILE_C)[:, None, None]
        kh_offset = tl.arange(0, TILE_H)[None, :, None]
        kw_offset = tl.arange(0, TILE_W)[None, None, :]
        
        # Bounds checking for weights
        weight_mask = (ic_offset < in_channels) & (kh_offset < kernel_height) & (kw_offset < kernel_width)
        
        # Load weights with proper indexing
        w = tl.load(weight_ptr + 
                   oc_idx * in_channels * kernel_height * kernel_width +
                   ic_offset * kernel_height * kernel_width +
                   kh_offset * kernel_width +
                   kw_offset, mask=weight_mask, other=0.0)
        
        # Calculate input indices
        ih_start = oh_idx * stride_h - pad_h
        iw_start = ow_idx * stride_w - pad_w
        
        # Load input tile
        ih = ih_start + kh_offset * dilation_h
        iw = iw_start + kw_offset * dilation_w
        
        # Bounds checking for input
        input_mask = (ih >= 0) & (ih < input_height) & (iw >= 0) & (iw < input_width)
        
        # Load input values
        input_vals = tl.load(input_ptr + 
                           batch_idx * in_channels * input_height * input_width +
                           ic_offset * input_height * input_width +
                           ih[:, None, :] * input_width +
                           iw[None, :, :], mask=input_mask & (ic_offset < in_channels), other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(w * input_vals, axis=0)
    
    # Write result
    if output_size > 0:
        tl.store(output_ptr + 
                batch_idx * out_channels * output_height * output_width +
                oc_idx * output_height * output_width +
                oh_idx * output_width +
                ow_idx, acc, mask=True)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device='cuda', dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 1024
    TILE_H = 8
    TILE_W = 8
    TILE_C = 16
    
    # Grid configuration
    grid = (
        batch_size,          # batch dimension
        out_channels,        # output channel dimension  
        output_height,       # output height dimension
        output_width         # output width dimension
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        TILE_C=TILE_C
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )