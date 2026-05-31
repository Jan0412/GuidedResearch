import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    height_out,
    width_out,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    """
    Triton kernel for 2D convolution.
    Processes a tile of the output tensor.
    """
    # Program ID mapping
    # We use a 1D grid where each program handles a tile of the output
    pid = tl.program_id(0)
    
    # Decompose pid into batch, output channel, and spatial tile indices
    # Total tiles per batch per channel
    tiles_per_spatial = ((height_out + BLOCK_H - 1) // BLOCK_H) * ((width_out + BLOCK_W - 1) // BLOCK_W)
    
    batch_idx = pid // (out_channels * tiles_per_spatial)
    oc_idx = (pid // tiles_per_spatial) % out_channels
    spatial_tile_idx = pid % tiles_per_spatial
    
    # Spatial tile coordinates
    oh_start = (spatial_tile_idx // ((width_out + BLOCK_W - 1) // BLOCK_W)) * BLOCK_H
    ow_start = (spatial_tile_idx % ((width_out + BLOCK_W - 1) // BLOCK_W)) * BLOCK_W
    
    # Create offsets for the output tile
    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)
    
    # Mask for valid output elements
    oh_mask = oh_offsets < height_out
    ow_mask = ow_offsets < width_out
    tile_mask = oh_mask[:, None] & ow_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Determine input channel range based on groups
    ic_start = oc_idx * (in_channels // groups)
    ic_end = ic_start + (in_channels // groups)
    
    # Loop over input channels in blocks
    for ic_block_start in range(ic_start, ic_end, BLOCK_IC):
        ic_block_end = min(ic_block_start + BLOCK_IC, ic_end)
        ic_offsets = ic_block_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offsets < ic_end
        
        # Load weights for the current IC block
        # Weights shape: (out_channels, in_channels // groups, kernel_height, kernel_width)
        # We need weights for current oc_idx and current ic block
        # Weight offset calculation:
        # w_ptr is contiguous in memory as (OC, IC/g, KH, KW)
        # Base offset for oc_idx: oc_idx * (in_channels // groups) * kernel_height * kernel_width
        # Base offset for ic_block_start: ic_block_start * kernel_height * kernel_width
        # We load a tile of weights: (BLOCK_IC, kernel_height, kernel_width)
        
        w_base = oc_idx * (in_channels // groups) * kernel_height * kernel_width + ic_block_start * kernel_height * kernel_width
        w_offsets = tl.arange(0, BLOCK_IC)[:, None, None] * kernel_height * kernel_width + \
                    tl.arange(0, kernel_height)[None, :, None] * kernel_width + \
                    tl.arange(0, kernel_width)[None, None, :]
        
        w_ptrs = w_ptr + w_base + w_offsets
        w = tl.load(w_ptrs, mask=ic_mask[:, None, None], other=0.0)
        
        # Load input patch for the current tile and IC block
        # Input shape: (batch, in_channels, height, width)
        # For each output element (oh, ow), we need input at (oh - pad, ow - pad) with dilation
        # Input offsets:
        # ih = oh - padding + kh * dilation
        # iw = ow - padding + kw * dilation
        
        # Calculate base input offset for the tile
        # We need to load a patch of shape (BLOCK_IC, kernel_height, kernel_width) for each (oh, ow)
        # This is complex to vectorize directly. 
        # Alternative: Load input in blocks of IC and iterate over kernel spatial dims.
        # Or load the whole patch for the tile and IC block.
        
        # Let's load input patch: (BLOCK_H, BLOCK_W, BLOCK_IC, kernel_height, kernel_width)
        # This might be too large. 
        # Better: Load input patch for current IC block and tile.
        # Shape: (BLOCK_H, BLOCK_W, BLOCK_IC, kernel_height, kernel_width)
        # We can compute indices dynamically.
        
        # Offsets for input
        # ih = oh_offsets[:, None, None] - padding_height + tl.arange(0, kernel_height)[None, None, :] * dilation_height
        # iw = ow_offsets[None, :, None] - padding_width + tl.arange(0, kernel_width)[None, None, :] * dilation_width
        # ic = ic_offsets[None, None, :, None]
        
        # This creates a 5D tensor of indices. 
        # To optimize, we can loop over kernel spatial dims or use tl.load with broadcasting.
        
        # Simplified approach: Loop over kernel spatial dims inside the IC block loop?
        # Or load input patch in a tiled manner.
        
        # Let's use a loop over kernel spatial dimensions to manage memory better.
        # This is often more efficient for small kernels.
        
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates for this kernel element
                ih = oh_offsets[:, None] - padding_height + kh * dilation_height
                iw = ow_offsets[None, :] - padding_width + kw * dilation_width
                
                # Mask for valid input coordinates
                ih_mask = (ih >= 0) & (ih < height)
                iw_mask = (iw >= 0) & (iw < width)
                patch_mask = ih_mask[:, None] & iw_mask[None, :]
                
                # Load input patch: (BLOCK_H, BLOCK_W, BLOCK_IC)
                # We need to gather from input tensor.
                # Input is (batch, IC, H, W).
                # We are processing batch_idx.
                # Input base offset for batch: batch_idx * in_channels * height * width
                # Input offsets:
                # batch: batch_idx * in_channels * height * width
                # ic: ic_offsets[None, None, :] * height * width
                # h: ih[:, None, None] * width
                # w: iw[None, :, None]
                
                input_base = batch_idx * in_channels * height * width
                input_offsets = input_base + \
                                ic_offsets[None, None, :] * height * width + \
                                ih[:, None, None] * width + \
                                iw[None, :, None]
                
                x_ptrs = x_ptr + input_offsets
                x = tl.load(x_ptrs, mask=patch_mask[:, :, None], other=0.0)
                
                # Multiply and accumulate
                # w shape: (BLOCK_IC, 1, 1) after broadcasting over kh, kw? No, w was (BLOCK_IC, KH, KW)
                # We need w[:, kh, kw]
                w_kh_kw = w[:, kh, kw]
                acc += x * w_kh_kw[None, None, :]
    
    # Add bias
    if b_ptr is not None:
        acc += b_ptr[oc_idx]
    
    # Store result
    out_base = batch_idx * out_channels * height_out * width_out + oc_idx * height_out * width_out
    out_offsets = out_base + oh_offsets[:, None] * width_out + ow_offsets[None, :]
    out_ptrs = out_ptr + out_offsets
    
    tl.store(out_ptrs, acc, mask=tile_mask)


def triton_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1
) -> torch.Tensor:
    """
    Wrapper for the Triton Conv2D kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    
    # Calculate output dimensions
    height_out = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    width_out = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=x.dtype, device=x.device)
    
    # Tunable block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_IC = 16
    
    # Grid configuration
    tiles_per_spatial = ((height_out + BLOCK_H - 1) // BLOCK_H) * ((width_out + BLOCK_W - 1) // BLOCK_W)
    num_tiles = batch_size * out_channels * tiles_per_spatial
    
    grid = (num_tiles,)
    
    conv2d_kernel[grid](
        x, w, b, out,
        batch_size, in_channels, height, width,
        out_channels, kernel_height, kernel_width,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        height_out, width_out,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_IC=BLOCK_IC
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )