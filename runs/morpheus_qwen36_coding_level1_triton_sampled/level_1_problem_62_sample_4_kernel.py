import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    x_shape, w_shape, out_shape,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_H, BLOCK_SIZE_W,
    BLOCK_SIZE_K,
):
    # Grid dimensions
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute n and c_out from pid_z
    c_out = pid_z % w_shape[0]
    n = pid_z // w_shape[0]
    
    # Compute h_out and w_out
    h_out = pid_h
    w_out = pid_w
    
    # Check if within output bounds
    if h_out >= out_shape[2] or w_out >= out_shape[3]:
        return
    
    # Constants for kernel size
    k_h = w_shape[2]
    k_w = w_shape[3]
    c_in = w_shape[1]
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over channels in groups
    for c_in_base in range(0, c_in, BLOCK_SIZE_K):
        # Determine the range of c_in for this block
        c_in_offsets = c_in_base + tl.arange(0, BLOCK_SIZE_K)
        mask_c = c_in_offsets < c_in
        
        # Determine the group for this c_in
        group = c_in_base // (c_in // groups)
        
        # Check if c_out belongs to this group
        c_out_group = c_out // (w_shape[0] // groups)
        if group != c_out_group:
            continue
        
        # Compute input patch coordinates
        # h_in = h_out * stride_h - padding_h + dilation_h * k_h_offset
        # w_in = w_out * stride_w - padding_w + dilation_w * k_w_offset
        
        k_h_offsets = tl.arange(0, k_h)
        k_w_offsets = tl.arange(0, k_w)
        
        # Create 2D grid for kernel offsets
        k_h_grid, k_w_grid = tl.meshgrid(k_h_offsets, k_w_offsets)
        k_h_flat = k_h_grid.flatten()
        k_w_flat = k_w_grid.flatten()
        
        # Compute h_in and w_in for each kernel element
        h_in = h_out * stride_h - padding_h + dilation_h * k_h_flat
        w_in = w_out * stride_w - padding_w + dilation_w * k_w_flat
        
        # Check bounds for h_in and w_in
        mask_h = (h_in >= 0) & (h_in < x_shape[2])
        mask_w = (w_in >= 0) & (w_in < x_shape[3])
        mask_hw = mask_h & mask_w
        
        # Compute input offsets for this tile
        # Input shape: (n, c_in, h_in, w_in)
        # Offset = n * (c_in * h_in * w_in) + c_in * (h_in * w_in) + h_in * w_in + w_in
        n_stride = c_in * x_shape[2] * x_shape[3]
        c_in_stride = x_shape[2] * x_shape[3]
        h_stride = x_shape[3]
        
        x_offsets = n * n_stride + c_in_offsets[:, None] * c_in_stride + h_in[None, :] * h_stride + w_in[None, :]
        
        # Load input patch
        x_patch = tl.load(x_ptr + x_offsets, mask=mask_hw[None, :] & mask_c[:, None], other=0.0)
        
        # Load weights for this tile
        # Weights shape: (c_out, c_in, k_h, k_w)
        # Offset = c_out * (c_in * k_h * k_w) + c_in * (k_h * k_w) + k_h * k_w + k_w
        w_c_out_stride = c_in * k_h * k_w
        w_c_in_stride = k_h * k_w
        
        w_offsets = c_out * w_c_out_stride + c_in_offsets[:, None] * w_c_in_stride + k_h_flat[None, :] * k_w + k_w_flat[None, :]
        
        w_tile = tl.load(w_ptr + w_offsets, mask=mask_c[:, None], other=0.0)
        
        # Compute dot product and accumulate
        # x_patch: (BLOCK_SIZE_K, k_h * k_w)
        # w_tile: (BLOCK_SIZE_K, k_h * k_w)
        # We need to transpose w_tile to (k_h * k_w, BLOCK_SIZE_K) for tl.dot
        w_tile_T = tl.trans(w_tile)
        acc += tl.dot(x_patch, w_tile_T)
    
    # Add bias if present
    if b_ptr is not None:
        acc += tl.load(b_ptr + c_out)
    
    # Store result
    out_offset = n * out_shape[1] * out_shape[2] * out_shape[3] + c_out * out_shape[2] * out_shape[3] + h_out * out_shape[3] + w_out
    tl.store(out_ptr + out_offset, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Store parameters for the Triton kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weights and bias
        w = self.conv2d.weight  # Shape: (out_channels, in_channels, k_h, k_w)
        b = self.conv2d.bias if self.bias else None
        
        # Compute output shape
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        height, width = x.shape[2], x.shape[3]
        k_h, k_w = self.kernel_size
        
        height_out = (height + 2 * self.padding[0] if isinstance(self.padding, tuple) else 2 * self.padding - self.dilation * (k_h - 1) - 1) // self.stride + 1
        width_out = (width + 2 * self.padding[1] if isinstance(self.padding, tuple) else 2 * self.padding - self.dilation * (k_w - 1) - 1) // self.stride + 1
        
        # Ensure padding is tuple
        if isinstance(self.padding, int):
            padding_h, padding_w = self.padding, self.padding
        else:
            padding_h, padding_w = self.padding
            
        if isinstance(self.dilation, int):
            dilation_h, dilation_w = self.dilation, self.dilation
        else:
            dilation_h, dilation_w = self.dilation
            
        if isinstance(self.stride, int):
            stride_h, stride_w = self.stride, self.stride
        else:
            stride_h, stride_w = self.stride
            
        # Prepare output tensor
        out = torch.empty((batch_size, self.out_channels, height_out, width_out), dtype=x.dtype, device=x.device)
        
        # Define block sizes
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        BLOCK_SIZE_K = 16
        
        # Define grid
        grid = (batch_size * self.out_channels, height_out, width_out)
        
        # Launch kernel
        conv2d_kernel[grid](
            x, w, b, out,
            x.shape, w.shape, out.shape,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            self.groups,
            BLOCK_SIZE_H, BLOCK_SIZE_W,
            BLOCK_SIZE_K
        )
        
        return out