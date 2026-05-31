import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    out_depth,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    groups,
    x_strides,
    w_strides,
    out_strides,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping
    b = tl.program_id(0)
    c_out = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Calculate output spatial coordinates
    hw_size = out_height * out_width
    d_prime = spatial_idx // hw_size
    rem = spatial_idx % hw_size
    h_prime = rem // out_width
    w_prime = rem % out_width
    
    # Check if spatial index is within bounds
    total_spatial = out_depth * out_height * out_width
    if spatial_idx >= total_spatial:
        return
    
    # Calculate group and input channel offset
    channels_per_group = in_channels // groups
    group = c_out // channels_per_group
    c_out_in_group = c_out % channels_per_group
    
    # Weight pointer for this output channel and group
    # Weights shape: (out_channels, in_channels // groups, K, K, K)
    # We can flatten the last 3 dims for contiguous access
    # w_ptr is already contiguous in memory for the slice [c_out, :, :, :, :]
    # But we need to offset by group for input channels
    w_ptr_offset = c_out * w_strides[0] + group * channels_per_group * kernel_size * kernel_size * kernel_size
    w_ptr_group = w_ptr + w_ptr_offset
    
    # Load weights into shared memory
    # Weights size: channels_per_group * kernel_size^3
    weight_size = channels_per_group * kernel_size * kernel_size * kernel_size
    weights_shared = tl.empty(weight_size, dtype=tl.float32, address_space=tl.shared)
    
    # Load weights
    offsets_w = tl.arange(0, weight_size)
    tl.store(weights_shared + offsets_w, tl.load(w_ptr_group + offsets_w, mask=offsets_w < weight_size, other=0.0))
    
    # Accumulator for this output element
    acc = 0.0
    
    # Loop over kernel dimensions
    for k_d in tl.range(kernel_size):
        d = d_prime * stride - padding + k_d
        if d < 0 or d >= depth:
            continue
            
        for k_h in tl.range(kernel_size):
            h = h_prime * stride - padding + k_h
            if h < 0 or h >= height:
                continue
                
            for k_w in tl.range(kernel_size):
                w = w_prime * stride - padding + k_w
                if w < 0 or w >= width:
                    continue
                
                # Compute input index
                # Input shape: (batch, in_channels, depth, height, width)
                # We need to access all input channels in this group
                # Input strides: (x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4])
                # Base index for this spatial location
                base_idx = b * x_strides[0] + d * x_strides[2] + h * x_strides[3] + w * x_strides[4]
                
                # Load input values for all channels in group
                # We can vectorize this if possible, but loop is safe
                for c_in_local in tl.range(channels_per_group):
                    c_in = group * channels_per_group + c_in_local
                    input_idx = base_idx + c_in * x_strides[1]
                    input_val = tl.load(x_ptr + input_idx)
                    
                    # Weight index
                    # Weight layout: (out_channels, in_channels//groups, K, K, K)
                    # We are in group, so c_in_local is the index within group
                    # Weight index: c_in_local * K^3 + k_d * K^2 + k_h * K + k_w
                    w_idx = c_in_local * kernel_size * kernel_size * kernel_size + k_d * kernel_size * kernel_size + k_h * kernel_size + k_w
                    weight_val = tl.load(weights_shared + w_idx)
                    
                    acc += input_val * weight_val
    
    # Store output
    out_idx = b * out_strides[0] + c_out * out_strides[1] + d_prime * out_strides[2] + h_prime * out_strides[3] + w_prime * out_strides[4]
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "FP32 precision required."
    
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = w.shape
    
    # Calculate output dimensions
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size
    
    out = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), dtype=torch.float32, device=x.device)
    
    # Strides
    x_strides = list(x.stride())
    w_strides = list(w.stride())
    out_strides = list(out.stride())
    
    # Grid configuration
    TILE_SIZE = 64
    num_spatial_tiles = (out_depth * out_height * out_width + TILE_SIZE - 1) // TILE_SIZE
    grid = (batch_size, out_channels, num_spatial_tiles)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding, groups,
        x_strides, w_strides, out_strides,
        TILE_SIZE=TILE_SIZE,
        BLOCK_SIZE=128,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights manually to ensure they are on CUDA and contiguous
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        w = self.weight.contiguous()
        
        out = triton_conv_transpose3d(x, w, self.stride, self.padding, self.groups)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
            
        return out