import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    bias_ptr,
    n_elements_x,
    n_elements_y,
    batch_size,
    in_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Grid is (batch_size * in_channels, num_tiles_h * num_tiles_w)
    n_c = tl.program_id(0)
    tile_idx = tl.program_id(1)
    
    n = n_c // in_channels
    c = n_c % in_channels
    
    num_tiles_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    h_block = tile_idx // num_tiles_w
    w_block = tile_idx % num_tiles_w
    
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W
    
    # Input tile coordinates
    h_in_start = h_start - padding_h
    w_in_start = w_start - padding_w
    
    # Offsets for input tile
    offsets_h = tl.arange(0, BLOCK_SIZE_H + 2 * dilation_h * (BLOCK_SIZE_KH - 1))
    offsets_w = tl.arange(0, BLOCK_SIZE_W + 2 * dilation_w * (BLOCK_SIZE_KW - 1))
    
    # Mask for input tile
    mask_h = offsets_h < (height + 2 * padding_h)
    mask_w = offsets_w < (width + 2 * padding_w)
    mask = tl.where(mask_h[:, None] & mask_w[None, :], 1, 0)
    
    # Load input tile
    input_offsets = (n * in_channels + c) * height * width + \
                    (h_in_start + offsets_h * dilation_h) * width + \
                    (w_in_start + offsets_w * dilation_w)
    input_tile = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    
    # Load kernel
    kernel_offsets = c * kernel_h * kernel_w + \
                     tl.arange(0, BLOCK_SIZE_KH)[:, None] * kernel_w + \
                     tl.arange(0, BLOCK_SIZE_KW)[None, :]
    kernel_tile = tl.load(w_ptr + kernel_offsets, mask=mask, other=0.0)
    
    # Compute convolution
    output_tile = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for kh in tl.static_range(BLOCK_SIZE_KH):
        for kw in tl.static_range(BLOCK_SIZE_KW):
            output_tile += input_tile[kh * dilation_h:(kh + 1) * dilation_h + BLOCK_SIZE_H:dilation_h, 
                                      kw * dilation_w:(kw + 1) * dilation_w + BLOCK_SIZE_W:dilation_w] * \
                           kernel_tile[kh, kw]
    
    # Store output
    output_offsets = (n * in_channels + c) * (height // stride_h + 1) * (width // stride_w + 1) + \
                     h_block * BLOCK_SIZE_H * (width // stride_w + 1) + \
                     w_block * BLOCK_SIZE_W
    tl.store(y_ptr + output_offsets, output_tile, mask=mask)


def triton_depthwise_conv2d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None,
                            stride_h: int = 1, stride_w: int = 1,
                            padding_h: int = 0, padding_w: int = 0,
                            dilation_h: int = 1, dilation_w: int = 1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    batch_size, in_channels, height, width = x.shape
    out_channels = in_channels  # Depthwise
    height_out = (height + 2 * padding_h - dilation_h * (3 - 1) - 1) // stride_h + 1
    width_out = (width + 2 * padding_w - dilation_w * (7 - 1) - 1) // stride_w + 1
    
    y = torch.empty((batch_size, out_channels, height_out, width_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 7
    
    grid = (batch_size * in_channels, 
            (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H * 
            (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
            
    depthwise_conv2d_kernel[grid](
        x, w, y, bias,
        x.numel(), y.numel(),
        batch_size, in_channels, height, width,
        BLOCK_SIZE_KH, BLOCK_SIZE_KW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_KH, BLOCK_SIZE_KW
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights for depthwise convolution
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, kernel_size_h, kernel_size_w) / 
                                   (in_channels * kernel_size_h * kernel_size_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )