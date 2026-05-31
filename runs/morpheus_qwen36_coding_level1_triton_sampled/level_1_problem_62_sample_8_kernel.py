import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    in_channels, out_channels, height, width,
    kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Grid dimensions: (batch_size * out_channels, out_height * out_width)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    b = pid0 // out_channels
    oc = pid0 % out_channels
    
    oh = pid1 // width
    ow = pid1 % width
    
    # Groups logic
    num_groups = groups
    out_channels_per_group = out_channels // num_groups
    in_channels_per_group = in_channels // num_groups
    group_idx = oc // out_channels_per_group
    ic_start = group_idx * in_channels_per_group
    ic_end = ic_start + in_channels_per_group
    
    acc = 0.0
    
    # Loop over input channels in blocks
    for ic_block_start in range(ic_start, ic_end, BLOCK_SIZE_IC):
        ic_offsets = ic_block_start + tl.arange(0, BLOCK_SIZE_IC)
        mask_ic = ic_offsets < in_channels
        
        # Load weight block: shape (BLOCK_SIZE_IC, kernel_h, kernel_w)
        w_offsets = oc * in_channels_per_group * kernel_h * kernel_w + ic_offsets[:, None, None] * kernel_h * kernel_w + tl.arange(0, kernel_h)[:, None] * kernel_w + tl.arange(0, kernel_w)[None, :]
        w = tl.load(w_ptr + w_offsets, mask=mask_ic[:, None, None], other=0.0)
        
        # Loop over kernel spatial dimensions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                ih = oh * stride_h + kh * dilation_h
                iw = ow * stride_w + kw * dilation_w
                
                mask_spatial = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                
                # Load input block: shape (BLOCK_SIZE_IC,)
                x_offsets = b * in_channels * height * width + ic_offsets * height * width + ih * width + iw
                x = tl.load(x_ptr + x_offsets, mask=mask_ic & mask_spatial, other=0.0)
                
                acc += tl.sum(x[:, None, None] * w, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
    
    # Store result
    out_offset = b * out_channels * height * width + oc * height * width + oh * width + ow
    tl.store(out_ptr + out_offset, acc)


def triton_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int,
    bias: bool
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    in_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
    out_channels = w.shape[0]
    kernel_h, kernel_w = w.shape[2], w.shape[3]
    batch_size = x.shape[0]
    
    out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_IC = 16
    
    grid = (batch_size * out_channels, out_height * out_width)
    
    conv2d_kernel[grid](
        x_ptr=x, w_ptr=w, b_ptr=b if bias else None, out_ptr=out,
        in_channels=in_channels, out_channels=out_channels,
        height=height, width=width,
        kernel_h=kernel_h, kernel_w=kernel_w,
        stride_h=stride_h, stride_w=stride_w,
        padding_h=padding_h, padding_w=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding_h, padding_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        dilation_h, dilation_w = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        
        return triton_conv2d(
            x, self.weight, self.bias if self.bias is not None else None,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, self.groups, self.bias is not None
        )