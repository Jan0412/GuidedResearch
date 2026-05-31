import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    width,
    height,
    depth,
    kernel_width,
    kernel_height,
    kernel_depth,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global output index
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * depth * height * width
    
    # Map linear index to output coordinates
    n = pid // (out_channels * depth * height * width)
    rem = pid % (out_channels * depth * height * width)
    oc = rem // (depth * height * width)
    rem = rem % (depth * height * width)
    od = rem // (height * width)
    rem = rem % (height * width)
    oh = rem // width
    ow = rem % width
    
    # Calculate input coordinates
    ih_base = oh * stride - padding
    iw_base = ow * stride - padding
    id_base = od * stride - padding
    
    # Determine group for this output channel
    group = oc // (out_channels // groups)
    in_group_start = group * (in_channels // groups)
    in_group_end = in_group_start + (in_channels // groups)
    
    # Accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_group_start, in_group_end):
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates with dilation
                    ih = ih_base + kh * dilation
                    iw = iw_base + kw * dilation
                    id_ = id_base + kd * dilation
                    
                    # Check bounds and apply padding logic
                    if 0 <= id_ < depth and 0 <= ih < height and 0 <= iw < width:
                        # Load input and weight
                        # x layout: (N, C, D, H, W)
                        x_idx = ((n * in_channels + ic) * depth + id_) * height * width + ih * width + iw
                        # w layout: (OutC, InC, Kd, Kh, Kw)
                        w_idx = ((oc * in_channels + ic) * kernel_depth + kd) * kernel_height * kernel_width + kh * kernel_width + kw
                        
                        x_val = tl.load(x_ptr + x_idx)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
    
    # Store result
    out_idx = ((n * out_channels + oc) * depth + od) * height * width + oh * width + ow
    tl.store(out_ptr + out_idx, acc)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    d_out = (depth + 2 * padding - dilation * (kernel_depth - 1) - 1) // stride + 1
    h_out = (height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    w_out = (width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    out = torch.empty(batch_size, out_channels, d_out, h_out, w_out, dtype=x.dtype, device=x.device)
    
    n_elements = batch_size * out_channels * d_out * h_out * w_out
    BLOCK_SIZE = 128
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    conv3d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        width, height, depth,
        kernel_width, kernel_height, kernel_depth,
        stride, padding, dilation, groups,
        BLOCK_SIZE
    )
    
    if bias is not None:
        out = out + bias.view(1, out_channels, 1, 1, 1)
        
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )