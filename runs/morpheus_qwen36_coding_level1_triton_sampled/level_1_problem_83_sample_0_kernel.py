import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    height,
    width,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    dilation,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_bcw = batch_size * in_channels * width_out
    bcw = pid // height_out
    h_out = pid % height_out

    b = bcw // (in_channels * width_out)
    c = (bcw // width_out) % in_channels
    w_out = bcw % width_out

    w_in = w_out * stride - padding
    w_mask = (w_in >= 0) & (w_in < width)

    base_x = x_ptr + b * in_channels * height * width + c * height * width + w_in
    base_out = out_ptr + b * in_channels * height_out * width_out + c * height_out * width_out + w_out

    result = 0.0

    if w_mask:
        base_w = weight_ptr + c * kernel_size
        if has_bias:
            bias_val = tl.load(bias_ptr + c)
        else:
            bias_val = 0.0

        for k in range(kernel_size):
            h_in = h_out * stride - padding + k * dilation
            h_mask = (h_in >= 0) & (h_in < height)
            mask = h_mask & w_mask
            x_val = tl.load(base_x + h_in * width, mask=mask, other=0.0)
            w_k = tl.load(base_w + k)
            result += w_k * x_val

        result += bias_val
        tl.store(base_out + h_out * width_out, result, mask=True)


def triton_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                          stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    kernel_size = weight.shape[2]
    
    height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    width_out = width + 2 * padding - 1
    
    out = torch.empty((batch_size, in_channels, height_out, width_out), device=x.device, dtype=x.dtype)
    
    has_bias = bias is not None
    bias_ptr = bias if has_bias else None
    
    grid = (batch_size * in_channels * width_out * height_out,)
    
    depthwise_conv_kernel[grid](
        x, out, weight, bias_ptr,
        batch_size, in_channels, height, width,
        height_out, width_out,
        kernel_size, stride, padding, dilation,
        has_bias,
        BLOCK_SIZE=128
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias if self.conv2d.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )