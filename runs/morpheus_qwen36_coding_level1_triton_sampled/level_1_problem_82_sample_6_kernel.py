import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_c, w_stride_kh, w_stride_kw,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    num_channels, height_out, width_out, height, width, kernel_size, stride, padding, has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    if pid_c >= num_channels:
        return
        
    out_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    h_mask = out_h < height_out
    w_mask = out_w < width_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    kh_offsets = tl.arange(0, kernel_size)
    kw_offsets = tl.arange(0, kernel_size)
    
    w_ptr_c = w_ptr + pid_c * w_stride_c
    w = tl.load(w_ptr_c + kh_offsets[:, None] * w_stride_kh + kw_offsets[None, :] * w_stride_kw)
    
    in_h = out_h[:, None] * stride + kh_offsets[None, :] - padding
    in_w = out_w[None, :] * stride + kw_offsets[:, None] - padding
    
    in_mask = (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
    
    x_ptr_c = x_ptr + pid_c * x_stride_c
    x_tile = tl.load(x_ptr_c + in_h * x_stride_h + in_w * x_stride_w, mask=in_mask, other=0.0)
    
    acc = tl.sum(x_tile[:, :, None] * w[None, :, :], axis=1)
    acc = tl.sum(acc, axis=1)
    
    if has_bias:
        acc += tl.load(bias_ptr + pid_c)
        
    tl.store(out_ptr + pid_c * out_stride_c + out_h * out_stride_h + out_w * out_stride_w, acc, mask=mask)


def triton_depthwise_conv2d(x, w, bias=None, stride=1, padding=0):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "FP32 required."
    
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    batch_size, in_channels, height, width = x.shape
    kernel_size = w.shape[2]
    
    height_out = (height + 2 * padding - kernel_size) // stride + 1
    width_out = (width + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((batch_size, in_channels, height_out, width_out), dtype=torch.float32, device=x.device)
    
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_c, w_stride_kh, w_stride_kw = w.stride()[0], w.stride()[2], w.stride()[3]
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()
    
    has_bias = bias is not None
    
    BLOCK_SIZE = 16
    grid = (in_channels, 
            (height_out + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (width_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
            
    depthwise_conv2d_kernel[grid](
        x, w, out, bias,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_c, w_stride_kh, w_stride_kw,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        in_channels, height_out, width_out, height, width, kernel_size, stride, padding, has_bias,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(in_channels, dtype=torch.float32))
        else:
            self.bias_param = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias_param if self.bias else None,
            stride=self.stride,
            padding=self.padding
        )