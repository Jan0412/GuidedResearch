import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    out_ptr, x_ptr, w_ptr, b_ptr,
    x_shape, w_shape, out_shape,
    stride, padding, dilation, groups, has_bias,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2) * BLOCK_H
    out_w_idx = tl.program_id(3) * BLOCK_W
    
    out_c = out_shape[1]
    out_h = out_shape[2]
    out_w = out_shape[3]
    
    in_ch = x_shape[1]
    in_h = x_shape[2]
    in_w = x_shape[3]
    
    k_h = w_shape[2]
    k_w = w_shape[3]
    
    for i in range(BLOCK_H):
        for j in range(BLOCK_W):
            o_h = out_h_idx + i
            o_w = out_w_idx + j
            
            if o_h >= out_h or o_w >= out_w:
                continue
                
            acc = 0.0
            
            for c in range(in_ch):
                for kh in range(k_h):
                    for kw in range(k_w):
                        i_h = o_h * stride - padding + kh * dilation
                        i_w = o_w * stride - padding + kw * dilation
                        
                        if i_h >= 0 and i_h < in_h and i_w >= 0 and i_w < in_w:
                            x_val = tl.load(x_ptr + batch_idx * in_ch * in_h * in_w + c * in_h * in_w + i_h * in_w + i_w)
                            w_val = tl.load(w_ptr + out_ch_idx * in_ch * k_h * k_w + c * k_h * k_w + kh * k_w + kw)
                            acc += x_val * w_val
            
            if has_bias:
                acc += tl.load(b_ptr + out_ch_idx)
                
            tl.store(out_ptr + batch_idx * out_c * out_h * out_w + out_ch_idx * out_h * out_w + o_h * out_w + o_w, acc)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, padding: int, dilation: int, groups: int, has_bias: bool):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    
    out = torch.empty_like(x)
    out[:, :w.shape[0], :, :] = 0  # Initialize output with zeros
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, k_h, k_w = w.shape
    
    out_height = (height + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    
    out[:, :out_channels, :out_height, :out_width] = 0
    
    BLOCK_H = 4
    BLOCK_W = 4
    
    grid = lambda meta: (
        batch_size,
        out_channels,
        (out_height + BLOCK_H - 1) // BLOCK_H,
        (out_width + BLOCK_W - 1) // BLOCK_W
    )
    
    x_shape = torch.tensor([batch_size, in_channels, height, width], device=x.device)
    w_shape = torch.tensor([out_channels, in_channels, k_h, k_w], device=w.device)
    out_shape = torch.tensor([batch_size, out_channels, out_height, out_width], device=out.device)
    
    conv2d_kernel[grid](
        out, x, w, b,
        x_shape, w_shape, out_shape,
        stride, padding, dilation, groups, has_bias,
        BLOCK_H, BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = torch.zeros(out_channels)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.has_bias)