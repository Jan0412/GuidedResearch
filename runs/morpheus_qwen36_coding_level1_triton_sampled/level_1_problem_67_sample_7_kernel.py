import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    stride, padding, dilation, groups,
    batch, in_c, out_c, k_size, length, length_out,
    has_bias,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    block_l_start = tl.program_id(2) * BLOCK_SIZE
    
    l_offsets = block_l_start + tl.arange(0, BLOCK_SIZE)
    mask_l = l_offsets < length_out
    
    group = out_c_idx // (out_c // groups)
    in_c_per_group = in_c // groups
    in_c_start = group * in_c_per_group
    in_c_end = in_c_start + in_c_per_group
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c_in in tl.static_range(in_c_start, in_c_end):
        for k in tl.static_range(k_size):
            x_idx = batch_idx * in_c * length + c_in * length + l_offsets + k * stride - padding
            mask_x = (l_offsets + k * stride - padding) >= 0
            x_val = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
            
            w_idx = out_c_idx * in_c * k_size + c_in * k_size + k
            w_val = tl.load(w_ptr + w_idx)
            
            acc += x_val * w_val
            
    if has_bias:
        bias = tl.load(bias_ptr + out_c_idx)
        acc += bias
        
    out_idx = batch_idx * out_c * length_out + out_c_idx * length_out + l_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_l)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, 
                  stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    batch, in_c, length = x.shape
    out_c, _, k_size = w.shape
    length_out = (length + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch, out_c, length_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    grid = (batch, out_c, (length_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    has_bias = bias is not None
    
    conv1d_kernel[grid](
        x, w, out, bias,
        stride, padding, dilation, groups,
        batch, in_c, out_c, k_size, length, length_out,
        has_bias,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(
            x, 
            self.conv1d.weight, 
            self.conv1d.bias, 
            self.conv1d.stride[0], 
            self.conv1d.padding[0], 
            self.conv1d.dilation[0], 
            self.conv1d.groups
        )