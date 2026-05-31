import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 256
BLOCK_K = 128

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    in_channels, out_channels, kH, kW,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    height, width,
    out_H, out_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = m_offsets < out_channels
    mask_n = n_offsets < (out_H * out_W)
    
    batch = n_offsets // (out_H * out_W)
    spatial_idx = n_offsets % (out_H * out_W)
    out_h = spatial_idx // out_W
    out_w = spatial_idx % out_W
    
    ih = out_h * stride_h - padding_h
    iw = out_w * stride_w - padding_w
    
    x_b_stride = in_channels * height * width
    x_c_stride = height * width
    x_h_stride = width
    
    x_b_ptr = x_ptr + batch[:, None] * x_b_stride
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    K = in_channels * kH * kW
    
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K
        
        w_ptr_offsets = w_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        w = tl.load(w_ptr_offsets, mask=mask_k[None, :], other=0.0)
        
        c = k_offsets // (kH * kW)
        kh_rem = k_offsets % (kH * kW)
        kh = kh_rem // kW
        kw = kh_rem % kW
        
        ih_global = ih + kh * dilation_h
        iw_global = iw + kw * dilation_w
        
        mask_x = (ih_global >= 0) & (ih_global < height) & (iw_global >= 0) & (iw_global < width)
        mask_x = mask_x & mask_k[None, :]
        
        x_ptr_offsets = c[None, :] * x_c_stride + ih_global[:, None] * x_h_stride + iw_global[:, None]
        
        x = tl.load(x_b_ptr[:, None] + x_ptr_offsets, mask=mask_x, other=0.0)
        
        acc += tl.dot(w, x, transpose_b=True, allow_tf32=False)
        
    out_ch_stride = out_H * out_W
    out_h_stride = out_W
    
    out_ptr_offsets = batch[:, None] * out_ch_stride + m_offsets[:, None] * out_h_stride + spatial_idx[None, :]
    
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_ptr_offsets, acc, mask=mask_out)

def triton_conv2d(x, w, in_channels, out_channels, kernel_size, stride, padding, dilation):
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    
    in_channels = in_channels
    out_channels = out_channels
    kH, kW = kernel_size
    stride_h, stride_w = stride, stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    height, width = x.shape[2], x.shape[3]
    batch_size = x.shape[0]
    
    out_H = (height + 2 * padding_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    out_W = (width + 2 * padding_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    
    out = torch.empty((batch_size, out_channels, out_H, out_W), dtype=x.dtype, device=x.device)
    
    N = batch_size * out_H * out_W
    K = in_channels * kH * kW
    
    grid_m = (out_channels + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    conv2d_kernel[(grid_m, grid_n)](
        x, w, out,
        in_channels, out_channels, kH, kW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        height, width,
        out_H, out_W,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_conv2d(x, self.weight, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out = out + self.bias[:, None, None]
        return out