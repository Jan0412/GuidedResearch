import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in, H, W, D,
    C_out, K,
    stride_h, stride_w, stride_d,
    pad_h, pad_w, pad_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = B * C_out * H * W * D
    
    base_pid = pid * BLOCK_SIZE
    if base_pid >= num_elements:
        return
        
    offsets = base_pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Vectorized coordinate calculation
    b = offsets // (C_out * H * W * D)
    rem = offsets % (C_out * H * W * D)
    c_out = rem // (H * W * D)
    rem = rem % (H * W * D)
    h = rem // (W * D)
    rem = rem % (W * D)
    w = rem // D
    d = rem % D
    
    in_h = h * stride_h - pad_h
    in_w = w * stride_w - pad_w
    in_d = d * stride_d - pad_d
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c_in in range(C_in):
        for kh in range(K):
            for kw in range(K):
                idx_x = b * C_in * H * W * D + c_in * H * W * D + (in_h + kh) * W * D + (in_w + kw) * D + in_d
                idx_w = c_out * C_in * K * K + c_in * K * K + kh * K + kw
                
                valid_mask = (in_h + kh >= 0) & (in_h + kh < H) & (in_w + kw >= 0) & (in_w + kw < W) & (in_d >= 0) & (in_d < D)
                
                x_val = tl.load(x_ptr + idx_x, mask=mask & valid_mask, other=0.0)
                w_val = tl.load(w_ptr + idx_w, mask=mask, other=0.0)
                acc += x_val * w_val
                
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, kernel_size: int) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    B, C_in, H, W, D = x.shape
    C_out, _, K, _, _ = w.shape
    
    out = torch.empty((B, C_out, H, W, D), dtype=x.dtype, device=x.device)
    
    num_elements = B * C_out * H * W * D
    BLOCK_SIZE = 128
    
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    
    conv3d_kernel[grid](
        x, w, out,
        B, C_in, H, W, D,
        C_out, K,
        stride, stride, 1,
        padding, padding, 0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_conv3d(x, self.weight, self.stride, self.padding, self.kernel_size)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    width = 256
    height = 256
    depth = 10
    x = torch.rand(batch_size, in_channels, height, width, depth).cuda()
    return [x]

def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]