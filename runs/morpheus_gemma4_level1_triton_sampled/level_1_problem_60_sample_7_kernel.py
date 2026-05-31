import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, W, H, D,
    Cout, Kw, Kh, Kd,
    Wo, Ho, Do,
    stride, padding, dilation, groups,
    stride_x_b, stride_x_c, stride_x_w, stride_x_h, stride_x_d,
    stride_w_cout, stride_w_c, stride_w_kw, stride_w_kh, stride_w_kd,
    stride_out_b, stride_out_cout, stride_out_wo, stride_out_ho, stride_out_do,
):
    # Get output indices from program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_wo = tl.program_id(2)
    pid_ho = tl.program_id(3)
    pid_do = tl.program_id(4)

    acc = 0.0
    
    # Calculate the start of the input channel group for this output channel
    cin_start = (pid_cout // (Cout // groups)) * (Cin // groups)
    
    # Iterate over the input channel group and kernel spatial dimensions
    for c in range(Cin // groups):
        c_idx = cin_start + c
        for kw in range(Kw):
            for kh in range(Kh):
                for kd in range(Kd):
                    # Compute input spatial indices considering stride, padding, and dilation
                    iw = pid_wo * stride + kw * dilation - padding
                    ih = pid_ho * stride + kh * dilation - padding
                    id_ = pid_do * stride + kd * dilation - padding
                    
                    # Mask to handle padding (zero-padding)
                    mask = (iw >= 0) & (iw < W) & (ih >= 0) & (ih < H) & (id_ >= 0) & (id_ < D)
                    
                    # Calculate pointers
                    x_off = (pid_b * stride_x_b + 
                             c_idx * stride_x_c + 
                             iw * stride_x_w + 
                             ih * stride_x_h + 
                             id_ * stride_x_d)
                    
                    w_off = (pid_cout * stride_w_cout + 
                             c * stride_w_c + 
                             kw * stride_w_kw + 
                             kh * stride_w_kh + 
                             kd * stride_w_kd)
                    
                    # Safe load using mask for padding
                    # Clamp index to 0 to avoid illegal memory access when mask is False
                    safe_x_off = tl.where(mask, x_off, 0)
                    x_val = tl.load(x_ptr + safe_x_off, mask=mask, other=0.0)
                    w_val = tl.load(w_ptr + w_off)
                    
                    acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        acc += tl.load(b_ptr + pid_cout)
        
    # Compute output pointer and store result
    out_off = (pid_b * stride_out_b + 
               pid_cout * stride_out_cout + 
               pid_wo * stride_out_wo + 
               pid_ho * stride_out_ho + 
               pid_do * stride_out_do)
    tl.store(out_ptr + out_off, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # Input dimensions
    B, Cin, W, H, D = x.shape
    Cout, Cin_g, Kw, Kh, Kd = weight.shape
    
    # Calculate output dimensions
    Wo = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    Ho = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    Do = (D + 2 * padding - dilation * (Kd - 1) - 1) // stride + 1
    
    out = torch.empty((B, Cout, Wo, Ho, Do), device=x.device, dtype=x.dtype)
    
    # Strides for x
    stride_x_b = Cin * W * H * D
    stride_x_c = W * H * D
    stride_x_w = H * D
    stride_x_h = D
    stride_x_d = 1
    
    # Strides for weight
    stride_w_cout = Cin_g * Kw * Kh * Kd
    stride_w_c = Kw * Kh * Kd
    stride_w_kw = Kh * Kd
    stride_w_kh = Kd
    stride_w_kd = 1
    
    # Strides for output
    stride_out_b = Cout * Wo * Ho * Do
    stride_out_cout = Wo * Ho * Do
    stride_out_wo = Ho * Do
    stride_out_ho = Do
    stride_out_do = 1
    
    # Grid: one program per output element
    grid = (B, Cout, Wo, Ho, Do)
    
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, W, H, D,
        Cout, Kw, Kh, Kd,
        Wo, Ho, Do,
        stride, padding, dilation, groups,
        stride_x_b, stride_x_c, stride_x_w, stride_x_h, stride_x_d,
        stride_w_cout, stride_w_c, stride_w_kw, stride_w_kh, stride_w_kd,
        stride_out_b, stride_out_cout, stride_out_wo, stride_out_ho, stride_out_do,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size # (Kw, Kh, Kd)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Weight shape: (out_channels, in_channels // groups, Kw, Kh, Kd)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on CUDA
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        return triton_conv3d(
            x, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups
        )