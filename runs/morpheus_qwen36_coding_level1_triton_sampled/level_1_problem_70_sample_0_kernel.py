import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    in_ptr, out_ptr, weight_ptr, bias_ptr,
    B, C_in, D, H, W,
    C_out, D_out, H_out, W_out,
    kernel_size, stride, padding, dilation, output_padding,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread computes one output element
    idx = tl.program_id(0)
    total_elements = B * C_out * D_out * H_out * W_out
    if idx >= total_elements:
        return
    
    # Decode linear index to 5D coordinates
    w = idx % W_out
    idx //= W_out
    h = idx % H_out
    idx //= H_out
    d = idx % D_out
    idx //= D_out
    c_out = idx % C_out
    idx //= C_out
    b = idx
    
    acc = 0.0
    for c_in in range(C_in):
        for k_d in range(kernel_size):
            for k_h in range(kernel_size):
                for k_w in range(kernel_size):
                    in_d = d * stride - padding + k_d * dilation
                    in_h = h * stride - padding + k_h * dilation
                    in_w = w * stride - padding + k_w * dilation
                    
                    if 0 <= in_d < D and 0 <= in_h < H and 0 <= in_w < W:
                        in_idx = b * C_in * D * H * W + c_in * D * H * W + in_d * H * W + in_h * W + in_w
                        w_idx = c_out * C_in * kernel_size**3 + c_in * kernel_size**3 + k_d * kernel_size**2 + k_h * kernel_size + k_w
                        in_val = tl.load(in_ptr + in_idx)
                        w_val = tl.load(weight_ptr + w_idx)
                        acc += in_val * w_val
    
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)
    
    out_idx = b * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, output_padding: int, dilation: int, groups: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, D, H, W = x.shape
    C_out, C_in_w, kD, kH, kW = weight.shape
    assert C_in == C_in_w and kD == kH == kW, "Inconsistent dimensions."
    kernel_size = kD
    
    # Compute output dimensions
    D_out = (D - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    
    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    total_elements = B * C_out * D_out * H_out * W_out
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv_transpose3d_kernel[grid](
        x, out, weight, bias,
        B, C_in, D, H, W,
        C_out, D_out, H_out, W_out,
        kernel_size, stride, padding, dilation, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, 
            self.dilation, self.groups
        )


def get_inputs():
    batch_size = 8
    in_channels = 48
    out_channels = 24
    kernel_size = 3
    depth = 96
    height = 96
    width = 96
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]


def get_init_inputs():
    return [48, 24, 3]