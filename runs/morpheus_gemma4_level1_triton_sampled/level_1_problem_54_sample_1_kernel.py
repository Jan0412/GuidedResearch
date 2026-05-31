import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, D, H, W,
    C_out, C_in_group, KD, KH, KW,
    D_out, H_out, W_out,
    stride, padding, dilation, groups,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    n = tl.program_id(0)
    oc = tl.program_id(1)
    od = tl.program_id(2)
    oh = tl.program_id(3)
    ow_block = tl.program_id(4)

    # Pointers and Strides for Input
    stride_n = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W
    stride_w = 1

    # Pointers and Strides for Weight
    stride_w_oc = C_in_group * KD * KH * KW
    stride_w_cin = KD * KH * KW
    stride_w_kd = KH * KW
    stride_w_kh = KW
    stride_w_kw = 1

    # Pointers and Strides for Output
    stride_out_n = C_out * D_out * H_out * W_out
    stride_out_oc = D_out * H_out * W_out
    stride_out_od = H_out * W_out
    stride_out_oh = W_out
    stride_out_ow = 1

    # Output width offsets
    ow = ow_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_mask = ow < W_out

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Convolution loops
    # Parallelize over output dimensions, loop over input channels and kernel window
    for ic in range(C_in_group):
        # Input channel offset based on groups
        c_offset = (oc // groups) * C_in_group + ic
        x_base_ptr = x_ptr + n * stride_n + c_offset * stride_c

        for kd in range(KD):
            id = od * stride + kd * dilation - padding
            if id < 0 or id >= D:
                continue
            
            for kh in range(KH):
                ih = oh * stride + kh * dilation - padding
                if ih < 0 or ih >= H:
                    continue
                
                for kw in range(KW):
                    # Load weight scalar
                    w_ptr_off = oc * stride_w_oc + ic * stride_w_cin + kd * stride_w_kd + kh * stride_w_kh + kw * stride_w_kw
                    weight_val = tl.load(w_ptr + w_ptr_off)

                    # Calculate input width offsets for the current kernel window
                    iw = ow * stride + kw * dilation - padding
                    
                    # Mask for boundary checks on the width dimension
                    iw_mask = (iw >= 0) & (iw < W)
                    
                    # Load input values (vectorized over BLOCK_SIZE_W)
                    x_ptr_off = x_base_ptr + id * stride_d + ih * stride_h + iw * stride_w
                    input_vals = tl.load(x_ptr + x_ptr_off, mask=iw_mask, other=0.0)
                    
                    acc += input_vals * weight_val

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store result
    out_ptr_off = n * stride_out_n + oc * stride_out_oc + od * stride_out_od + oh * stride_out_oh + ow * stride_out_ow
    tl.store(out_ptr + out_ptr_off, acc, mask=out_mask)


def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # Input shapes
    N, C, D, H, W = x.shape
    C_out, C_in_group, KD, KH, KW = weight.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - dilation * (KD - 1) - 1) // stride + 1
    H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_W = 16
    grid = (N, C_out, D_out, H_out, triton.cdiv(W_out, BLOCK_SIZE_W))

    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C, D, H, W,
        C_out, C_in_group, KD, KH, KW,
        D_out, H_out, W_out,
        stride, padding, dilation, groups,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use Conv3d to manage weights and bias
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Store parameters for the Triton kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()
        
        # Extract weights and bias from the Conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        
        return triton_conv3d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )