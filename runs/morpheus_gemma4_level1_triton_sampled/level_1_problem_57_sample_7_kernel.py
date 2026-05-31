import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, y_ptr,
    S, P,
    H_in, W_in, H_out, W_out,
    C_in, C_out, kH, kW,
    stride_x_n, stride_x_c, stride_x_h,
    stride_w_cin, stride_w_cout, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_cout, stride_y_h,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    # Program IDs
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Batch and Output Channel
    n = pid_nc // C_out
    c_out = pid_nc % C_out

    # Output block coordinates
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Output mask
    mask_h = h_offs < H_out
    mask_w = w_offs < W_out

    # Accumulator for the block
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over kernel dimensions
    for kh in range(kH):
        for kw in range(kW):
            # Calculate input coordinates for the current kernel offset
            # i = (h + P - kh) / S
            # j = (w + P - kw) / S
            i_offs = (h_offs + P - kh) // S
            j_offs = (w_offs + P - kw) // S

            # Valid input index conditions:
            # 1. Boundary check
            # 2. Stride check: (h + P - kh) % S == 0
            valid_i = (i_offs >= 0) & (i_offs < H_in) & ((h_offs + P - kh) % S == 0)
            valid_j = (j_offs >= 0) & (j_offs < W_in) & ((w_offs + P - kw) % S == 0)
            
            # Combined spatial mask (BLOCK_H, BLOCK_W)
            spatial_mask = valid_i[:, None] & valid_j[None, :]

            # Iterate over input channels in blocks
            for cin_start in range(0, C_in, BLOCK_C):
                c_in_offs = cin_start + tl.arange(0, BLOCK_C)
                c_in_mask = c_in_offs < C_in

                # Load input block: (BLOCK_C, BLOCK_H, BLOCK_W)
                # x is (N, C_in, H_in, W_in)
                x_base = x_ptr + n * stride_x_n
                x_idx = (c_in_offs[:, None, None] * stride_x_c + 
                         i_offs[None, :, None] * stride_x_h + 
                         j_offs[None, None, :])
                
                # Final mask for input load
                x_mask = c_in_mask[:, None, None] & spatial_mask[None, :, :]
                x_val = tl.load(x_base + x_idx, mask=x_mask, other=0.0)

                # Load weight block: (BLOCK_C, 1, 1)
                # w is (C_in, C_out, kH, kW)
                w_base = w_ptr + c_out * stride_w_cout + kh * stride_w_kh + kw * stride_w_kw
                w_idx = c_in_offs[:, None, None] * stride_w_cin
                w_val = tl.load(w_base + w_idx, mask=c_in_mask[:, None, None], other=0.0)

                # Multiply and accumulate
                acc += tl.sum(x_val * w_val, axis=0)

    # Store final result to output tensor
    y_base = y_ptr + n * stride_y_n + c_out * stride_y_cout
    y_idx = h_offs[:, None] * stride_y_h + w_offs[None, :]
    y_mask = mask_h[:, None] & mask_w[None, :]
    tl.store(y_base + y_idx, acc, mask=y_mask)


def triton_conv_transpose2d(x, weight, stride, padding, output_padding, bias=None):
    # Input: x (N, C_in, H_in, W_in), weight (C_in, C_out, kH, kW)
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, kH, kW = weight.shape
    
    # Output dimensions calculation
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding
    
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Contiguous tensors are required for pointer arithmetic
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Strides
    stride_x_n = C_in * H_in * W_in
    stride_x_c = H_in * W_in
    stride_x_h = W_in
    
    stride_w_cin = C_out * kH * kW
    stride_w_cout = kH * kW
    stride_w_kh = kW
    stride_w_kw = 1
    
    stride_y_n = C_out * H_out * W_out
    stride_y_cout = H_out * W_out
    stride_y_h = W_out
    
    # Tuning parameters
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32
    
    grid = (N * C_out, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    conv_transpose_kernel[grid](
        x, weight, y,
        stride, padding,
        H_in, W_in, H_out, W_out,
        C_in, C_out, kH, kW,
        stride_x_n, stride_x_c, stride_x_h,
        stride_w_cin, stride_w_cout, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_cout, stride_y_h,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    
    if bias is not None:
        y += bias.view(1, -1, 1, 1)
        
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use the original ConvTranspose2d to initialize weights and bias
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the nn.ConvTranspose2d layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        # Ensure input is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Call the custom Triton implementation
        return triton_conv_transpose2d(
            x, 
            weight, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            bias
        )