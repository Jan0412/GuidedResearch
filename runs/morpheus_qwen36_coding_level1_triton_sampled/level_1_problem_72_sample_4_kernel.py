import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    in_ptr, w_ptr, out_ptr,
    B, C_in, D_in, H_in, W_in,
    C_out, D_out, H_out, W_out,
    D_k, H_k, W_k,
    D_s, H_s, W_s,
    D_p, H_p, W_p,
    D_op, H_op, W_op,
    groups,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Decode output coordinates from 1D program ID
    tmp = pid
    w = tmp % W_out; tmp //= W_out
    h = tmp % H_out; tmp //= H_out
    d = tmp % D_out; tmp //= D_out
    c = tmp % C_out; tmp //= C_out
    b = tmp
    
    # Bounds check for output coordinates
    mask_b = b < B
    mask_c = c < C_out
    mask_d = d < D_out
    mask_h = h < H_out
    mask_w = w < W_out
    
    if not (mask_b and mask_c and mask_d and mask_h and mask_w):
        return
        
    # Determine group for this output channel
    group = c // (C_out // groups)
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over groups, input channels, and kernel dimensions
    for g in range(groups):
        if g != group:
            continue
            
        c_in_start = g * (C_in // groups)
        c_in_end = (g + 1) * (C_in // groups)
        
        for c_in_off in tl.range(c_in_start, c_in_end, BLOCK_C):
            c_in_idx = c_in_off + tl.arange(0, BLOCK_C)
            mask_c_in = c_in_idx < C_in
            
            for kd_off in tl.range(0, D_k, BLOCK_KD):
                kd_idx = kd_off + tl.arange(0, BLOCK_KD)
                mask_kd = kd_idx < D_k
                
                for kh_off in tl.range(0, H_k, BLOCK_KH):
                    kh_idx = kh_off + tl.arange(0, BLOCK_KH)
                    mask_kh = kh_idx < H_k
                    
                    for kw_off in tl.range(0, W_k, BLOCK_KW):
                        kw_idx = kw_off + tl.arange(0, BLOCK_KW)
                        mask_kw = kw_idx < W_k
                        
                        # Compute input coordinates
                        d_in = d + kd_idx - D_op - D_p
                        h_in = h + kh_idx - H_op - H_p
                        w_in = w + kw_idx - W_op - W_p
                        
                        # Create combined mask for input bounds and channel bounds
                        mask_d_in = d_in >= 0
                        mask_h_in = h_in >= 0
                        mask_w_in = w_in >= 0
                        mask_d_in = mask_d_in & (d_in < D_in)
                        mask_h_in = mask_h_in & (h_in < H_in)
                        mask_w_in = mask_w_in & (w_in < W_in)
                        
                        mask_input = mask_d_in & mask_h_in & mask_w_in & mask_c_in & mask_kd & mask_kh & mask_kw
                        
                        if tl.sum(mask_input) > 0:
                            # Load input values
                            in_off = b * C_in * D_in * H_in * W_in + c_in_idx * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in
                            x = tl.load(in_ptr + in_off, mask=mask_input, other=0.0)
                            
                            # Load weight values
                            w_off = c * C_in * D_k * H_k * W_k + c_in_idx * D_k * H_k * W_k + kd_idx * H_k * W_k + kh_idx * W_k + kw_idx
                            w = tl.load(w_ptr + w_off, mask=mask_input, other=0.0)
                            
                            acc += x * w

    # Store result
    out_off = b * C_out * D_out * H_out * W_out + c * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w
    tl.store(out_ptr + out_off, acc, mask=mask_b & mask_c & mask_d & mask_h & mask_w)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, output_padding: tuple, groups: int) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    B, C_in, D_in, H_in, W_in = x.shape
    C_out, _, D_k, H_k, W_k = weight.shape
    
    # Compute output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + D_k + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + H_k + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + W_k + output_padding[2]
    
    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    total_output_elements = B * C_out * D_out * H_out * W_out
    grid = (total_output_elements,)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, out,
        B, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        D_k, H_k, W_k,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        BLOCK_KD=4,
        BLOCK_KH=4,
        BLOCK_KW=4,
        BLOCK_C=4
    )
    
    # Add bias if present
    if bias is not None:
        out += bias.view(1, C_out, 1, 1, 1)
        
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        # Initialize weights to match PyTorch's default initialization for consistency during testing
        nn.init.kaiming_uniform_(self.conv_transpose3d.weight, nonlinearity='linear')
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose3d.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.conv_transpose3d.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        return triton_conv_transpose3d(
            x, weight, bias,
            self.conv_transpose3d.stride,
            self.conv_transpose3d.padding,
            self.conv_transpose3d.output_padding,
            self.conv_transpose3d.groups
        )