import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H, W,
    C_out, kH, kW,
    s, p, d, g,
    H_out, W_out,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_oc, stride_out_oh, stride_out_ow,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_W_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # Batch * H_out
    pid_n = tl.program_id(1)  # C_out / BLOCK_SIZE_C_OUT
    pid_k = tl.program_id(2)  # W_out / BLOCK_SIZE_W_OUT

    # Map pid_m to batch and output height
    batch_idx = pid_m // H_out
    oh = pid_m % H_out

    # Map pid_n and pid_k to output channel and width
    oc_start = pid_n * BLOCK_SIZE_C_OUT
    ow_start = pid_k * BLOCK_SIZE_W_OUT

    # Ranges
    oc = oc_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    ow = ow_start + tl.arange(0, BLOCK_SIZE_W_OUT)

    # Masks for output boundaries
    oc_mask = oc < C_out
    ow_mask = ow < W_out

    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_C_OUT, BLOCK_SIZE_W_OUT), dtype=tl.float32)

    # Convolution logic: Iterate over kernel window and input channels
    # For simplicity and performance, we handle groups=1 primarily, but include group offset logic
    # Each output channel oc is connected to a slice of input channels
    # Group offset for the first channel in the current block
    # Note: This assumes BLOCK_SIZE_C_OUT <= C_out // g
    group_id = oc_start // (C_out // g)
    ic_group_offset = group_id * (C_in // g)
    C_in_per_group = C_in // g

    for kh in range(kH):
        for kw in range(kW):
            for ic_start in range(0, C_in_per_group, BLOCK_SIZE_C_IN):
                ic = ic_group_offset + ic_start + tl.arange(0, BLOCK_SIZE_C_IN)
                ic_mask = ic < (ic_group_offset + C_in_per_group)

                # Input indices
                # x_idx = batch_idx * stride_x_b + ic * stride_x_c + (oh*s - p + kh*d) * stride_x_h + (ow*s - p + kw*d) * stride_x_w
                h_in = oh * s - p + kh * d
                w_in = ow * s - p + kw * d
                
                # Load X slice: (BLOCK_SIZE_C_IN, BLOCK_SIZE_W_OUT)
                # We use offsets to create the 2D block
                x_offsets_ic = ic[:, None] * stride_x_c
                x_offsets_ow = ow[None, :] * stride_x_w
                x_base = batch_idx * stride_x_b + h_in * stride_x_h
                
                # Mask for X: check input channel and spatial boundaries
                x_mask = ic_mask[:, None] & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & ow_mask[None, :]
                x_val = tl.load(x_ptr + x_base + x_offsets_ic + x_offsets_ow, mask=x_mask, other=0.0)

                # Weight indices
                # w_idx = oc * stride_w_oc + ic * stride_w_ic + kh * stride_w_kh + kw * stride_w_kw
                w_offsets_oc = oc[:, None] * stride_w_oc
                w_offsets_ic = ic[None, :] * stride_w_ic
                w_base = kh * stride_w_kh + kw * stride_w_kw
                
                # Mask for W: check output channel and input channel boundaries
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_base + w_offsets_oc + w_offsets_ic, mask=w_mask, other=0.0)

                # Compute dot product: (C_out_block, C_in_block) @ (C_in_block, W_out_block)
                acc += tl.dot(w_val, x_val)

    # Add bias
    bias = tl.load(b_ptr + oc, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # Store result
    out_base = batch_idx * stride_out_b + oh * stride_out_oh
    out_offsets_oc = oc[:, None] * stride_out_oc
    out_offsets_ow = ow[None, :] * stride_out_ow
    tl.store(out_ptr + out_base + out_offsets_oc + out_offsets_ow, acc, mask=oc_mask[:, None] & ow_mask[None, :])


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    B, C_in, H, W = x.shape
    C_out, C_in_g, kH, kW = weight.shape
    
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw = weight.stride()
    stride_out_b, stride_out_oc, stride_out_oh, stride_out_ow = out.stride()

    # Tuning parameters
    BLOCK_SIZE_C_OUT = 32
    BLOCK_SIZE_W_OUT = 32
    BLOCK_SIZE_C_IN = 32

    grid = (B * H_out, triton.cdiv(C_out, BLOCK_SIZE_C_OUT), triton.cdiv(W_out, BLOCK_SIZE_W_OUT))

    conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out, kH, kW,
        stride, padding, dilation, groups,
        H_out, W_out,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
        stride_out_b, stride_out_oc, stride_out_oh, stride_out_ow,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_W_OUT=BLOCK_SIZE_W_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # We use nn.Conv2d to manage weights and bias initialization and registration
        self.conv_layer = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous for Triton
        x = x.contiguous()
        weight = self.conv_layer.weight.contiguous()
        bias = self.conv_layer.bias.contiguous() if self.conv_layer.bias is not None else torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
        
        return triton_conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)