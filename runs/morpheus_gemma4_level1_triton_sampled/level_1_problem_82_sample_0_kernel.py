import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, H, W, KH, KW, S, P, OH, OW,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh, stride_wkw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    has_bias,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    # Map pid_bc to batch and channel indices
    batch_idx = pid_bc // C
    channel_idx = pid_bc % C

    # Output spatial offsets
    oh_offsets = pid_oh * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = pid_ow * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Pointers to the start of the current batch and channel
    x_base = x_ptr + batch_idx * stride_xb + channel_idx * stride_xc
    w_base = weight_ptr + channel_idx * stride_wc
    out_base = out_ptr + batch_idx * stride_ob + channel_idx * stride_oc

    # Accumulator for the output block
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Convolution loop
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates for the current kernel element
            # Broadcast oh_offsets to (BLOCK_SIZE_H, 1) and ow_offsets to (1, BLOCK_SIZE_W)
            h_in = oh_offsets[:, None] * S + kh - P
            w_in = ow_offsets[None, :] * S + kw - P

            # Mask for boundary/padding checks
            mask = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

            # Compute linear offsets for the input tensor
            offsets = h_in * stride_xh + w_in * stride_xw
            
            # Load input values and multiply by the corresponding weight
            vals = tl.load(x_base + offsets, mask=mask, other=0.0)
            w_val = tl.load(w_base + kh * stride_wkh + kw * stride_wkw)
            acc += vals * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val

    # Store the final result in the output tensor
    out_offsets = oh_offsets[:, None] * stride_oh + ow_offsets[None, :] * stride_ow
    out_mask = (oh_offsets[:, None] < OH) & (ow_offsets[None, :] < OW)
    tl.store(out_base + out_offsets, acc, mask=out_mask)


def triton_depthwise_conv(x, weight, bias, stride, padding):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    KH, _, KW, _ = weight.shape # weight is (C, 1, KH, KW) for depthwise
    # Note: nn.Conv2d weight shape is (out_channels, in_channels/groups, kH, kW)
    # For depthwise, out_channels = C and groups = C, so shape is (C, 1, KH, KW)
    KH = weight.shape[2]
    KW = weight.shape[3]

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)

    # Strides for pointer calculation
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wc, stride_wkh, stride_wkw = weight.stride(0), weight.stride(2), weight.stride(3)
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()

    # Handle bias
    has_bias = bias is not None
    bias_ptr = bias.contiguous() if has_bias else torch.zeros(1, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (B * C, triton.cdiv(OH, BLOCK_SIZE_H), triton.cdiv(OW, BLOCK_SIZE_W))

    depthwise_conv_kernel[grid](
        x, weight, bias_ptr, out,
        B, C, H, W, KH, KW, stride, padding, OH, OW,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wkh, stride_wkw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        has_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_enabled = bias
        
        # We use nn.Conv2d to manage weights and bias initialization and registration
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding
        )