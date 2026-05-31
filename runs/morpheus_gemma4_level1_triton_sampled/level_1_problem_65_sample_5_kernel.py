import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, Hin, Win, Cout, kH, kW, S, P, Hout, Wout,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a block of the output tensor for a specific batch and output channel
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid to get batch and output channel
    b = pid // Cout
    co = pid % Cout

    # Output block offsets
    oh_start = pid_h * BLOCK_SIZE_H
    ow_start = pid_w * BLOCK_SIZE_W
    
    offsets_h = oh_start + tl.arange(0, BLOCK_SIZE_H)
    offsets_w = ow_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for output boundaries
    mask_h = offsets_h < Hout
    mask_w = offsets_w < Wout
    
    # Accumulator for the output block
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over input channels and kernel dimensions
    # Since Cin, kH, kW are relatively small, we can loop over them
    for ci in range(Cin):
        for kh in range(kH):
            for kw in range(kW):
                # Transposed Conv coordinate mapping:
                # oh = h * S + kh - P  =>  h = (oh + P - kh) / S
                # ow = w * S + kw - P  =>  w = (ow + P - kw) / S
                
                h = (offsets_h + P - kh) // S
                w = (offsets_w + P - kw) // S
                
                # Condition for valid input indices
                # 1. Must be within input boundaries
                # 2. Must satisfy the stride condition (divisible by S)
                valid_h = (offsets_h + P - kh) % S == 0
                valid_w = (offsets_w + P - kw) % S == 0
                
                mask = (mask_h[:, None] & mask_w[None, :]) & \
                       (valid_h[:, None] & valid_w[None, :]) & \
                       (h[:, None] >= 0) & (h[:, None] < Hin) & \
                       (w[None, :] >= 0) & (w[None, :] < Win)
                
                # Load input value: x[b, ci, h, w]
                # x_ptr is (B, Cin, Hin, Win)
                x_offset = b * (Cin * Hin * Win) + ci * (Hin * Win) + h * Win + w
                x_val = tl.load(x_ptr + x_offset[:, None] * 1 + w[None, :], mask=mask, other=0.0)
                
                # Load weight value: w[ci, co, kh, kw]
                # w_ptr is (Cin, Cout, kH, kW)
                w_offset = ci * (Cout * kH * kW) + co * (kH * kW) + kh * kW + kw
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val

    # Add bias: b[co]
    bias_val = tl.load(b_ptr + co) if b_ptr is not None else 0.0
    acc += bias_val

    # Store the result: out[b, co, oh, ow]
    out_offset = b * (Cout * Hout * Wout) + co * (Hout * Wout) + offsets_h[:, None] * Wout + offsets_w[None, :]
    tl.store(out_ptr + out_offset, acc, mask=(mask_h[:, None] & mask_w[None, :]))


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0):
    # x: (B, Cin, Hin, Win)
    # weight: (Cin, Cout, kH, kW)
    # bias: (Cout,)
    B, Cin, Hin, Win = x.shape
    Cin_w, Cout, kH, kW = weight.shape
    
    # Calculate output dimensions
    Hout = (Hin - 1) * stride - 2 * padding + kH + output_padding
    Wout = (Win - 1) * stride - 2 * padding + kW + output_padding
    
    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    grid = (B * Cout, (Hout + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (Wout + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Hin, Win, Cout, kH, kW, stride, padding, Hout, Wout,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use the original nn.ConvTranspose2d to manage parameters and initialization
        # Note: This custom Triton implementation currently supports groups=1
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the PyTorch layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        
        # Use the custom Triton kernel for the forward pass
        return triton_conv_transpose2d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding
        )