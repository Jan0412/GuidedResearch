import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, OC, L, K, S, P, D, G, L_out,
    stride_x_b, stride_x_c, stride_x_l,
    stride_w_oc, stride_w_ic, stride_w_k,
    stride_out_b, stride_out_oc, stride_out_l,
    BLOCK_L: tl.constexpr,
):
    # Program IDs
    pid_b_oc = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Decompose pid_b_oc into batch_id and oc_id
    batch_id = pid_b_oc // OC
    oc_id = pid_b_oc % OC

    # Output length range for this block
    l_out_range = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_out_range < L_out

    # Initialize accumulator
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Input channel group start for the current output channel
    # PyTorch Conv1d: each group has OC // G output channels and IC // G input channels
    group_id = oc_id // (OC // G)
    ic_start = group_id * (IC // G)

    # Loop over input channels within the group
    for ic_offset in range(0, IC // G):
        curr_ic = ic_start + ic_offset
        # Loop over kernel window
        for k in range(0, K):
            # Calculate input index: l_in = l_out * stride - padding + k * dilation
            l_in = l_out_range * S - P + k * D
            mask_in = (l_in >= 0) & (l_in < L) & mask_l
            
            # Load input value
            # x is (B, IC, L)
            x_off = batch_id * stride_x_b + curr_ic * stride_x_c + l_in * stride_x_l
            val_x = tl.load(x_ptr + x_off, mask=mask_in, other=0.0)
            
            # Load weight value
            # w is (OC, IC // G, K)
            w_off = oc_id * stride_w_oc + ic_offset * stride_w_ic + k * stride_w_k
            val_w = tl.load(w_ptr + w_off)
            
            acc += val_x * val_w

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc_id)
        acc += bias_val

    # Store result
    # out is (B, OC, L_out)
    out_off = batch_id * stride_out_b + oc_id * stride_out_oc + l_out_range * stride_out_l
    tl.store(out_ptr + out_off, acc, mask=mask_l)


def triton_conv1d(x, weight, bias, stride, padding, dilation, groups):
    B, IC, L = x.shape
    OC, IC_per_group, K = weight.shape
    
    # Calculate output length
    L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    out = torch.empty((B, OC, L_out), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    out = out.contiguous()

    # Strides
    stride_x_b, stride_x_c, stride_x_l = x.stride()
    stride_w_oc, stride_w_ic, stride_w_k = weight.stride()
    stride_out_b, stride_out_oc, stride_out_l = out.stride()

    BLOCK_L = 128
    grid = (B * OC, (L_out + BLOCK_L - 1) // BLOCK_L)

    conv1d_kernel[grid](
        x, weight, bias, out,
        B, IC, OC, L, K, stride, padding, dilation, groups, L_out,
        stride_x_b, stride_x_c, stride_x_l,
        stride_w_oc, stride_w_ic, stride_w_k,
        stride_out_b, stride_out_oc, stride_out_l,
        BLOCK_L=BLOCK_L,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Use nn.Conv1d to manage weights and bias initialization
        self.conv_params = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on GPU and FP32
        x = x.cuda().float()
        weight = self.conv_params.weight.cuda().float()
        bias = self.conv_params.bias.cuda().float() if self.conv_params.bias is not None else None
        
        return triton_conv1d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )