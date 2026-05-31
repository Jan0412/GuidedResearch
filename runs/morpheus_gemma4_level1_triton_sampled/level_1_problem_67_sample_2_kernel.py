import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, dilation, groups,
    L_in, L_out,
    x_stride_b, x_stride_c, x_stride_l,
    w_stride_oc, w_stride_ic, w_stride_k,
    out_stride_b, out_stride_oc, out_stride_l,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Program IDs
    pid_oc_b = tl.program_id(0)  # Combined batch and output channel
    pid_l = tl.program_id(1)     # Output length block

    # Decompose pid_oc_b into batch index and output channel index
    batch_idx = pid_oc_b // out_channels
    out_chan_idx = pid_oc_b % out_channels

    # Calculate the group for this output channel
    # Each group has out_channels // groups filters
    group_id = out_chan_idx // (out_channels // groups)
    in_channels_per_group = in_channels // groups
    in_chan_start = group_id * in_channels_per_group

    # Output length offsets for this block
    l_start = pid_l * BLOCK_SIZE_L
    l_offsets = l_start + tl.arange(0, BLOCK_SIZE_L)
    mask_l = l_offsets < L_out

    # Initialize accumulator with bias
    # Bias is per output channel
    bias = tl.load(b_ptr + out_chan_idx) if b_ptr is not None else 0.0
    acc = tl.full((BLOCK_SIZE_L,), bias, dtype=tl.float32)

    # Loop over the input channels in the assigned group
    for ic_offset in range(in_channels_per_group):
        ic = in_chan_start + ic_offset
        
        # Loop over the kernel window
        for k in range(kernel_size):
            # Calculate the input position for each element in the output block
            # input_pos = l_out * stride + k * dilation - padding
            input_pos = l_offsets * stride + k * dilation - padding
            
            # Mask for input bounds
            mask_input = (input_pos >= 0) & (input_pos < L_in) & mask_l
            
            # Load input: (batch, in_channel, length)
            x_ptr_current = x_ptr + (batch_idx * x_stride_b) + (ic * x_stride_c) + input_pos
            x_val = tl.load(x_ptr_current, mask=mask_input, other=0.0)
            
            # Load weight: (out_channel, in_channel_per_group, kernel_size)
            # weight index: [out_chan_idx, ic_offset, k]
            w_ptr_current = w_ptr + (out_chan_idx * w_stride_oc) + (ic_offset * w_stride_ic) + (k * w_stride_k)
            w_val = tl.load(w_ptr_current)
            
            acc += x_val * w_val

    # Store the result: (batch, out_channel, length)
    out_ptr_current = out_ptr + (batch_idx * out_stride_b) + (out_chan_idx * out_stride_oc) + l_offsets
    tl.store(out_ptr_current, acc, mask=mask_l)


def triton_conv1d(x, weight, bias, stride, padding, dilation, groups):
    # Input shapes
    batch_size, in_channels, L_in = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output length
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, L_out), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    out = out.contiguous()

    # Strides
    x_stride_b, x_stride_c, x_stride_l = x.stride()
    w_stride_oc, w_stride_ic, w_stride_k = weight.stride()
    out_stride_b, out_stride_oc, out_stride_l = out.stride()

    BLOCK_SIZE_L = 1024
    # Grid: (Batch * OutChannels, ceil(L_out / BLOCK_SIZE_L))
    grid = (batch_size * out_channels, triton.cdiv(L_out, BLOCK_SIZE_L))

    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        stride, padding, dilation, groups,
        L_in, L_out,
        x_stride_b, x_stride_c, x_stride_l,
        w_stride_oc, w_stride_ic, w_stride_k,
        out_stride_b, out_stride_oc, out_stride_l,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv1d to manage the parameters (weights and bias)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton kernel instead of the PyTorch Conv1d forward pass
        return triton_conv1d(
            x, 
            self.conv1d.weight, 
            self.conv1d.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )