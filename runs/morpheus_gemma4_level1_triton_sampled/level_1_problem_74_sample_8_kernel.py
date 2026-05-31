import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, in_channels, out_channels, length,
    kernel_size, stride, padding, dilation,
    length_out,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Grid: (batch * out_channels, ceil(length_out / BLOCK_SIZE_L))
    pid_bc = tl.program_id(0)
    pid_l = tl.program_id(1)

    b = pid_bc // out_channels
    oc = pid_bc % out_channels

    offsets_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = offsets_l < length_out

    # Load bias for the current output channel
    # bias shape: (out_channels,)
    bias_val = tl.load(bias_ptr + oc)
    acc = tl.broadcast_to(bias_val, (BLOCK_SIZE_L,))

    # Pre-calculate indices for input channels
    ic_offsets = tl.arange(0, BLOCK_SIZE_C)
    ic_mask = ic_offsets < in_channels

    for k in range(kernel_size):
        # Calculate input index j for each output index i in the block
        # Formula: i = j * stride - padding + k * dilation
        # Therefore: j = (i + padding - k * dilation) / stride
        j_num = offsets_l + padding - k * dilation
        
        # Condition: j_num must be divisible by stride and within [0, length)
        j_divisible = (j_num % stride) == 0
        j = j_num // stride
        j_mask = j_divisible & (j >= 0) & (j < length) & mask_l

        # Load weight vector for fixed oc and k: weight[ic, oc, k]
        # weight shape: (in_channels, out_channels, kernel_size)
        w_ptr = weight_ptr + ic_offsets * (out_channels * kernel_size) + oc * kernel_size + k
        w = tl.load(w_ptr, mask=ic_mask, other=0.0)  # Shape (BLOCK_SIZE_C,)

        # Load x values: x[b, ic, j]
        # x shape: (batch, in_channels, length)
        j_expanded = j[:, None]  # (BLOCK_SIZE_L, 1)
        ic_expanded = ic_offsets[None, :]  # (1, BLOCK_SIZE_C)
        
        x_ptr_calc = x_ptr + b * (in_channels * length) + ic_expanded * length + j_expanded
        x_mask = j_mask[:, None] & ic_mask[None, :]
        x = tl.load(x_ptr_calc, mask=x_mask, other=0.0)  # (BLOCK_SIZE_L, BLOCK_SIZE_C)

        # Dot product over input channels: sum(x[b, ic, j] * w[ic, oc, k])
        acc += tl.sum(x * w[None, :], axis=1)

    # Store result: out[b, oc, i]
    out_ptr_calc = out_ptr + b * (out_channels * length_out) + oc * length_out + offsets_l
    tl.store(out_ptr_calc, acc, mask=mask_l)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # x: (batch, in_channels, length)
    # weight: (in_channels, out_channels, kernel_size)
    # bias: (out_channels,) or None
    batch, in_channels, length = x.shape
    in_channels_w, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    # L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    x = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    
    if bias is not None:
        bias = bias.contiguous().cuda()
    else:
        bias = torch.zeros(out_channels, device=x.device, dtype=x.dtype)
        
    out = torch.empty((batch, out_channels, length_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_L = 128
    BLOCK_SIZE_C = triton.next_power_of_2(in_channels)
    
    grid = (batch * out_channels, triton.cdiv(length_out, BLOCK_SIZE_L))
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch, in_channels, out_channels, length,
        kernel_size, stride, padding, dilation,
        length_out,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Use nn.ConvTranspose1d to manage weights and bias
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the PyTorch layer
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias
        
        # Call the Triton-optimized implementation
        return triton_conv_transpose1d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )