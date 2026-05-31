import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, length, length_out,
    kernel_size, stride, padding, dilation, groups,
    BLOCK_SIZE_L: tl.constexpr,
):
    # pid 0: batch * out_channels
    # pid 1: block index for output length
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    batch_idx = pid_0 // out_channels
    oc_idx = pid_0 % out_channels
    
    # Output length offsets for this block
    l_offsets = pid_1 * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = l_offsets < length_out
    
    # Grouping logic
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups
    group_id = oc_idx // out_channels_per_group
    
    # Initialize accumulator with bias if it exists
    acc = 0.0
    if b_ptr is not None:
        acc = tl.load(b_ptr + oc_idx)
    
    # Convolution loop
    # We iterate over input channels within the group and the kernel size
    for ic_in_group in range(in_channels_per_group):
        ic_idx = group_id * in_channels_per_group + ic_in_group
        for k in range(kernel_size):
            # Calculate input index: l_in = l_out * stride + k * dilation - padding
            l_in = l_offsets * stride + k * dilation - padding
            mask_in = mask_l & (l_in >= 0) & (l_in < length)
            
            # Load input value: x shape (batch, in_channels, length)
            x_offset = batch_idx * in_channels * length + ic_idx * length + l_in
            x_val = tl.load(x_ptr + x_offset, mask=mask_in, other=0.0)
            
            # Load weight value: w shape (out_channels, in_channels // groups, kernel_size)
            w_offset = oc_idx * in_channels_per_group * kernel_size + ic_in_group * kernel_size + k
            w_val = tl.load(w_ptr + w_offset)
            
            acc += x_val * w_val
    
    # Store result: out shape (batch, out_channels, length_out)
    out_ptr_offset = batch_idx * out_channels * length_out + oc_idx * length_out + l_offsets
    tl.store(out_ptr + out_ptr_offset, acc, mask=mask_l)

def triton_conv1d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output length based on PyTorch Conv1d formula
    length_out = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, length_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_L = 256
    grid = (batch_size * out_channels, (length_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, length_out,
        kernel_size, stride, padding, dilation, groups,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized 1D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv1d to manage parameters and initialization consistently with the original model
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel.
        """
        return triton_conv1d(
            x, 
            self.conv1d.weight, 
            self.conv1d.bias, 
            self.conv1d.stride, 
            self.conv1d.padding[0], 
            self.conv1d.dilation[0], 
            self.conv1d.groups
        )