import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,          # Input tensor
    w_ptr,          # Weight tensor
    b_ptr,          # Bias tensor (can be None)
    out_ptr,        # Output tensor
    batch_size, 
    in_channels, 
    out_channels, 
    length_in, 
    length_out, 
    kernel_size, 
    stride, 
    dilation, 
    BLOCK_SIZE_L: tl.constexpr,
):
    # Grid: (batch * out_channels, ceil(length_out / BLOCK_SIZE_L))
    pid_out_chan = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Identify which batch and which output channel this program is processing
    batch_id = pid_out_chan // out_channels
    oc_id = pid_out_chan % out_channels

    # Calculate offsets for the output length block
    l_offsets = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = l_offsets < length_out

    # Initialize accumulator with bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc_id)
        acc = tl.full([BLOCK_SIZE_L], bias_val, dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)

    # Pointers to the start of the specific batch and output channel
    # x: (batch, in_channels, length_in)
    # w: (out_channels, in_channels, kernel_size)
    
    # We iterate over in_channels and kernel_size to compute the convolution
    for ic in range(in_channels):
        # Offset to the current input channel for this batch
        x_chan_ptr = x_ptr + batch_id * (in_channels * length_in) + ic * length_in
        # Offset to the current input channel for this output channel in weights
        w_chan_ptr = w_ptr + oc_id * (in_channels * kernel_size) + ic * kernel_size
        
        for k in range(kernel_size):
            # Load the weight for the current (oc, ic, k)
            weight_val = tl.load(w_chan_ptr + k)
            
            # Calculate input indices: length_out_idx * stride + k * dilation
            x_offsets = l_offsets * stride + k * dilation
            mask_x = x_offsets < length_in
            
            # Load input values and multiply by weight
            x_vals = tl.load(x_chan_ptr + x_offsets, mask=mask_x, other=0.0)
            acc += x_vals * weight_val

    # Store the final result
    out_ptr_block = out_ptr + batch_id * (out_channels * length_out) + oc_id * length_out + l_offsets
    tl.store(out_ptr_block, acc, mask=mask_l)


def triton_conv1d(x, weight, bias, stride, dilation):
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, length_in = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    length_out = (length_in - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, length_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_L = 256
    # Grid: (batch * out_channels, ceil(length_out / BLOCK_SIZE_L))
    grid = (batch_size * out_channels, (length_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)

    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length_in, length_out,
        kernel_size, stride, dilation,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution operation using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the nn.Conv1d module
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        stride = self.conv1d.stride[0]
        dilation = self.conv1d.dilation[0]
        
        return triton_conv1d(x, weight, bias, stride, dilation)