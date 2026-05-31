import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride, padding, dilation, kernel_size,
    in_channels, out_channels, L_in, L_out,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Map pid_bc to batch_id and oc_id
    batch_id = pid_bc // out_channels
    oc_id = pid_bc % out_channels
    
    # Output sequence offsets for this block
    offsets_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = offsets_l < L_out
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)
    
    # Add bias if it exists
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc_id)
        acc += bias

    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel elements
        for k in range(kernel_size):
            # The formula for transposed convolution input index:
            # j = i * stride - padding + k * dilation
            # => i = (j + padding - k * dilation) / stride
            
            term = offsets_l + padding - k * dilation
            mask_div = (term % stride == 0)
            i = term // stride
            mask_bounds = (i >= 0) & (i < L_in)
            
            # Combined mask for valid input access
            mask = mask_l & mask_div & mask_bounds
            
            # Load x[batch, ic, i]
            # x shape: (batch_size, in_channels, L_in)
            x_off = batch_id * in_channels * L_in + ic * L_in + i
            val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            
            # Load w[ic, oc, k]
            # w shape: (in_channels, out_channels, kernel_size)
            w_off = ic * out_channels * kernel_size + oc_id * kernel_size + k
            weight = tl.load(w_ptr + w_off)
            
            acc += val * weight
    
    # Store the final accumulated result
    y_off = batch_id * out_channels * L_out + oc_id * L_out + offsets_l
    tl.store(y_ptr + y_off, acc, mask=mask_l)

def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation, kernel_size):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, L_in = x.shape
    # Weight shape for ConvTranspose1d: (in_channels, out_channels, kernel_size)
    _, out_channels, _ = weight.shape
    
    # Calculate output length based on PyTorch ConvTranspose1d formula
    # L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # output_padding is 0 by default in the provided Model
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    out = torch.empty((batch_size, out_channels, L_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_L = 256
    grid = (batch_size * out_channels, (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        stride, padding, dilation, kernel_size,
        in_channels, out_channels, L_in, L_out,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for ConvTranspose1d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the ConvTranspose1d module to handle weight/bias initialization and parameter management
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the ConvTranspose1d module
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias
        stride = self.conv1d_transpose.stride[0]
        padding = self.conv1d_transpose.padding[0]
        dilation = self.conv1d_transpose.dilation[0]
        kernel_size = self.conv1d_transpose.kernel_size[0]
        
        # Call the Triton optimized implementation
        return triton_conv_transpose1d(x, weight, bias, stride, padding, dilation, kernel_size)