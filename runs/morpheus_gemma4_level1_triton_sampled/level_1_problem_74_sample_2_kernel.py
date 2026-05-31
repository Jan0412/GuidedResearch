import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, 
    w_ptr, 
    bias_ptr, 
    out_ptr, 
    B, Cin, Cout, L, Lout, 
    kernel_size, stride, padding, dilation,
    has_bias,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Grid: (B, Cout, Lout // BLOCK_SIZE_L)
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    pid_l = tl.program_id(2)

    # Output length offsets for this block
    j_offsets = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_j = j_offsets < Lout

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)

    # Transposed Convolution Logic:
    # The output y[b, co, j] is the sum of x[b, ci, i] * w[ci, co, k]
    # where j = i * stride + k * dilation - padding
    # Therefore: i = (j + padding - k * dilation) / stride
    # Condition: (j + padding - k * dilation) must be divisible by stride and 0 <= i < L

    for k in range(kernel_size):
        # Calculate the input index i for the current output index j and kernel index k
        # i = (j + padding - k * dilation) / stride
        numerator = j_offsets + padding - k * dilation
        i_offsets = numerator // stride
        
        # Valid indices must satisfy the stride condition and be within bounds of the input length L
        valid_i = (numerator % stride == 0) & (i_offsets >= 0) & (i_offsets < L)
        
        # Combine with the output boundary mask
        mask = mask_j & valid_i

        # Sum over input channels
        for ci in range(Cin):
            # Load input x[batch, ci, i]
            # x shape: (B, Cin, L)
            x_off = batch_idx * (Cin * L) + ci * L + i_offsets
            val_x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            
            # Load weight w[ci, co, k]
            # w shape: (Cin, Cout, K)
            w_off = ci * (Cout * kernel_size) + out_channel_idx * kernel_size + k
            val_w = tl.load(w_ptr + w_off)
            
            acc += val_x * val_w

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val

    # Store result: out shape (B, Cout, Lout)
    out_off = batch_idx * (Cout * Lout) + out_channel_idx * Lout + j_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_j)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # x: (B, Cin, L)
    # weight: (Cin, Cout, K)
    # bias: (Cout)
    B, Cin, L = x.shape
    Cin_w, Cout, K = weight.shape
    
    # Calculate output length
    Lout = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()

    # Grid configuration
    BLOCK_SIZE_L = 128
    grid = (B, Cout, (Lout + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)

    conv_transpose1d_kernel[grid](
        x, weight, bias if has_bias else None, out,
        B, Cin, Cout, L, Lout,
        K, stride, padding, dilation,
        has_bias,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use the standard ConvTranspose1d to manage parameters (weights and bias)
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the PyTorch module
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias
        
        # Call the custom Triton implementation
        return triton_conv_transpose1d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )