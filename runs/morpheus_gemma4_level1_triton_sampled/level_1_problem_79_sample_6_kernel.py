import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,          # Input tensor pointer
    weight_ptr,     # Weight tensor pointer
    bias_ptr,       # Bias tensor pointer (optional)
    out_ptr,        # Output tensor pointer
    B, C_in, L_in, C_out, K, S, P, D, L_out,
    BLOCK_L: tl.constexpr,
):
    # Program ID for batch and output channel
    pid_b_cout = tl.program_id(0)
    # Program ID for output length block
    pid_l = tl.program_id(1)

    # Decompose pid_b_cout into batch index and output channel index
    b = pid_b_cout // C_out
    cout = pid_b_cout % C_out

    # Calculate the range of output length indices this block handles
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L_out

    # Initialize accumulator with bias if available
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + cout)
        acc += bias_val

    # Loop over input channels and kernel elements
    # Since C_in and K are typically small, we iterate over them inside the kernel
    for cin in range(C_in):
        for k in range(K):
            # The transposed convolution relation: l_out = l_in * stride - padding + k * dilation
            # Solving for l_in: l_in * stride = l_out + padding - k * dilation
            
            # Calculate the potential input index for each l_out in the block
            tmp = l_offsets + P - k * D
            
            # Condition 1: tmp must be divisible by stride
            # Condition 2: tmp must be non-negative
            # Condition 3: the resulting l_in must be within bounds [0, L_in)
            l_in = tmp // S
            cond = (tmp % S == 0) & (tmp >= 0) & (l_in < L_in)
            
            # Load the weight for this (cin, cout, k) combination
            # Weight shape: (C_in, C_out, K)
            w_ptr = weight_ptr + cin * (C_out * K) + cout * K + k
            w_val = tl.load(w_ptr)
            
            # Load the input value x[b, cin, l_in]
            # Input shape: (B, C_in, L_in)
            x_ptr_offset = b * (C_in * L_in) + cin * L_in + l_in
            x_val = tl.load(x_ptr + x_ptr_offset, mask=cond & mask_l, other=0.0)
            
            # Accumulate the product
            acc += x_val * w_val

    # Store the result in the output tensor
    # Output shape: (B, C_out, L_out)
    out_ptr_offset = b * (C_out * L_out) + cout * L_out + l_offsets
    tl.store(out_ptr + out_ptr_offset, acc, mask=mask_l)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # Tensors must be contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, L_in = x.shape
    C_in_w, C_out, K = weight.shape
    
    # Calculate output length based on ConvTranspose1d formula
    # L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    out = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    BLOCK_L = 256
    grid = (B * C_out, triton.cdiv(L_out, BLOCK_L))
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, L_in, C_out, K, stride, padding, dilation, L_out,
        BLOCK_L=BLOCK_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for ConvTranspose1d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the nn.ConvTranspose1d layer to manage weights and bias
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Store parameters for the Triton wrapper
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom Triton kernel.
        """
        return triton_conv_transpose1d(
            x, 
            self.conv1d_transpose.weight, 
            self.conv1d_transpose.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )