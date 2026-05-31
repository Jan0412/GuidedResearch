import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in, C_out, L_in, L_out,
    K, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID maps to (batch, out_channel)
    pid_b_co = tl.program_id(0)
    b = pid_b_co // C_out
    c_out = pid_b_co % C_out
    
    # Program ID 1 maps to input blocks
    pid_block = tl.program_id(1)
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < L_in
    
    # Base pointers
    x_b_ptr = x_ptr + b * C_in * L_in
    w_b_co_ptr = w_ptr + c_out * C_in * K
    out_b_co_ptr = out_ptr + b * C_out * L_out + c_out * L_out
    
    # Load weights for this output channel
    # w shape: (C_out, C_in, K) -> we need (C_in, K)
    w_offsets = tl.arange(0, C_in * K)
    w_mask = w_offsets < C_in * K
    w = tl.load(w_b_co_ptr + w_offsets, mask=w_mask, other=0.0)
    
    # Iterate over input channels and kernel positions
    for c_in in range(0, C_in):
        for k in range(0, K):
            # Compute output positions
            # t_out = t_in * stride + k * dilation - padding
            t_out = offsets * stride + k * dilation - padding
            
            # Mask for valid output positions
            mask_out = (t_out >= 0) & (t_out < L_out)
            
            # Load input values
            # x shape: (B, C_in, L_in)
            x_val = tl.load(x_b_ptr + c_in * L_in + offsets, mask=mask, other=0.0)
            
            # Load weight value
            w_val = tl.load(w_b_co_ptr + c_in * K + k)
            
            # Compute contribution
            val = x_val * w_val
            
            # Atomic add to output
            tl.atomic_add(out_b_co_ptr + t_out, val, mask=mask_out)


def triton_conv_transpose1d(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    kernel_size: int = 5
) -> torch.Tensor:
    """
    Custom Triton implementation of ConvTranspose1d using scatter approach.
    Optimized for FP32 precision.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    B, C_in, L_in = x.shape
    C_out, _, K = w.shape
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + K + (K - 1) * (dilation - 1)
    
    # Prepare output tensor
    out = torch.zeros((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Tunable block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    # 2D grid: (batch * out_channels, num_input_blocks)
    num_blocks = (L_in + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B * C_out, num_blocks)
    
    # Launch kernel
    conv_transpose1d_scatter_kernel[grid](
        x, w, out,
        B, C_in, C_out, L_in, L_out,
        K, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        out = out + bias.unsqueeze(0).unsqueeze(2)
        
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernel for ConvTranspose1d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias manually for Triton kernel
        self.w = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.b = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('b', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x, self.w, self.b,
            self.stride, self.padding, self.dilation, self.kernel_size
        )