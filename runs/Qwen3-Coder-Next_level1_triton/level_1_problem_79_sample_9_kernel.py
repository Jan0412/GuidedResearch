import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    # Pointers to tensors
    x_ptr,          # Input: (batch, in_channels, L_in)
    w_ptr,          # Weight: (in_channels, out_channels, kernel_size)
    b_ptr,          # Bias: (out_channels,) or None
    out_ptr,        # Output: (batch, out_channels, L_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels, kernel_size,
    L_in, L_out,
    stride, padding, dilation,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_N: tl.constexpr,  # Block size for out_channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels
    BLOCK_SIZE_L: tl.constexpr,  # Block size for sequence length
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    # Compute output position
    out_pos = pid_seq * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    out_mask = out_pos < L_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels
    for ic in range(in_channels):
        # For each output position, compute which input positions contribute
        # In transposed conv: out_pos = ic_pos * stride + kernel_offset - padding
        # So: ic_pos = (out_pos + padding - kernel_offset) / stride
        
        # Kernel offset range [0, kernel_size-1]
        kernel_offset = tl.arange(0, BLOCK_SIZE_K)
        kernel_mask = kernel_offset < kernel_size
        
        # Compute the corresponding input positions
        # ic_pos = (out_pos + padding - kernel_offset * dilation) / stride
        # Only valid if the division is exact and within bounds
        
        # We need to compute for each out_pos and kernel_offset
        # Expand out_pos for broadcasting with kernel_offset
        out_pos_expanded = out_pos[:, None]  # (BLOCK_SIZE_L, 1)
        kernel_offset_expanded = kernel_offset[None, :]  # (1, BLOCK_SIZE_K)
        
        # Compute input position
        ic_pos = (out_pos_expanded + padding - kernel_offset_expanded * dilation) // stride
        
        # Check if position is valid: divisible and within bounds
        is_valid = (
            ((out_pos_expanded + padding - kernel_offset_expanded * dilation) % stride == 0) &
            (ic_pos >= 0) & (ic_pos < L_in)
        )
        
        # Load input values where valid
        ic_pos_flat = ic_pos.flatten()
        is_valid_flat = is_valid.flatten()
        mask_flat = is_valid_flat & (ic_pos_flat < L_in)
        
        # We need to handle this more carefully - use a different approach
        # For each output position, iterate through kernel positions
        
        # Actually, let's restructure: iterate through kernel positions for each output position
        pass
    
    # Better approach: iterate through kernel positions
    for k in range(kernel_size):
        # For each kernel position, compute which output positions get contributions from which input positions
        # out_pos = in_pos * stride + k * dilation - padding
        # So for fixed k, in_pos -> out_pos = in_pos * stride + k * dilation - padding
        
        in_pos = tl.arange(0, BLOCK_SIZE_L)
        in_mask = in_pos < L_in
        
        # Compute output position for this input position and kernel offset
        out_pos_calc = in_pos * stride + k * dilation - padding
        
        # Only keep valid output positions
        out_pos_valid = out_pos_calc >= 0
        out_pos_final = tl.where(out_pos_valid, out_pos_calc, 0)
        out_mask_valid = out_pos_final < L_out
        
        # Load input: shape (batch, in_channels, L_in)
        # We need to load for all batches and this in_pos
        # Since BLOCK_SIZE_M=1 (batch), we handle one batch at a time
        
        # Load input value at this position
        x_offset = pid_batch * in_channels * L_in + ic * L_in + in_pos
        x_val = tl.load(x_ptr + x_offset, mask=in_mask & out_mask_valid, other=0.0)
        
        # Load weight: w[ic, oc, k]
        w_offset = ic * out_channels * kernel_size + pid_out_c * kernel_size + k
        w_val = tl.load(w_ptr + w_offset)
        
        # Accumulate: x_val * w_val goes to out_pos_final
        if out_pos_final.shape[0] == BLOCK_SIZE_L:
            # Broadcast x_val (BLOCK_SIZE_L,) to (BLOCK_SIZE_L, 1) and multiply by w_val
            acc += x_val[:, None] * w_val * tl.where(
                (out_pos_final[:, None] == out_pos[None, :]) & out_mask_valid[:, None], 1.0, 0.0
            )
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_out_c
        b_val = tl.load(b_ptr + b_offset)
        acc += b_val
    
    # Store result
    out_offset = pid_batch * out_channels * L_out + pid_out_c * L_out + out_pos
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, L_in = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, L_out, device=x.device, dtype=x.dtype)
    
    # Configure grid and block sizes
    # Use multiple blocks for out_channels and sequence length
    BLOCK_SIZE_M = 1  # One batch per block
    BLOCK_SIZE_N = 16  # out_channels per block
    BLOCK_SIZE_L = 256  # sequence length per block
    BLOCK_SIZE_K = 8   # kernel size per block (not used in this approach)
    
    # Grid: (batch, out_channels // BLOCK_SIZE_N, L_out // BLOCK_SIZE_L)
    grid = (
        batch_size,
        triton.cdiv(out_channels, BLOCK_SIZE_N),
        triton.cdiv(L_out, BLOCK_SIZE_L)
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        L_in, L_out,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters manually
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights (using the same initialization as nn.ConvTranspose1d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 1D convolution using Triton kernel.
        """
        return triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


import math