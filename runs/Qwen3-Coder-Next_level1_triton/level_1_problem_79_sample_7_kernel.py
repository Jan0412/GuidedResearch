import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Input tensor (B, C_in, L_in)
    w_ptr,  # Weight tensor (C_in, C_out, K)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, L_out)
    B, C_in, C_out, L_in, L_out, K, 
    stride, padding, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Batch size block
    BLOCK_SIZE_N: tl.constexpr,  # Output channels block
    BLOCK_SIZE_L: tl.constexpr,  # Output length block
    BLOCK_SIZE_K: tl.constexpr,  # Kernel size block
    BLOCK_SIZE_C: tl.constexpr,  # Input channels block
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Batch index
    batch_idx = pid_b
    out_channel_idx = pid_c_out * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_channel_mask = out_channel_idx < C_out
    
    # Output position
    out_start = pid_l * BLOCK_SIZE_L
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE_L)
    out_mask = out_offsets < L_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_offsets < C_in
        
        # Load input values for this batch and channel block
        # Shape: (BLOCK_SIZE_C, L_in)
        x_block = tl.load(
            x_ptr + batch_idx * C_in * L_in + c_in_offsets[:, None] * L_in + tl.arange(0, L_in)[None, :],
            mask=c_in_mask[:, None] & (tl.arange(0, L_in)[None, :] < L_in),
            other=0.0
        )
        
        # Iterate over kernel size
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < K
            
            # Compute input position for each output position and kernel offset
            # out_pos = in_pos * stride + k * dilation - padding
            # => in_pos = (out_pos + padding - k * dilation) / stride
            
            # Compute input positions for all combinations
            out_pos = out_offsets[None, None, :]  # [1, 1, BLOCK_SIZE_L]
            k_pos = k_offsets[None, :, None]      # [1, BLOCK_SIZE_K, 1]
            c_in_pos = c_in_offsets[:, None, None]  # [BLOCK_SIZE_C, 1, 1]
            
            # Calculate input positions: in_pos = (out_pos + padding - k * dilation) / stride
            in_pos_num = out_pos + padding - k_pos * dilation
            in_pos = in_pos_num // stride
            
            # Check if valid input position (integer division must be exact)
            valid_in_pos = (in_pos_num % stride == 0) & (in_pos >= 0) & (in_pos < L_in)
            
            # Load weight values for this kernel block and output channel
            # Shape: (BLOCK_SIZE_C, BLOCK_SIZE_N, BLOCK_SIZE_K)
            w_block = tl.load(
                w_ptr + c_in_offsets[:, None, None] * C_out * K + out_channel_idx[None, :, None] * K + k_offsets[None, None, :],
                mask=c_in_mask[:, None, None] & out_channel_mask[None, :, None] & k_mask[None, None, :],
                other=0.0
            )
            
            # For each output position, accumulate contributions from valid input positions
            # x_block shape: (BLOCK_SIZE_C, L_in)
            # w_block shape: (BLOCK_SIZE_C, BLOCK_SIZE_N, BLOCK_SIZE_K)
            
            # We need to index x_block using in_pos, which is [BLOCK_SIZE_C, BLOCK_SIZE_K, BLOCK_SIZE_L]
            # First, flatten the indices for efficient gathering
            
            # Reshape for broadcasting
            x_block_expanded = x_block[:, None, :, :]  # [BLOCK_SIZE_C, 1, 1, L_in]
            w_block_expanded = w_block[:, :, :, None]  # [BLOCK_SIZE_C, BLOCK_SIZE_N, BLOCK_SIZE_K, 1]
            
            # Create a mask for valid positions
            valid_mask = valid_in_pos[:, None, :, :]  # [BLOCK_SIZE_C, 1, BLOCK_SIZE_K, BLOCK_SIZE_L]
            
            # Compute contribution: x[in_pos] * w[c_in, c_out, k]
            # We need to gather x values at positions in_pos
            # Since Triton doesn't have advanced indexing, we use a different approach
            
            # For each c_in, k, compute the contribution to each out_pos
            # Flatten to work with 2D operations
            c_in_flat = c_in_offsets[:, None, None]  # [BLOCK_SIZE_C, 1, 1]
            k_flat = k_offsets[None, :, None]  # [1, BLOCK_SIZE_K, 1]
            
            # Compute the actual in_pos for indexing
            in_pos_flat = in_pos.reshape(BLOCK_SIZE_C * BLOCK_SIZE_K * BLOCK_SIZE_L)
            valid_flat = valid_in_pos.reshape(BLOCK_SIZE_C * BLOCK_SIZE_K * BLOCK_SIZE_L)
            
            # Create a temporary tensor to store the result
            # Since Triton doesn't support dynamic indexing well, we use a different approach
            
            # Alternative approach: iterate through each c_in and k
            for c_idx in range(BLOCK_SIZE_C):
                c_offset = c_in_start + c_idx
                if c_offset >= C_in:
                    continue
                    
                for k_idx in range(BLOCK_SIZE_K):
                    k_offset = k_start + k_idx
                    if k_offset >= K:
                        continue
                    
                    # Compute input positions for this kernel element
                    in_pos_k = (out_offsets + padding - k_offset * dilation) // stride
                    valid_pos = ((out_offsets + padding - k_offset * dilation) % stride == 0) & \
                               (in_pos_k >= 0) & (in_pos_k < L_in)
                    
                    # Load x values at valid positions
                    x_vals = tl.load(
                        x_ptr + batch_idx * C_in * L_in + c_offset * L_in + in_pos_k,
                        mask=valid_pos,
                        other=0.0
                    )
                    
                    # Load weight value
                    w_val = tl.load(
                        w_ptr + c_offset * C_out * K + out_channel_idx * K + k_offset,
                        mask=out_channel_mask,
                        other=0.0
                    )
                    
                    # Broadcast and accumulate
                    # x_vals: [BLOCK_SIZE_L], w_val: [BLOCK_SIZE_N]
                    x_broadcast = x_vals[None, :]  # [1, BLOCK_SIZE_L]
                    w_broadcast = w_val[:, None]    # [BLOCK_SIZE_N, 1]
                    acc += x_broadcast * w_broadcast
                    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx, mask=out_channel_mask)
        acc += bias[:, None]
    
    # Store result
    acc = acc.to(tl.float32)
    tl.store(
        out_ptr + batch_idx * C_out * L_out + out_channel_idx[:, None] * L_out + out_offsets[None, :],
        acc,
        mask=out_channel_mask[:, None] & out_mask[None, :]
    )


def triton_conv1d_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Tunable parameters for block sizes
    BLOCK_SIZE_M = 1  # Batch size block (1 for simplicity)
    BLOCK_SIZE_N = 16  # Output channels block
    BLOCK_SIZE_L = 128  # Output length block
    BLOCK_SIZE_K = 8   # Kernel size block
    BLOCK_SIZE_C = 16  # Input channels block
    
    # Determine grid dimensions
    grid = (
        B,  # batch blocks
        (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,  # output channel blocks
        (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L,  # output length blocks
    )
    
    # Launch the Triton kernel
    conv1d_transpose_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) / (in_channels * kernel_size) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, 
                                      self.stride, self.padding, self.dilation)