import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // groups, kD, kH, kW)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride, padding, output_padding,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels per group
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Get batch index and output channel index
    batch_idx = pid_batch
    out_channel_start = pid_c_out * BLOCK_SIZE_M
    
    # Compute group index
    group_size_out = C_out // groups
    group_idx = out_channel_start // group_size_out
    local_out_channel = out_channel_start % group_size_out
    
    # Offset for output tensor
    out_ptr_batch = out_ptr + batch_idx * C_out * D_out * H_out * W_out
    
    # For each output position
    for d_out in range(D_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Compute the corresponding input position
                d_in = d_out - padding + output_padding // 2
                h_in = h_out - padding + output_padding // 2
                w_in = w_out - padding + output_padding // 2
                
                # Adjust for stride
                d_in = d_in // stride
                h_in = h_in // stride
                w_in = w_in // stride
                
                # Check if input position is valid
                if (0 <= d_in < D_in and 0 <= h_in < H_in and 0 <= w_in < W_in):
                    # Compute input offset
                    input_offset = batch_idx * C_in * D_in * H_in * W_in + \
                                  d_in * C_in * H_in * W_in + \
                                  h_in * C_in * W_in + \
                                  w_in * C_in
                    # Compute kernel offset based on group
                    kernel_base_offset = group_idx * (C_in // groups) * group_size_out * kD * kH * kW
                    
                    # Initialize accumulator
                    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
                    
                    # Loop over input channels in the group
                    for c_in_start in range(0, C_in // groups, BLOCK_SIZE_K):
                        # Compute input channel offsets
                        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_K)
                        c_in_mask = c_in_offsets < (C_in // groups)
                        
                        # Compute actual input channel indices (including group offset)
                        actual_c_in = group_idx * (C_in // groups) + c_in_offsets
                        
                        # Load input values
                        x_ptrs = x_ptr + input_offset + actual_c_in
                        x_vals = tl.load(x_ptrs, mask=c_in_mask, other=0.0)
                        
                        # Load kernel values
                        # Compute kernel indices for this output position
                        d_k = d_out - d_in * stride + padding
                        h_k = h_out - h_in * stride + padding
                        w_k = w_out - w_in * stride + padding
                        
                        # Only compute if kernel indices are valid
                        if (0 <= d_k < kD and 0 <= h_k < kH and 0 <= w_k < kW):
                            # Compute kernel offsets
                            kernel_offsets = kernel_base_offset + \
                                           c_in_start * group_size_out * kD * kH * kW + \
                                           local_out_channel * kD * kH * kW + \
                                           d_k * kH * kW + h_k * kW + w_k
                            k_ptrs = w_ptr + kernel_offsets
                            k_vals = tl.load(k_ptrs, mask=c_in_mask, other=0.0)
                            
                            # Accumulate
                            acc += x_vals * k_vals
                    
                    # Add bias if present
                    if b_ptr is not None:
                        bias_offset = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
                        bias_vals = tl.load(b_ptr + bias_offset, 
                                          mask=bias_offset < C_out, other=0.0)
                        acc += bias_vals
                    
                    # Store result
                    out_offsets = tl.arange(0, BLOCK_SIZE_M)
                    out_mask = out_offsets < BLOCK_SIZE_M
                    out_vals = acc.to(tl.float32)
                    
                    out_ptr_offset = out_ptr_batch + \
                                    (out_channel_start + out_offsets) * D_out * H_out * W_out + \
                                    d_out * H_out * W_out + \
                                    h_out * W_out + w_out
                    tl.store(out_ptr + out_ptr_offset, out_vals, mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, kD, kH, kW = weight.shape
    C_out = C_in_w * C_out_per_group
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride - 2 * padding + (kD - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + (kH - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (kW - 1) + output_padding + 1
    
    # Initialize output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_K = 8
    
    # Grid dimensions
    grid = (B, (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride, padding, output_padding,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights using the same shape as ConvTranspose3d
        # Weight shape: (in_channels, out_channels // groups, kD, kH, kW)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )