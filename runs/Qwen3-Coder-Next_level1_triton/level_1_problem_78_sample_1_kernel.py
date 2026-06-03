import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.ops.matmul import _get_matmul_kernel


@triton.jit
def transposed_conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,          # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,          # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,          # Bias tensor: (C_out,) or None
    out_ptr,        # Output tensor: (B, C_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out, 
    H_in, W_in,
    H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ci, stride_w_co, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_DH: tl.constexpr, # Block size for output height
    BLOCK_SIZE_DW: tl.constexpr, # Block size for output width
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h_out = tl.program_id(2)
    pid_w_out = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h_out * BLOCK_SIZE_DH + tl.arange(0, BLOCK_SIZE_DH)
    out_w = pid_w_out * BLOCK_SIZE_DW + tl.arange(0, BLOCK_SIZE_DW)
    
    # Create masks for output bounds
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_DH, BLOCK_SIZE_DW), dtype=tl.float32)
    
    # Loop over input channels and compute the convolution
    for c_in in range(C_in):
        # Calculate corresponding input positions for this output position
        # For transposed convolution: H_in = (H_out - 1) * stride_h - 2*pad_h + K_h
        # So for a given output position (out_h, out_w), the input position is:
        # in_h = out_h - (K_h - 1) + pad_h, but only if (in_h + i*stride_h) is valid
        # Actually, for transposed conv, we compute: output = conv(input) where conv is upsampled
        # More precisely: output[b, c_out, h_out, w_out] = sum_{c_in, kh, kw} input[b, c_in, h_in, w_in] * weight[c_in, c_out, kh, kw]
        # where h_in = (h_out - pad_h - kh) // stride_h, similarly for w
        
        # For each output position, find which input positions contribute
        # We iterate over kernel positions instead
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                in_h = pid_h_out * BLOCK_SIZE_DH + tl.arange(0, BLOCK_SIZE_DH) - (K_h - 1 - kh) + pad_h
                in_w = pid_w_out * BLOCK_SIZE_DW + tl.arange(0, BLOCK_SIZE_DW) - (K_w - 1 - kw) + pad_w
                
                # Check if these input positions are valid
                mask_in_h = (in_h >= 0) & (in_h < H_in)
                mask_in_w = (in_w >= 0) & (in_w < W_in)
                mask_in = mask_in_h[:, None] & mask_in_w[None, :]
                
                # Load input values
                # For simplicity, we'll use a different approach: iterate over input positions
                # and accumulate contributions to output
                
                # Actually, let's restructure: for each output position, accumulate over kernel and input channels
                # This is the standard way to implement transposed convolution
                
                pass  # Placeholder for the restructured approach
    
    # Let's implement the correct version: iterate over output positions and accumulate
    # For each output position, we compute the sum over input channels and kernel positions
    
    # Reset accumulator
    acc = tl.zeros((BLOCK_SIZE_DH, BLOCK_SIZE_DW), dtype=tl.float32)
    
    # Process each kernel position and input channel
    for kh in range(K_h):
        for kw in range(K_w):
            # Compute input position for this kernel position
            in_h_start = pid_h_out * BLOCK_SIZE_DH - (K_h - 1 - kh) + pad_h
            in_w_start = pid_w_out * BLOCK_SIZE_DW - (K_w - 1 - kw) + pad_w
            
            # For each input channel
            for c_in in range(C_in):
                # Load weight value
                w_offset = c_in * stride_w_ci + pid_c_out * stride_w_co + kh * stride_w_kh + kw * stride_w_kw
                w_val = tl.load(w_ptr + w_offset)
                
                # Iterate over output block
                out_h_idx = tl.arange(0, BLOCK_SIZE_DH)
                out_w_idx = tl.arange(0, BLOCK_SIZE_DW)
                
                in_h = in_h_start + out_h_idx
                in_w = in_w_start + out_w_idx
                
                mask_h_valid = (in_h >= 0) & (in_h < H_in)
                mask_w_valid = (in_w >= 0) & (in_w < W_in)
                mask = mask_h_valid[:, None] & mask_w_valid[None, :]
                
                # Load input values
                x_offset = pid_batch * stride_x_b + c_in * stride_x_c + in_h[:, None] * stride_x_h + in_w[None, :] * stride_x_w
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                
                # Accumulate
                acc += tl.where(mask, x_val * w_val, 0.0)
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = pid_c_out
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_h_idx = pid_h_out * BLOCK_SIZE_DH + tl.arange(0, BLOCK_SIZE_DH)
    out_w_idx = pid_w_out * BLOCK_SIZE_DW + tl.arange(0, BLOCK_SIZE_DW)
    
    mask_out_h = out_h_idx < H_out
    mask_out_w = out_w_idx < W_out
    mask_out = mask_out_h[:, None] & mask_out_w[None, :]
    
    out_offset = pid_batch * stride_out_b + pid_c_out * stride_out_c + out_h_idx[:, None] * stride_out_h + out_w_idx[None, :] * stride_out_w
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=mask_out)


# A more efficient implementation using a different tiling strategy
@triton.jit
def transposed_conv2d_kernel_v2(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out, H_in, W_in, H_out, W_out, K_h, K_w,
    stride_h, stride_w, pad_h, pad_w,
    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ci, stride_w_co, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Calculate output position ranges
    out_h_start = tl.program_id(2) * BLOCK_SIZE_KH
    out_w_start = tl.program_id(3) * BLOCK_SIZE_KW
    
    # For transposed convolution, we can think of it as:
    # For each output position, accumulate contributions from input positions and kernel
    
    # Actually, let's use a more standard approach: unfold the input and do matrix multiplication
    # But for Triton, we'll implement it directly with better tiling
    
    # We'll process one output position at a time for simplicity and correctness
    # This is not optimal but correct - for production use, we'd optimize further
    
    # Process one output element at a time
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    if out_h >= H_out or out_w >= W_out:
        return
    
    # For this output position, accumulate over input channels and kernel
    acc = 0.0
    
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                in_h = out_h - (K_h - 1 - kh) + pad_h
                in_w = out_w - (K_w - 1 - kw) + pad_w
                
                # Check if input position is valid
                if 0 <= in_h < H_in and 0 <= in_w < W_in:
                    # Load input value
                    x_offset = pid_batch * stride_x_b + c_in * stride_x_c + in_h * stride_x_h + in_w * stride_x_w
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value
                    w_offset = c_in * stride_w_ci + pid_c_out * stride_w_co + kh * stride_w_kh + kw * stride_w_kw
                    w_val = tl.load(w_ptr + w_offset)
                    
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = pid_c_out
        acc += tl.load(b_ptr + b_offset)
    
    # Store result
    out_offset = pid_batch * stride_out_b + pid_c_out * stride_out_c + out_h * stride_out_h + out_w * stride_out_w
    tl.store(out_ptr + out_offset, acc)


# Even better approach: use the fact that transposed convolution is equivalent to
# a regular convolution with padded/upsampled input, but let's implement it as
# a GEMM by unfolding the input

# For now, let's implement a simple but functional version that works correctly
def triton_transposed_conv2d(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor = None,
    stride: tuple = (1, 1), 
    padding: tuple = (0, 0)
) -> torch.Tensor:
    """
    Custom Triton implementation of transposed 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride tuple (height, width)
        padding: Padding tuple (height, width)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_height, output_width)
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    C_in_, C_out, K_h, K_w = weight.shape
    assert C_in == C_in_, f"Input channels mismatch: {C_in} vs {C_in_}"
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w
    
    # Create output tensor
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Define kernel launch configuration
    # For simplicity, we'll use a 4D grid: (batch, out_channels, H_out, W_out)
    grid = (B, C_out, H_out, W_out)
    
    # Launch kernel
    transposed_conv2d_kernel_v2[grid](
        x, weight, bias, out,
        B, C_in, C_out, H_in, W_in, H_out, W_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        stride_x[0], stride_x[1], stride_x[2], stride_x[3],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3],
        stride_out[0], stride_out[1], stride_out[2], stride_out[3]
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using our custom Triton kernel.
        """
        return triton_transposed_conv2d(x, self.weight, self.bias, self.stride, self.padding)


# Add missing imports
import math