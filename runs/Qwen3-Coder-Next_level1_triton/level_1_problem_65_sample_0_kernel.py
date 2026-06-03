import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (N, C_in, H, W)
    w_ptr,  # Weight tensor (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,) or None
    y_ptr,  # Output tensor (N, C_out, H_out, W_out)
    N, C_in, C_out, H_in, W_in, K_h, K_w, 
    stride, padding, output_padding,
    H_out, W_out,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1) * BLOCK_SIZE_C_out
    h_block = tl.program_id(2) * BLOCK_SIZE_H
    w_block = tl.program_id(3) * BLOCK_SIZE_W
    
    # Compute output coordinates
    c_out_start = c_out_block + tl.arange(0, BLOCK_SIZE_C_out)
    h_start = h_block + tl.arange(0, BLOCK_SIZE_H)
    w_start = w_block + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid indices
    c_out_mask = c_out_start < C_out
    h_mask = h_start < H_out
    w_mask = w_start < W_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_block in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_start = c_in_block + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_start < C_in
        
        # Load input data: shape [C_in, H_in, W_in]
        # For each c_in in the block, load [H_in, W_in] data
        # Since we're iterating over channels, we'll process each c_in individually
        for i_c_in in range(BLOCK_SIZE_C_in):
            if c_in_block + i_c_in < C_in:
                # Compute input position for each output position
                # For transposed conv: output = stride * (input - 1) + (kernel - 1) - 2*padding + output_padding + 1
                # Rearranged: input = (output + padding - (kernel - 1) - output_padding - 1) / stride + 1
                
                # Compute h_out positions that map to valid h_in positions
                # h_in = (h_out - (K_h - 1) - output_padding + 2*padding) / stride + 1
                # Simplified: h_in = (h_out - K_h + 1 + output_padding + 2*padding) / stride + 1
                
                for i_h in range(BLOCK_SIZE_H):
                    h_out_idx = h_block + i_h
                    if h_out_idx < H_out:
                        # Calculate the starting h_in index for this h_out
                        h_in_start_idx = h_out_idx - (K_h - 1) + padding
                        h_in_start_idx = (h_in_start_idx + stride - 1) // stride  # ceiling division
                        
                        # Only proceed if there are valid h_in indices
                        for i_w in range(BLOCK_SIZE_W):
                            w_out_idx = w_block + i_w
                            if w_out_idx < W_out:
                                # Calculate the starting w_in index for this w_out
                                w_in_start_idx = w_out_idx - (K_w - 1) + padding
                                w_in_start_idx = (w_in_start_idx + stride - 1) // stride  # ceiling division
                                
                                # Now iterate over the kernel positions
                                sum_val = 0.0
                                valid = True
                                for k_h in range(K_h):
                                    for k_w in range(K_w):
                                        # Calculate corresponding input position
                                        h_in = h_out_idx - k_h + padding
                                        w_in = w_out_idx - k_w + padding
                                        
                                        # Check if this position maps to a valid input
                                        if (h_in % stride == 0 and w_in % stride == 0):
                                            h_in_valid = h_in // stride
                                            w_in_valid = w_in // stride
                                            
                                            if (0 <= h_in_valid < H_in and 0 <= w_in_valid < W_in):
                                                # Load input value
                                                x_offset = (batch_idx * C_in * H_in * W_in + 
                                                           (c_in_block + i_c_in) * H_in * W_in + 
                                                           h_in_valid * W_in + w_in_valid)
                                                x_val = tl.load(x_ptr + x_offset)
                                                
                                                # Load weight value
                                                w_offset = ((c_in_block + i_c_in) * C_out * K_h * K_w + 
                                                           c_out_start[:, None, None] * K_h * K_w + 
                                                           k_h * K_w + k_w)
                                                w_val = tl.load(w_ptr + w_offset, 
                                                              mask=c_out_mask[None, :, None] & (k_h == 0) & (k_w == 0),
                                                              other=0.0)
                                                
                                                # Accumulate
                                                sum_val += x_val * w_val[0]
                                
                                # Store to accumulator
                                if h_mask[i_h] and w_mask[i_w]:
                                    acc[:, i_h, i_w] += sum_val
    
    # Apply bias if present
    if b_ptr is not None:
        b_offset = c_out_start[:, None, None]
        bias = tl.load(b_ptr + b_offset, mask=c_out_mask[:, None, None], other=0.0)
        acc += bias
    
    # Store result
    for i_c_out in range(BLOCK_SIZE_C_out):
        for i_h in range(BLOCK_SIZE_H):
            for i_w in range(BLOCK_SIZE_W):
                if (c_out_block + i_c_out < C_out and 
                    h_block + i_h < H_out and 
                    w_block + i_w < W_out):
                    y_offset = (batch_idx * C_out * H_out * W_out + 
                               (c_out_block + i_c_out) * H_out * W_out + 
                               (h_block + i_h) * W_out + (w_block + i_w))
                    tl.store(y_ptr + y_offset, acc[i_c_out, i_h, i_w])

# Optimized implementation using PyTorch's native conv_transpose2d with a wrapper
# For simplicity and correctness, we'll use PyTorch's implementation but wrap it properly
# The above kernel is complex and error-prone; instead, we'll use a more practical approach

# Better approach: Implement a simple but efficient kernel for the specific case
# using the direct convolution approach but for transposed convolution

@triton.jit
def conv_transpose2d_simple_kernel(
    x_ptr,  # Input: (N, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,)
    y_ptr,  # Output: (N, C_out, H_out, W_out)
    N, C_in, C_out, H_in, W_in, K_h, K_w,
    stride, padding, output_padding,
    H_out, W_out,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Each program computes one output element
    n_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    h_out_idx = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out_idx = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Masks
    h_mask = h_out_idx < H_out
    w_mask = w_out_idx < W_out
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in_idx in range(C_in):
        # Iterate over kernel height
        for k_h in range(K_h):
            # Compute input height index
            h_in = h_out_idx - k_h + padding
            if stride > 1:
                h_valid = (h_in % stride == 0)
                h_in = h_in // stride
                h_valid = h_valid & (h_in >= 0) & (h_in < H_in)
            else:
                h_valid = (h_in >= 0) & (h_in < H_in)
            
            # Iterate over kernel width
            for k_w in range(K_w):
                # Compute input width index
                w_in = w_out_idx - k_w + padding
                if stride > 1:
                    w_valid = (w_in % stride == 0)
                    w_in = w_in // stride
                    w_valid = w_valid & (w_in >= 0) & (w_in < W_in)
                else:
                    w_valid = (w_in >= 0) & (w_in < W_in)
                
                # Combined valid mask
                valid_mask = h_mask[:, None] & w_mask[None, :] & (h_valid[:, None] & w_valid[None, :])
                
                # Load input values where valid
                if stride > 1:
                    # Only compute for valid positions
                    x_offsets = n_idx * C_in * H_in * W_in + c_in_idx * H_in * W_in + h_in[:, None] * W_in + w_in[None, :]
                    x_vals = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
                else:
                    x_offsets = n_idx * C_in * H_in * W_in + c_in_idx * H_in * W_in + h_in[:, None] * W_in + w_in[None, :]
                    x_vals = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
                
                # Load weight value
                w_offset = c_in_idx * C_out * K_h * K_w + c_out_idx * K_h * K_w + k_h * K_w + k_w
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_vals * w_val
    
    # Apply bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out_idx)
        acc += b_val
    
    # Store output
    y_offsets = (n_idx * C_out * H_out * W_out + 
                c_out_idx * H_out * W_out + 
                h_out_idx[:, None] * W_out + w_out_idx[None, :])
    tl.store(y_ptr + y_offsets, acc, mask=h_mask[:, None] & w_mask[None, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Performs transposed 2D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    N, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + (K_h - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (K_w - 1) + output_padding + 1
    
    # Prepare output tensor
    y = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_KH = 1
    BLOCK_SIZE_KW = 1
    
    grid = (N, C_out, (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, 
            (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    conv_transpose2d_simple_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out, H_in, W_in, K_h, K_w,
        stride, padding, output_padding,
        H_out, W_out,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 2D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create the weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding,
            groups=self.groups
        )