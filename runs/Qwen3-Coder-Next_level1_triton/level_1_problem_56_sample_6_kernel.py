import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Weight dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    # Output dimensions
    H_out, W_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block_id = tl.program_id(1)
    h_out_block_id = tl.program_id(2)
    w_out_block_id = tl.program_id(3)
    
    # Calculate output positions
    c_out = c_out_block_id * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    h_out = h_out_block_id * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = w_out_block_id * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    c_out_mask = c_out < C_out
    h_out_mask = h_out < H_out
    w_out_mask = w_out < W_out
    c_out_grid, h_out_grid, w_out_grid = tl.meshgrid(c_out, h_out, w_out)
    c_out_grid = c_out_grid.flatten()
    h_out_grid = h_out_grid.flatten()
    w_out_grid = w_out_grid.flatten()
    
    # Compute input positions for each output position
    h_in = h_out_grid * stride_h - pad_h + tl.arange(0, BLOCK_SIZE_KH)[None, :] * dil_h
    w_in = w_out_grid * stride_w - pad_w + tl.arange(0, BLOCK_SIZE_KW)[None, :] * dil_w
    
    # Reshape for broadcasting
    h_in = h_in[None, :, :]  # [1, BLOCK_SIZE_H, BLOCK_SIZE_KH]
    w_in = w_in[None, :, :]  # [1, BLOCK_SIZE_W, BLOCK_SIZE_KW]
    
    # Compute convolution
    acc = tl.zeros((BLOCK_SIZE_COUT * BLOCK_SIZE_H * BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_block in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_idx = c_in_block + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_idx < C_in
        
        # Load input block: [N, C_in, H, W]
        # For current batch and channel block
        h_idx = h_out_grid[:, None, None] * stride_h - pad_h + tl.arange(0, BLOCK_SIZE_KH)[None, None, :] * dil_h
        w_idx = w_out_grid[:, None, None] * stride_w - pad_w + tl.arange(0, BLOCK_SIZE_KW)[None, None, :] * dil_w
        
        # Flatten h_idx and w_idx for indexing
        h_idx_flat = h_idx.flatten()
        w_idx_flat = w_idx.flatten()
        
        # Create mask for valid indices
        valid_mask = (h_idx_flat >= 0) & (h_idx_flat < H) & (w_idx_flat >= 0) & (w_idx_flat < W)
        
        # Compute input offsets
        input_offsets = batch_id * (C_in * H * W) + c_in_idx[:, None, None] * (H * W) + \
                       h_idx[None, :, :] * W + w_idx[None, :, :]
        input_offsets = input_offsets.flatten()
        
        # Load input values
        x_val = tl.load(x_ptr + input_offsets, mask=valid_mask, other=0.0)
        x_val = tl.reshape(x_val, (BLOCK_SIZE_CIN, BLOCK_SIZE_H, BLOCK_SIZE_KH, BLOCK_SIZE_W))
        
        # Load weight block: [C_out, C_in, K_h, K_w]
        weight_offsets = c_out[:, None, None, None] * (C_in * K_h * K_w) + \
                        c_in_idx[None, :, None, None] * (K_h * K_w) + \
                        tl.arange(0, BLOCK_SIZE_KH)[None, None, :, None] * K_w + \
                        tl.arange(0, BLOCK_SIZE_KW)[None, None, None, :]
        weight_offsets = weight_offsets.flatten()
        
        # Load weight values
        w_val = tl.load(w_ptr + weight_offsets, mask=c_out_mask[:, None, None, None] & c_in_mask[None, :, None, None], other=0.0)
        w_val = tl.reshape(w_val, (BLOCK_SIZE_COUT, BLOCK_SIZE_CIN, BLOCK_SIZE_KH, BLOCK_SIZE_KW))
        
        # Compute convolution: multiply and accumulate
        # Reshape for multiplication: [C_out, C_in, K_h, K_w] * [C_in, H_out, K_h, W_out] -> [C_out, H_out, W_out]
        # Reshape x_val to [C_in, H_out, K_h, W_out] and w_val to [C_out, C_in, K_h, W_out]
        x_val_reshaped = tl.reshape(x_val, (BLOCK_SIZE_CIN, BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W))
        w_val_reshaped = tl.reshape(w_val, (BLOCK_SIZE_COUT * BLOCK_SIZE_CIN * BLOCK_SIZE_KH * BLOCK_SIZE_KW,))
        
        # Compute outer product and accumulate
        for i in range(BLOCK_SIZE_CIN):
            x_part = x_val[i, :, :, :]  # [BLOCK_SIZE_H, BLOCK_SIZE_KH, BLOCK_SIZE_W]
            w_part = w_val[:, i, :, :]  # [BLOCK_SIZE_COUT, BLOCK_SIZE_KH, BLOCK_SIZE_W]
            
            # Compute convolution sum for this channel
            x_flat = tl.reshape(x_part, (BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W,))
            w_flat = tl.reshape(w_part, (BLOCK_SIZE_COUT * BLOCK_SIZE_KH * BLOCK_SIZE_W,))
            
            # Matrix multiplication approximation: [BLOCK_SIZE_COUT, BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W]
            # and [BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W]
            x_mat = tl.reshape(x_flat, (BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W,))
            w_mat = tl.reshape(w_flat, (BLOCK_SIZE_COUT, BLOCK_SIZE_H * BLOCK_SIZE_KH * BLOCK_SIZE_W))
            
            # Compute dot product
            acc += tl.sum(w_mat * x_mat[None, :], axis=1)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out, mask=c_out_mask)
        acc += bias[:, None, None].flatten()
    
    # Store result
    output_offsets = batch_id * (C_out * H_out * W_out) + c_out[:, None, None] * (H_out * W_out) + \
                    h_out_grid[None, :, :] * W_out + w_out_grid[None, :, :]
    output_offsets = output_offsets.flatten()
    
    tl.store(out_ptr + output_offsets, acc, mask=(c_out_mask[:, None, None] & h_out_mask[None, :, None] & w_out_mask[None, None, :]).flatten())


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Performs 2D convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    # Use reasonable block sizes based on the problem dimensions
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_COUT = min(32, C_out)
    BLOCK_SIZE_CIN = min(16, C_in)
    BLOCK_SIZE_KH = min(5, K_h)
    BLOCK_SIZE_KW = min(7, K_w)
    BLOCK_SIZE_H = min(8, H_out)
    BLOCK_SIZE_W = min(8, W_out)
    
    # Calculate grid dimensions
    grid = (
        N,
        (C_out + BLOCK_SIZE_COUT - 1) // BLOCK_SIZE_COUT,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, padding=self.padding, 
                            dilation=self.dilation)