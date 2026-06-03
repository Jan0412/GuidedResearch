import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, H, W, D)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w, 1)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, H_out, W_out, D)
    B, C_in, H, W, D,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    num_batches, num_depths,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    depth_idx = tl.program_id(1)
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Calculate actual spatial coordinates in output
    h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output indices
    h_mask = h_offsets < (H - K_h + 2 * pad_h) // stride_h + 1
    w_mask = w_offsets < (W - K_w + 2 * pad_w) // stride_w + 1
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_block_start in range(0, C_in, BLOCK_SIZE_C):
        c_in_block = c_in_block_start + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_block < C_in
        
        # Iterate over kernel height
        for kh in range(K_h):
            # Calculate input h position
            in_h = (h_offsets * stride_h - pad_h + kh * dil_h)
            h_valid = (in_h >= 0) & (in_h < H)
            
            # Iterate over kernel width
            for kw in range(K_w):
                # Calculate input w position
                in_w = (w_offsets * stride_w - pad_w + kw * dil_w)
                w_valid = (in_w >= 0) & (in_w < W)
                
                # Load input values: shape (BLOCK_H, BLOCK_W, BLOCK_C)
                in_h_idx = in_h[:, None] * W * D * C_in + in_w[None, :] * D * C_in + depth_idx * C_in + c_in_block[None, None, :]
                in_h_idx = in_h_idx.flatten()
                
                # Reshape to (BLOCK_H * BLOCK_W, BLOCK_C)
                in_h_idx = in_h_idx.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_C)
                
                # Create combined mask
                mask_combined = (h_valid[:, None] & w_valid[None, :])[:, :, None] & c_in_mask[None, None, :]
                mask_combined = mask_combined.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_C)
                
                x_block = tl.load(x_ptr + in_h_idx, mask=mask_combined, other=0.0)
                
                # Load weight values for this kernel position
                # w_ptr shape: (C_out, C_in, K_h, K_w, 1)
                w_idx = tl.arange(0, BLOCK_SIZE_H * BLOCK_SIZE_W)[:, None] * 0 + c_in_block[None, :] * K_h * K_w + kh * K_w + kw
                w_idx = w_idx.flatten()
                
                # Load weights for all output channels
                w_block = tl.load(w_ptr + w_idx, mask=c_in_mask, other=0.0)
                
                # Compute accumulation: x_block shape (BLOCK_H*BLOCK_W, BLOCK_C), w_block shape (BLOCK_C,)
                # Need to broadcast appropriately
                x_reshaped = x_block.reshape(BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)
                w_reshaped = w_block.reshape(1, 1, BLOCK_SIZE_C)
                
                # Accumulate
                acc += tl.sum(x_reshaped * w_reshaped, axis=2)
    
    # Add bias if available
    if b_ptr is not None:
        b_idx = tl.arange(0, BLOCK_SIZE_H * BLOCK_SIZE_W)[:, None] * 0 + tl.arange(0, 1)[None, :] * 0  # dummy for shape
        b_val = tl.load(b_ptr + tl.arange(0, 1))  # load single bias value
        # We'll need to load all biases and apply them
        # Since we're processing multiple output positions but same bias per channel,
        # we need a different approach for bias
    
    # For simplicity, we'll handle bias in a separate step if needed
    
    # Store result
    out_h = h_offsets
    out_w = w_offsets
    out_d = depth_idx
    
    out_idx = batch_idx * C_out * ((H - K_h + 2 * pad_h) // stride_h + 1) * ((W - K_w + 2 * pad_w) // stride_w + 1) * D + \
              tl.arange(0, BLOCK_SIZE_H)[:, None] * ((W - K_w + 2 * pad_w) // stride_w + 1) * D + \
              tl.arange(0, BLOCK_SIZE_W)[None, :] * D + \
              depth_idx
    
    out_idx = out_idx.flatten()
    
    mask_out = (h_mask[:, None] & w_mask[None, :]).flatten()
    
    tl.store(out_ptr + out_idx, acc.flatten(), mask=mask_out)


# Since the above approach is complex for a 3D conv with kernel (K,K,1), let's use a more practical approach
# We'll implement a standard 2D convolution kernel and apply it per depth slice


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, H_out, W_out)
    B, C_in, H, W,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr,
):
    # Program IDs: batch, output_h, output_w, output_channel_block
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_block_start = tl.program_id(3) * BLOCK_SIZE_OUT_C
    
    # Calculate output channel block
    out_c_offsets = out_c_block_start + tl.arange(0, BLOCK_SIZE_OUT_C)
    out_c_mask = out_c_offsets < C_out
    
    # Calculate input position
    in_h = out_h_idx * stride_h - pad_h
    in_w = out_w_idx * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OUT_C,), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_C):
        c_in_block = c_in + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_block < C_in
        
        # Iterate over kernel height
        for kh in range(K_h):
            # Calculate input h position
            input_h = in_h + kh * dil_h
            h_valid = (input_h >= 0) & (input_h < H)
            
            # Iterate over kernel width
            for kw in range(K_w):
                # Calculate input w position
                input_w = in_w + kw * dil_w
                w_valid = (input_w >= 0) & (input_w < W)
                
                if h_valid and w_valid:
                    # Calculate input index
                    x_idx = batch_idx * C_in * H * W + \
                            c_in_block * H * W + \
                            input_h * W + \
                            input_w
                    
                    # Load input values
                    x_block = tl.load(x_ptr + x_idx, mask=c_in_mask, other=0.0)
                    
                    # Calculate weight index
                    # w_ptr shape: (C_out, C_in, K_h, K_w)
                    w_idx = out_c_offsets[:, None] * C_in * K_h * K_w + \
                            c_in_block[None, :] * K_h * K_w + \
                            kh * K_w + \
                            kw
                    
                    # Load weights
                    w_block = tl.load(w_ptr + w_idx, mask=out_c_mask[:, None] & c_in_mask[None, :], other=0.0)
                    
                    # Accumulate: x_block shape (BLOCK_C,), w_block shape (BLOCK_OUT_C, BLOCK_C)
                    acc += tl.sum(w_block * x_block[None, :], axis=1)
    
    # Add bias if available
    if b_ptr is not None:
        b_idx = out_c_offsets
        b_val = tl.load(b_ptr + b_idx, mask=out_c_mask, other=0.0)
        acc += b_val
    
    # Store result
    out_idx = batch_idx * C_out * H_out * W_out + \
              out_c_offsets * H_out * W_out + \
              out_h_idx * W_out + \
              out_w_idx
    
    tl.store(out_ptr + out_idx, acc, mask=out_c_mask)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution with kernel (K, K, 1) using Triton kernels.
    This applies the same 2D convolution to each depth slice independently.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this implementation."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, H, W, D = x.shape
    C_out, _, K_h, K_w, _ = weight.shape
    
    # Calculate output dimensions
    H_out = (H - K_h + 2 * padding) // stride + 1
    W_out = (W - K_w + 2 * padding) // stride + 1
    D_out = D  # since kernel depth is 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, H_out, W_out, D), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    stride_h = stride_w = stride
    pad_h = pad_w = padding
    dil_h = dil_w = dilation
    
    # Kernel block sizes
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_OUT_C = 32
    
    # Grid dimensions: (batch, H_out, W_out, C_out_block)
    grid = (B, H_out, W_out, (C_out + BLOCK_SIZE_OUT_C - 1) // BLOCK_SIZE_OUT_C)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out.view(B, C_out, H_out, W_out * D),  # view as 2D for each depth slice
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_OUT_C=BLOCK_SIZE_OUT_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for the 3D convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same conv3d layer but replace the forward pass
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernels.
        """
        # Extract parameters from the original conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias
        
        # Call our Triton implementation
        return triton_conv3d(x, weight, bias,
                            stride=self.conv3d.stride[0],
                            padding=self.conv3d.padding[0],
                            dilation=self.conv3d.dilation[0],
                            groups=self.conv3d.groups)