import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [batch_size, in_channels, height_in, width_in]
    w_ptr,  # [in_channels, out_channels, kernel_size, kernel_size]
    b_ptr,  # [out_channels] (optional, can be nullptr)
    out_ptr,  # [batch_size, out_channels, height_out, width_out]
    # Dimensions
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kernel_size, stride, padding, dilation,
    # Block sizes for tiling
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create meshgrid for output positions
    out_h_grid, out_w_grid = tl.meshgrid(out_h, out_w)
    out_h_grid = out_h_grid.T
    out_w_grid = out_w_grid.T
    
    # Compute corresponding input positions (considering stride and padding)
    in_h = (out_h_grid - padding) // stride
    in_w = (out_w_grid - padding) // stride
    
    # Check if output position corresponds to valid input position
    h_valid = (in_h >= 0) & (in_h < height_in) & ((out_h_grid - padding) % stride == 0)
    w_valid = (in_w >= 0) & (in_w < width_in) & ((out_w_grid - padding) % stride == 0)
    valid = h_valid & w_valid
    
    # Accumulate over input channels and kernel positions
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for offset_in in range(0, in_channels, BLOCK_IN):
        in_channel_offsets = offset_in + tl.arange(0, BLOCK_IN)
        mask_in = in_channel_offsets < in_channels
        
        # Load input block: [BLOCK_IN, BLOCK_H, BLOCK_W]
        # Need to index input with in_h, in_w
        in_h_flat = in_h * width_in + in_w
        in_batch_offset = pid_b * (in_channels * height_in * width_in)
        
        # Process each input channel in the block
        for i in range(BLOCK_IN):
            if offset_in + i < in_channels:
                ch = offset_in + i
                # Compute input pointer for this channel
                in_ptr = x_ptr + in_batch_offset + ch * (height_in * width_in)
                
                # Load input values with masking
                in_values = tl.load(
                    in_ptr + in_h_flat, 
                    mask=valid & (mask_in[i:i+1] if False else tl.full((BLOCK_H, BLOCK_W), True, tl.bool)),
                    other=0.0
                )
                
                # Now loop over kernel positions
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Compute effective kernel position with dilation
                        eff_kh = kh * dilation
                        eff_kw = kw * dilation
                        
                        # Compute corresponding input position
                        comp_in_h = out_h_grid - padding - eff_kh * stride
                        comp_in_w = out_w_grid - padding - eff_kw * stride
                        
                        # Check if this kernel position contributes
                        kernel_valid = (comp_in_h >= 0) & (comp_in_h < height_in) & \
                                      (comp_in_w >= 0) & (comp_in_w < width_in)
                        
                        # Only compute if kernel position is valid for this output
                        # But we need to be careful - we're computing contributions to out_h_grid, out_w_grid
                        
                        # For each kernel position, compute the weight
                        # Weight indexing: [in_channel, out_channel, kh, kw]
                        w_ptr_offset = in_channel_offsets[i] * (out_channels * kernel_size * kernel_size) + \
                                      pid_out * (kernel_size * kernel_size) + \
                                      kh * kernel_size + kw
                        w_ptr_base = w_ptr + w_ptr_offset
                        weight = tl.load(w_ptr_base)
                        
                        # Compute the input value at the correct position
                        # Actually, for transposed conv, we need to think differently:
                        # out[b, out_c, oh, ow] += sum_in_c sum_kh kw x[b, in_c, ih, iw] * w[in_c, out_c, kh, kw]
                        # where ih = oh*stride + kh*dilation - padding
                        #       iw = ow*stride + kw*dilation - padding
                        
                        # So for fixed output position (oh, ow), we compute contributions from input positions
                        # determined by kernel positions.
                        
                        # Let's restructure: for each output position, we sum over kernel positions and input channels
                        # Input position: ih = oh*stride + kh*dilation - padding
                        #                iw = ow*stride + kw*dilation - padding
                        
                        # This approach is better: loop over kernel positions and compute input positions
                        # But we already have out_h_grid, out_w_grid as our output positions
                        pass  # We'll restructure the kernel below
                
    # Let's rewrite this kernel with a cleaner approach
    # For each output position (oh, ow), we compute:
    # out[b, out_c, oh, ow] = bias[out_c] + sum_{in_c, kh, kw} x[b, in_c, ih, iw] * w[in_c, out_c, kh, kw]
    # where ih = oh*stride + kh*dilation - padding, iw = ow*stride + kw*dilation - padding
    
    # Reset accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # For each output position in the block
    for oh_idx in range(BLOCK_H):
        for ow_idx in range(BLOCK_W):
            oh = pid_h * BLOCK_H + oh_idx
            ow = pid_w * BLOCK_W + ow_idx
            
            if oh >= height_out or ow >= width_out:
                continue
                
            # For each input channel
            for in_c in range(in_channels):
                # For each kernel position
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Compute input position
                        ih = oh * stride + kh * dilation - padding
                        iw = ow * stride + kw * dilation - padding
                        
                        # Check if input position is valid
                        if ih >= 0 and ih < height_in and iw >= 0 and iw < width_in:
                            # Compute input pointer
                            x_offset = pid_b * (in_channels * height_in * width_in) + \
                                      in_c * (height_in * width_in) + \
                                      ih * width_in + iw
                            x_val = tl.load(x_ptr + x_offset)
                            
                            # Compute weight pointer
                            w_offset = in_c * (out_channels * kernel_size * kernel_size) + \
                                      pid_out * (kernel_size * kernel_size) + \
                                      kh * kernel_size + kw
                            w_val = tl.load(w_ptr + w_offset)
                            
                            acc = acc + x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out)
        acc = acc + bias
    
    # Store result
    out_offset = pid_b * (out_channels * height_out * width_out) + \
                pid_out * (height_out * width_out) + \
                pid_h * BLOCK_H * width_out + pid_w * BLOCK_W
    
    # Handle boundaries
    out_h_global = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_global = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_h_grid, out_w_grid = tl.meshgrid(out_h_global, out_w_global)
    out_h_grid = out_h_grid.T
    out_w_grid = out_w_grid.T
    
    mask = (out_h_grid < height_out) & (out_w_grid < width_out)
    
    # Store acc values
    for oh_idx in range(BLOCK_H):
        for ow_idx in range(BLOCK_W):
            if oh_idx < BLOCK_H and ow_idx < BLOCK_W:
                if pid_h * BLOCK_H + oh_idx < height_out and pid_w * BLOCK_W + ow_idx < width_out:
                    tl.store(out_ptr + out_offset + oh_idx * width_out + ow_idx, acc[oh_idx, ow_idx])


@triton.jit
def transposed_conv2d_kernel_v2(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    height_in, width_in, height_out, width_out,
    kernel_size, stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Optimized transposed convolution kernel using tiling.
    """
    # Compute output tile indices
    pid_bh = tl.program_id(0)
    pid_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Batch index
    pid_b = pid_bh // out_channels
    pid_out_tile = pid_bh % out_channels
    
    # If we're using multiple output tiles, adjust
    if out_channels > 1:
        pid_out = pid_out_tile
    
    # Output positions
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_h_grid, out_w_grid = tl.meshgrid(out_h, out_w)
    out_h_grid = out_h_grid.T
    out_w_grid = out_w_grid.T
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for offset_in in range(0, in_channels, BLOCK_K):
        in_c_block = offset_in + tl.arange(0, BLOCK_K)
        in_c_mask = in_c_block < in_channels
        
        # Process each input channel in the block
        for i in range(BLOCK_K):
            if offset_in + i < in_channels:
                in_c = offset_in + i
                
                # Compute input positions for all output positions
                # ih = oh*stride + kh*dilation - padding
                # For each kernel position
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Compute input position
                        ih = out_h_grid * stride + kh * dilation - padding
                        iw = out_w_grid * stride + kw * dilation - padding
                        
                        # Valid mask
                        valid = (ih >= 0) & (ih < height_in) & (iw >= 0) & (iw < width_in)
                        
                        # Load input values
                        in_ptr = x_ptr + pid_b * (in_channels * height_in * width_in) + \
                                in_c * (height_in * width_in)
                        ih_flat = ih * width_in + iw
                        x_val = tl.load(in_ptr + ih_flat, mask=valid, other=0.0)
                        
                        # Load weight
                        w_ptr_offset = in_c * (out_channels * kernel_size * kernel_size) + \
                                      pid_out * (kernel_size * kernel_size) + \
                                      kh * kernel_size + kw
                        w_val = tl.load(w_ptr + w_ptr_offset)
                        
                        # Accumulate
                        acc = tl.where(valid, acc + x_val * w_val, acc)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out)
        acc = acc + bias
    
    # Store result
    out_ptr_base = pid_b * (out_channels * height_out * width_out) + \
                  pid_out * (height_out * width_out)
    
    # Store with bounds checking
    for h_idx in range(BLOCK_H):
        for w_idx in range(BLOCK_W):
            oh = pid_h * BLOCK_H + h_idx
            ow = pid_w * BLOCK_W + w_idx
            if oh < height_out and ow < width_out:
                tl.store(out_ptr + out_ptr_base + oh * width_in + ow, acc[h_idx, w_idx])


@triton.jit
def transposed_conv2d_kernel_v3(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    height_in, width_in, height_out, width_out,
    kernel_size, stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Optimized transposed convolution kernel - final version.
    """
    # Compute output tile indices
    pid_bh = tl.program_id(0)
    pid_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Batch index
    pid_b = pid_bh // out_channels
    if pid_b >= batch_size:
        return
    
    pid_out = pid_bh % out_channels
    
    # Output positions
    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Loop over kernel positions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input position for each output position
                # ih = oh*stride + kh*dilation - padding
                out_h = tl.arange(0, BLOCK_H) + out_h_start
                out_w = tl.arange(0, BLOCK_W) + out_w_start
                
                out_h_grid, out_w_grid = tl.meshgrid(out_h, out_w)
                out_h_grid = out_h_grid.T
                out_w_grid = out_w_grid.T
                
                ih = out_h_grid * stride + kh * dilation - padding
                iw = out_w_grid * stride + kw * dilation - padding
                
                # Valid mask
                valid = (ih >= 0) & (ih < height_in) & (iw >= 0) & (iw < width_in)
                
                # Load input values
                x_offset = pid_b * (in_channels * height_in * width_in) + \
                          in_c * (height_in * width_in) + \
                          ih * width_in + iw
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                
                # Load weight
                w_offset = in_c * (out_channels * kernel_size * kernel_size) + \
                          pid_out * (kernel_size * kernel_size) + \
                          kh * kernel_size + kw
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc = tl.where(valid, acc + x_val * w_val, acc)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out)
        acc = acc + bias
    
    # Store result
    out_offset = pid_b * (out_channels * height_out * width_out) + \
                pid_out * (height_out * width_out) + \
                out_h_start * width_out + out_w_start
    
    # Store with bounds checking
    for h_idx in range(BLOCK_H):
        for w_idx in range(BLOCK_W):
            oh = out_h_start + h_idx
            ow = out_w_start + w_idx
            if oh < height_out and ow < width_out:
                tl.store(out_ptr + out_offset + h_idx * width_out + w_idx, acc[h_idx, w_idx])


def triton_transposed_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """
    Performs 2D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor [batch, in_channels, H_in, W_in]
        weight: Weight tensor [in_channels, out_channels, K_h, K_w]
        bias: Optional bias [out_channels]
        stride: Stride
        padding: Padding
        dilation: Dilation
    
    Returns:
        Output tensor [batch, out_channels, H_out, W_out]
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height_in, width_in = x.shape
    _, out_channels, kernel_h, kernel_w = weight.shape
    kernel_size = kernel_h  # Assuming square kernel as per problem description
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_K = 8
    
    grid = lambda meta: (
        batch_size * out_channels,  # pid_bh
        out_channels,  # pid_out (for compatibility)
        (height_out + BLOCK_H - 1) // BLOCK_H,
        (width_out + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    transposed_conv2d_kernel_v3[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height_in, width_in, height_out, width_out,
        kernel_size, stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights (same as PyTorch's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_transposed_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )