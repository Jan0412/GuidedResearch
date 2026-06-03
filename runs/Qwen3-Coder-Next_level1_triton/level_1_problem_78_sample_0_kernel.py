import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    x_ptr,           # Input tensor: (batch, in_channels, h, w)
    w_ptr,           # Weight tensor: (in_channels, out_channels, kh, kw)
    b_ptr,           # Bias tensor: (out_channels,) or None
    y_ptr,           # Output tensor: (batch, out_channels, h_out, w_out)
    batch_size, 
    in_channels, 
    out_channels,
    in_h, in_w,
    out_h, out_w,
    kh, kw,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    pid_g = tl.program_id(2)  # For output height groups (for tiling)

    # Calculate output channel range
    out_c_start = pid_m * BLOCK_SIZE_M
    out_c_range = tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_start + out_c_range < out_channels
    
    # Calculate batch index
    batch_idx = pid_n
    
    # Calculate output height tile
    tile_h_start = pid_g * BLOCK_SIZE_N
    tile_h_range = tl.arange(0, BLOCK_SIZE_N)
    tile_h_mask = tile_h_start + tile_h_range < out_h
    
    # Create output coordinate grid
    out_h_indices = tile_h_start + tile_h_range
    out_w_indices = tl.arange(0, out_w)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Calculate corresponding input positions for this output position
        # For transposed convolution: input_h = (output_h - pad_h) // stride_h
        in_h_start = (tile_h_start + pad_h) // stride_h
        in_h_offsets = tl.arange(0, BLOCK_SIZE_N)
        in_h_pos = in_h_start + in_h_offsets
        in_h_mask = (in_h_pos >= 0) & (in_h_pos < in_h)
        
        # Loop over kernel height
        for kh_idx in range(kh):
            # Calculate input height for this kernel position
            input_h = in_h_start - kh_idx + stride_h * in_h_offsets
            valid_h = (input_h >= 0) & (input_h < in_h)
            
            # Load input values (broadcasted)
            if kh_idx == 0:
                # For efficiency, we'll load input in chunks
                input_ptr = x_ptr + batch_idx * (in_channels * in_h * in_w) + \
                           in_c * (in_h * in_w) + input_h * in_w
                input_vals = tl.load(input_ptr, mask=valid_h[:, None], other=0.0)
            else:
                input_ptr = x_ptr + batch_idx * (in_channels * in_h * in_w) + \
                           in_c * (in_h * in_w) + input_h * in_w
                input_vals = tl.load(input_ptr, mask=valid_h[:, None], other=0.0)
            
            # Loop over kernel width
            for kw_idx in range(kw):
                # Get kernel weight
                kernel_w_pos = kw_idx
                kernel_h_pos = kh_idx
                
                # Calculate corresponding input width
                in_w_start = (tl.arange(0, out_w) + pad_w - kw_idx) // stride_w
                in_w_mask = (in_w_start >= 0) & (in_w_start < in_w)
                
                # Load input values for this kernel position
                input_ptr = x_ptr + batch_idx * (in_channels * in_h * in_w) + \
                           in_c * (in_h * in_w) + in_w_start * in_w
                input_vals = tl.load(input_ptr, mask=in_w_mask[None, :], other=0.0)
                
                # Load kernel weight
                kernel_ptr = w_ptr + in_c * (out_channels * kh * kw) + \
                            out_c_start * (kh * kw) + kernel_h_pos * kw + kernel_w_pos
                kernel_vals = tl.load(kernel_ptr + out_c_range * (kh * kw), 
                                     mask=out_c_mask, other=0.0)
                
                # Accumulate: output = sum(input * kernel)
                # Broadcast for matrix multiplication
                acc += input_vals * kernel_vals[None, :]
    
    # Add bias if present
    if b_ptr is not None:
        bias_ptr = b_ptr + out_c_start + out_c_range
        bias_vals = tl.load(bias_ptr, mask=out_c_mask, other=0.0)
        acc += bias_vals[None, :]
    
    # Store result
    y_ptr_offset = batch_idx * (out_channels * out_h * out_w) + \
                  out_c_start * (out_h * out_w) + \
                  (tile_h_start + tl.arange(0, BLOCK_SIZE_N)) * out_w + \
                  tl.arange(0, out_w)
    
    tl.store(y_ptr + y_ptr_offset, acc.to(y_ptr.dtype.element_ty), 
            mask=(tile_h_mask[:, None] & out_c_mask[None, :]))


# More efficient implementation using tiling and better memory access patterns
@triton.jit
def transposed_conv2d_kernel_optimized(
    x_ptr,           # Input tensor: (batch, in_channels, h, w)
    w_ptr,           # Weight tensor: (in_channels, out_channels, kh, kw)
    b_ptr,           # Bias tensor: (out_channels,) or None
    y_ptr,           # Output tensor: (batch, out_channels, h_out, w_out)
    batch_size, 
    in_channels, 
    out_channels,
    in_h, in_w,
    out_h, out_w,
    kh, kw,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr,
    BLOCK_SIZE_IN_C: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    
    # Calculate output channel range
    out_c_offset = pid_out_c * BLOCK_SIZE_OUT_C
    out_c_range = tl.arange(0, BLOCK_SIZE_OUT_C)
    out_c_mask = out_c_offset + out_c_range < out_channels
    
    # Calculate input channel range
    in_c_offset = 0
    in_c_range = tl.arange(0, BLOCK_SIZE_IN_C)
    in_c_mask = in_c_offset + in_c_range < in_channels
    
    # Output height and width indices
    out_h_range = tl.arange(0, out_h)
    out_w_range = tl.arange(0, out_w)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OUT_C, out_h, out_w), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for in_c_block in range(0, in_channels, BLOCK_SIZE_IN_C):
        in_c_block_range = in_c_offset + tl.arange(0, BLOCK_SIZE_IN_C)
        in_c_block_mask = in_c_block_range < in_channels
        
        # Process each input channel in this block
        for i in range(BLOCK_SIZE_IN_C):
            in_c_idx = in_c_block + i
            if in_c_idx >= in_channels:
                continue
                
            # Load input for this channel and batch
            x_offset = pid_batch * (in_channels * in_h * in_w) + \
                      in_c_idx * (in_h * in_w)
            
            # Loop over kernel height
            for kh_idx in range(kh):
                # Calculate output height range affected by this kernel row
                out_h_start = kh_idx - pad_h
                out_h_indices = tl.arange(0, out_h)
                in_h_indices = (out_h_indices - out_h_start) // stride_h
                valid_h = (in_h_indices >= 0) & (in_h_indices < in_h) & \
                         ((out_h_indices - out_h_start) % stride_h == 0)
                
                # Load input values
                x_data = tl.load(x_ptr + x_offset + in_h_indices * in_w, 
                               mask=valid_h[:, None], other=0.0)
                
                # Loop over kernel width
                for kw_idx in range(kw):
                    # Calculate output width range affected by this kernel col
                    out_w_start = kw_idx - pad_w
                    in_w_indices = (out_w_range - out_w_start) // stride_w
                    valid_w = (in_w_indices >= 0) & (in_w_indices < in_w) & \
                             ((out_w_range - out_w_start) % stride_w == 0)
                    
                    # Load input values
                    x_vals = tl.load(x_ptr + x_offset + in_h_indices[:, None] * in_w + in_w_indices[None, :], 
                                   mask=valid_h[:, None] & valid_w[None, :], other=0.0)
                    
                    # Load kernel weight
                    w_offset = in_c_idx * (out_channels * kh * kw) + \
                              out_c_offset * (kh * kw) + \
                              kh_idx * kw + kw_idx
                    w_vals = tl.load(w_ptr + w_offset + out_c_range * (kh * kw), 
                                   mask=out_c_mask, other=0.0)
                    
                    # Accumulate
                    acc += x_vals[None, :, :] * w_vals[:, None, None]
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_offset + out_c_range, mask=out_c_mask, other=0.0)
        acc += bias[:, None, None]
    
    # Store result
    y_offset = pid_batch * (out_channels * out_h * out_w) + \
              out_c_offset * (out_h * out_w)
    y_data = acc.to(y_ptr.dtype.element_ty)
    
    # Store with proper masking
    tl.store(y_ptr + y_offset + out_c_range[:, None, None] * (out_h * out_w) +
            out_h_range[None, :, None] * out_w + out_w_range[None, None, :],
            y_data, mask=out_c_mask[:, None, None] & 
            tl.full((BLOCK_SIZE_OUT_C, out_h, out_w), True, dtype=tl.bool))


# Even simpler and more efficient direct implementation
@triton.jit
def transposed_conv2d_direct_kernel(
    x_ptr,           # Input: (B, IC, H, W)
    w_ptr,           # Weight: (IC, OC, KH, KW)
    b_ptr,           # Bias: (OC,)
    y_ptr,           # Output: (B, OC, OH, OW)
    B, IC, OC, 
    IH, IW,
    OH, OW,
    KH, KW,
    SH, SW,
    PH, PW,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    oc_id = tl.program_id(1)
    
    # Compute output channel indices
    oc_offsets = oc_id * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < OC
    
    # Compute output position
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Compute input position for this output position
    ih_start = oh - PH
    iw_start = ow - PW
    
    # Check if this output position is valid
    valid_pos = (ih_start % SH == 0) and (iw_start % SW == 0)
    ih = ih_start // SH
    iw = iw_start // SW
    
    if valid_pos and ih >= 0 and ih < IH and iw >= 0 and iw < IW:
        # Loop over input channels
        for ic in range(IC):
            # Load input value
            x_offset = batch_id * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
            x_val = tl.load(x_ptr + x_offset)
            
            # Load kernel weights for all output channels
            w_offset = ic * (OC * KH * KW) + oc_offsets * (KH * KW) + KH * KW - 1 - ih_start - iw_start
            # Adjust kernel index: kernel[ic, oc, kh, kw] where kh = KH-1-ih_start, kw = KW-1-iw_start
            kh_idx = KH - 1 - (ih_start % SH)  # This is incorrect, need proper indexing
            kw_idx = KW - 1 - (iw_start % SW)
            
            # Correct kernel indexing for transposed convolution
            # In transposed conv: output[oh, ow] += input[ih, iw] * weight[ic, oc, kh, kw]
            # where ih = (oh - PH) // SH, iw = (ow - PW) // SW
            # and kh = oh - PH - ih * SH, kw = ow - PW - iw * SW
            
            kh_pos = ih_start - ih * SH  # Should be 0
            kw_pos = iw_start - iw * SW  # Should be 0
            
            # Load kernel weight at position (ic, oc, kh_pos, kw_pos)
            w_idx = ic * (OC * KH * KW) + oc_offsets * (KH * KW) + kh_pos * KW + kw_pos
            w_val = tl.load(w_ptr + w_idx, mask=oc_mask)
            
            acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + oc_offsets, mask=oc_mask)
        acc += b_val
    
    # Store result
    y_offset = batch_id * (OC * OH * OW) + oc_offsets * (OH * OW) + oh * OW + ow
    tl.store(y_ptr + y_offset, acc.to(y_ptr.dtype.element_ty), mask=oc_mask)


# Final optimized implementation
@triton.jit
def transposed_conv2d_fused_kernel(
    x_ptr,           # Input: (B, IC, H, W)
    w_ptr,           # Weight: (IC, OC, KH, KW)
    b_ptr,           # Bias: (OC,) or None
    y_ptr,           # Output: (B, OC, OH, OW)
    B, IC, OC, 
    IH, IW,
    OH, OW,
    KH, KW,
    SH, SW,
    PH, PW,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    oc_id = tl.program_id(1)
    ic_id = tl.program_id(2)
    
    # Compute ranges
    oc_offsets = oc_id * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < OC
    
    ic_offset = ic_id * BLOCK_SIZE_IC
    ic_range = tl.arange(0, BLOCK_SIZE_IC)
    ic_mask = ic_offset + ic_range < IC
    
    # For each output position
    for oh in range(OH):
        for ow in range(OW):
            # Compute corresponding input position
            ih = (oh - PH) // SH
            iw = (ow - PW) // SW
            
            # Check if valid
            valid = ((oh - PH) % SH == 0) and ((ow - PW) % SW == 0) and \
                   (ih >= 0) and (ih < IH) and (iw >= 0) and (iw < IW)
            
            if valid:
                # Initialize accumulator
                acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
                
                # Loop over input channels in this block
                for ic_block in range(BLOCK_SIZE_IC):
                    ic_idx = ic_offset + ic_block
                    if ic_idx >= IC:
                        continue
                    
                    # Load input value
                    x_idx = batch_id * (IC * IH * IW) + ic_idx * (IH * IW) + ih * IW + iw
                    x_val = tl.load(x_ptr + x_idx)
                    
                    # Load kernel weights for all output channels
                    # Kernel indices for transposed conv: 
                    # kh = oh - PH - ih * SH = 0 (since ih = (oh-PH)//SH and oh-PH = ih*SH)
                    # kw = ow - PW - iw * SW = 0
                    # So we want kernel[ic, oc, 0, 0] when position is valid
                    
                    kh_idx = oh - PH - ih * SH
                    kw_idx = ow - PW - iw * SW
                    
                    w_idx = ic_idx * (OC * KH * KW) + oc_offsets * (KH * KW) + kh_idx * KW + kw_idx
                    w_val = tl.load(w_ptr + w_idx, mask=oc_mask)
                    
                    acc += x_val * w_val
                
                # Add bias
                if b_ptr is not None:
                    b_val = tl.load(b_ptr + oc_offsets, mask=oc_mask)
                    acc += b_val
                
                # Store result
                y_idx = batch_id * (OC * OH * OW) + oc_offsets * (OH * OW) + oh * OW + ow
                tl.store(y_ptr + y_idx, acc.to(y_ptr.dtype.element_ty), mask=oc_mask)


def triton_transposed_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride tuple (height, width)
        padding: Padding tuple (height, width)
    
    Returns:
        Output tensor of shape (batch, out_channels, output_height, output_width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, IC, IH, IW = x.shape
    _, OC, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    
    # Calculate output dimensions
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW
    
    # Prepare output tensor
    y = torch.empty((B, OC, OH, OW), dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_OC = 8
    BLOCK_SIZE_IC = 4
    
    # Calculate grid dimensions
    grid = (B, (OC + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC, 
            (IC + BLOCK_SIZE_IC - 1) // BLOCK_SIZE_IC)
    
    # Launch kernel
    transposed_conv2d_fused_kernel[grid](
        x, weight, bias, y,
        B, IC, OC, IH, IW, OH, OW, KH, KW,
        SH, SW, PH, PW,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized Model with Triton-based transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters manually
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_transposed_conv2d(x, self.weight, self.bias, 
                                       self.stride, self.padding)