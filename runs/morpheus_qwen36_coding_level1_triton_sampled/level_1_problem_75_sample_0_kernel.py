import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_n, stride_c_in, stride_h_in, stride_w_in,
    stride_n_out, stride_c_out, stride_h_out, stride_w_out,
    stride_w_c_out, stride_w_c_in, stride_w_kh, stride_w_kw,
    n_elements_x, n_elements_out,
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    has_bias,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Grid mapping
    pid = tl.program_id(0)
    num_blocks_h = (height_out + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (width_out + BLOCK_W - 1) // BLOCK_W
    
    block_idx_h = pid // num_blocks_w
    block_idx_w = pid % num_blocks_w
    
    # Determine the range of output H and W indices this program handles
    h_start = block_idx_h * BLOCK_H
    w_start = block_idx_w * BLOCK_W
    
    # Offsets for H and W within the block
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Masks for valid H and W indices
    mask_h = h_offsets < height_out
    mask_w = w_offsets < width_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Output channel stride is constant for the grid launch (each program handles one output channel)
    # However, we launch grid over (N, C_out, H_blocks, W_blocks)
    # We need to decode N and C_out from pid or pass them.
    # Better grid: (N, C_out, H_blocks, W_blocks) flattened.
    # Let's assume grid is 1D and we decode.
    # Actually, standard practice is grid = (N, C_out, H_blocks, W_blocks).
    # But Triton grid is usually 1D or 2D.
    # Let's use 1D grid and decode.
    # num_programs = batch_size * out_channels * num_blocks_h * num_blocks_w
    # pid = tl.program_id(0)
    # This requires computing total size in the launcher.
    
    # For simplicity in kernel, let's assume we pass batch and channel indices or decode.
    # Decoding is safer.
    # total_blocks = num_blocks_h * num_blocks_w
    # pid_hw = pid % total_blocks
    # pid_nc = pid // total_blocks
    # block_idx_h = pid_hw // num_blocks_w
    # block_idx_w = pid_hw % num_blocks_w
    # n_idx = pid_nc // out_channels
    # c_out_idx = pid_nc % out_channels
    
    # Re-decode
    total_blocks = num_blocks_h * num_blocks_w
    pid_hw = pid % total_blocks
    pid_nc = pid // total_blocks
    
    block_idx_h = pid_hw // num_blocks_w
    block_idx_w = pid_hw % num_blocks_w
    n_idx = pid_nc // out_channels
    c_out_idx = pid_nc % out_channels
    
    h_start = block_idx_h * BLOCK_H
    w_start = block_idx_w * BLOCK_W
    
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    mask_h = h_offsets < height_out
    mask_w = w_offsets < width_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Determine group for this output channel
    channels_per_group = in_channels // groups
    group_idx = c_out_idx // channels_per_group
    c_in_start = group_idx * channels_per_group
    
    # Loop over input channels in blocks
    for c_in_base in range(c_in_start, c_in_start + channels_per_group, BLOCK_C_IN):
        c_in_offsets = c_in_base + tl.arange(0, BLOCK_C_IN)
        mask_c_in = c_in_offsets < in_channels
        
        # Load weights for current output channel and input channel block
        # Weights shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
        # Access: w_ptr + c_out_idx * stride_w_c_out + c_in * stride_w_c_in + kh * stride_w_kh + kw * stride_w_kw
        # We can load weights for all kh, kw for the current c_in block.
        # This might be large, but we can loop or load in chunks.
        # For simplicity, load weights into registers if small, or loop.
        # Let's loop over kh and kw to keep memory pressure low.
        
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input spatial indices for each output element
                # h_in = h * stride_h + kh * dilation_h - padding_h
                # w_in = w * stride_w + kw * dilation_w - padding_w
                
                h_in_offsets = h_offsets * stride_h + kh * dilation_h - padding_h
                w_in_offsets = w_offsets * stride_w + kw * dilation_w - padding_w
                
                # Mask for valid input indices
                mask_h_in = (h_in_offsets >= 0) & (h_in_offsets < height_in)
                mask_w_in = (w_in_offsets >= 0) & (w_in_offsets < width_in)
                mask_input = mask_h_in[:, None] & mask_w_in[None, :] & mask_hw
                
                # Load input data
                # x_ptr offset: n * stride_n + c_in * stride_c_in + h_in * stride_h_in + w_in * stride_w_in
                # We need to compute offsets for each c_in in the block.
                # This is complex for vectorized load.
                # Instead, we can load x slice for each c_in or loop c_in.
                # Since we are inside kh, kw loop, we can load x for the c_in block.
                # x shape: (N, C_in, H_in, W_in)
                # We need x[n, c_in, h_in, w_in]
                # This is scattered in memory relative to c_in.
                # We can load x for the c_in block by iterating or using advanced indexing.
                # Triton supports loading tensors with offsets.
                # We can compute the base pointer for x and add offsets.
                
                # Base pointer for x: n_idx * stride_n
                # We need to access c_in offsets.
                # x_ptr + n_idx * stride_n + c_in_offsets * stride_c_in + ...
                # This requires broadcasting.
                
                # Load x for the c_in block
                # x_ptrs = x_ptr + n_idx * stride_n + c_in_offsets[:, None] * stride_c_in + h_in_offsets[None, :] * stride_h_in + w_in_offsets[None, :] * stride_w_in
                # This creates a shape (BLOCK_C_IN, BLOCK_H, BLOCK_W) tensor of pointers.
                # We can load this.
                
                x_ptrs = x_ptr + n_idx * stride_n + c_in_offsets[:, None] * stride_c_in + h_in_offsets[None, :] * stride_h_in + w_in_offsets[None, :] * stride_w_in
                x_vals = tl.load(x_ptrs, mask=mask_c_in[:, None] & mask_input, other=0.0)
                
                # Load weights for current kh, kw
                # w_ptrs = w_ptr + c_out_idx * stride_w_c_out + c_in_offsets * stride_w_c_in + kh * stride_w_kh + kw * stride_w_kw
                # Shape: (BLOCK_C_IN,)
                w_ptrs = w_ptr + c_out_idx * stride_w_c_out + c_in_offsets * stride_w_c_in + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptrs, mask=mask_c_in, other=0.0)
                
                # Multiply and accumulate
                # x_vals shape: (BLOCK_C_IN, BLOCK_H, BLOCK_W)
                # w_vals shape: (BLOCK_C_IN,)
                # We need to multiply w_vals[:, None, None] * x_vals
                acc += w_vals[:, None, None] * x_vals
    
    # Add bias if present
    if has_bias:
        # b_ptr + c_out_idx
        b_val = tl.load(b_ptr + c_out_idx)
        acc += b_val
    
    # Store result
    # out_ptr offset: n * stride_n_out + c * stride_c_out + h * stride_h_out + w * stride_w_out
    out_ptrs = out_ptr + n_idx * stride_n_out + c_out_idx * stride_c_out + h_offsets[:, None] * stride_h_out + w_offsets[None, :] * stride_w_out
    tl.store(out_ptrs, acc, mask=mask_hw)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups
        self.has_bias = bias
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get parameters from the underlying layer or use stored ones
        # Use stored ones for constants in kernel
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_h, self.kernel_w
        stride_h, stride_w = self.stride_h, self.stride_w
        padding_h, self.padding_w = self.padding_h, self.padding_w
        dilation_h, dilation_w = self.dilation_h, self.dilation_w
        groups = self.groups
        has_bias = self.has_bias
        
        # Get weights and bias
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if has_bias else None
        
        # Ensure contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
            
        # Output shape calculation
        batch_size, _, height_in, width_in = x.shape
        height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1
        width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1
        
        out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=x.dtype, device=x.device)
        
        # Strides
        stride_n, stride_c_in, stride_h_in, stride_w_in = x.stride()
        stride_n_out, stride_c_out, stride_h_out, stride_w_out = out.stride()
        stride_w_c_out, stride_w_c_in, stride_w_kh, stride_w_kw = weight.stride()
        
        # Block sizes
        BLOCK_H = 32
        BLOCK_W = 32
        BLOCK_C_IN = 32
        
        # Grid size
        num_blocks_h = (height_out + BLOCK_H - 1) // BLOCK_H
        num_blocks_w = (width_out + BLOCK_W - 1) // BLOCK_W
        num_programs = batch_size * out_channels * num_blocks_h * num_blocks_w
        
        # Launch kernel
        conv_transpose2d_kernel[(num_programs,)](
            x, weight, bias, out,
            stride_n, stride_c_in, stride_h_in, stride_w_in,
            stride_n_out, stride_c_out, stride_h_out, stride_w_out,
            stride_w_c_out, stride_w_c_in, stride_w_kh, stride_w_kw,
            x.numel(), out.numel(),
            batch_size, in_channels, out_channels,
            height_in, width_in,
            height_out, width_out,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            has_bias,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            BLOCK_C_IN=BLOCK_C_IN,
        )
        
        return out