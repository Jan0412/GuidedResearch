import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_width_kernel(
    x_ptr,  # Input tensor pointer: [B, C, H, W]
    w_ptr,  # Weight tensor pointer: [C, K_w] (K_h=1)
    b_ptr,  # Bias pointer: [C] (optional)
    out_ptr,  # Output tensor pointer: [B, C, H, W_out]
    B, C, H, W,  # Input dimensions
    W_out,  # Output width dimension
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w_block = tl.program_id(2)
    
    # Calculate output width index range
    w_start = pid_w_block * BLOCK_SIZE_W
    
    # Create offsets for output width
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_offsets < W_out
    
    # Calculate input width positions for this output position
    # For depthwise conv: each channel has its own kernel
    # Input position corresponding to output position w
    # x_pos = w * stride - padding + k * dilation
    w_input_start = w_offsets * stride - padding
    
    # Create kernel offsets [0, kernel_size)
    k_offsets = tl.arange(0, BLOCK_SIZE_C) if BLOCK_SIZE_C >= kernel_size else tl.arange(0, kernel_size)
    
    # Loop over channels in blocks
    for c_start in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        mask_c = c_offsets < C
        
        # Load input data: [B, C, H, W]
        # We need to access x[pid_batch, c_offsets, pid_h, w_input + k*dilation]
        # For each output position, we need kernel_size input positions
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate input width position for this kernel element
            w_input_k = w_input_start + k * dilation
            
            # Mask for valid input positions
            mask_valid = (w_input_k >= 0) & (w_input_k < W)
            
            # Create full mask
            full_mask = mask_c[:, None] & mask_valid[None, :]
            
            # Load input values: shape [BLOCK_SIZE_C, BLOCK_SIZE_W]
            # Need to flatten and reorganize for efficient loading
            # Get input indices: [B, C, H, W]
            # For batch pid_batch, height pid_h, channels c_offsets, widths w_input_k
            
            # Compute flat offset for x[pid_batch, c, pid_h, w_input_k]
            # x_ptr offset = ((pid_batch * C + c) * H + pid_h) * W + w_input_k
            if BLOCK_SIZE_C >= kernel_size:
                # Process all channels at once for this kernel position
                x_offsets = (((pid_batch * C + c_offsets[:, None]) * H + pid_h) * W + w_input_k[None, :])
                x_vals = tl.load(x_ptr + x_offsets, mask=full_mask, other=0.0)
                
                # Load weights: w[c, k]
                w_offsets_k = c_offsets * kernel_size + k
                w_vals = tl.load(w_ptr + w_offsets_k, mask=mask_c, other=0.0)
                
                # Accumulate: x_vals shape [C, W_out_block], w_vals shape [C]
                # Need to multiply each row of x_vals by corresponding w_vals
                acc += tl.sum(x_vals * w_vals[:, None], axis=1)
            else:
                # Process channels in smaller blocks
                for c_idx in range(BLOCK_SIZE_C):
                    if c_start + c_idx < C:
                        c = c_start + c_idx
                        x_offset = ((pid_batch * C + c) * H + pid_h) * W + w_input_k
                        x_vals = tl.load(x_ptr + x_offset, mask=mask_valid, other=0.0)
                        
                        w_offset = c * kernel_size + k
                        w_val = tl.load(w_ptr + w_offset)
                        
                        acc = tl.where(c_offsets == c_idx, acc + x_vals * w_val, acc)
        
        # Add bias if present
        if HAS_BIAS:
            bias_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
            mask_bias = bias_offset < C
            bias_vals = tl.load(b_ptr + bias_offset, mask=mask_bias, other=0.0)
            acc += bias_vals
        
        # Store output
        out_offsets = (((pid_batch * C + c_offsets) * H + pid_h) * W_out + w_offsets)
        mask_out = mask_c[:, None] & mask_w[None, :]
        
        if BLOCK_SIZE_C >= BLOCK_SIZE_W:
            # Reshape acc to match expected shape for storage
            acc_reshaped = acc[:, None]  # [C, 1]
            out_vals = acc_reshaped.flatten()
            out_offsets_flat = out_offsets.flatten()
            mask_flat = mask_out.flatten()
            
            tl.store(out_ptr + out_offsets_flat, out_vals, mask=mask_flat)
        else:
            # Store in chunks
            for c_idx in range(min(BLOCK_SIZE_C, C - c_start)):
                c = c_start + c_idx
                for w_idx in range(min(BLOCK_SIZE_W, W_out - w_start)):
                    if w_start + w_idx < W_out:
                        out_offset = ((pid_batch * C + c) * H + pid_h) * W_out + (w_start + w_idx)
                        tl.store(out_ptr + out_offset, acc[c_idx])


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Triton implementation of depthwise 2D convolution with kernel (kernel_size, 1)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    B, C, H, W = x.shape
    kernel_size = weight.shape[2]  # weight shape: [C, 1, kernel_size, 1] -> actually [C, kernel_size] for our case
    
    # Calculate output dimensions
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    H_out = H  # Since kernel height is 1 and no padding/dilation/stride on height
    
    # Prepare output tensor
    out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions
    # grid = (batch_size, height, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_C = 32
    
    grid = lambda meta: (B, H, (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"])
    
    # Launch kernel
    depthwise_conv1d_width_kernel[grid](
        x, weight.view(C, kernel_size), bias,
        out,
        B, C, H, W,
        W_out,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the depthwise convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight with shape [in_channels, kernel_size]
        # Note: original nn.Conv2d has shape [out_channels, in_channels/groups, kH, kW]
        # For depthwise with groups=in_channels: shape is [in_channels, 1, kernel_size, 1]
        # We'll store it as [in_channels, kernel_size] for our kernel
        self.weight = nn.Parameter(torch.Tensor(in_channels, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias,
                                      stride=self.stride, 
                                      padding=self.padding, 
                                      dilation=self.dilation)