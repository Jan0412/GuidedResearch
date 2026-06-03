import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def transposed_conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, H_in, W_in]
    w_ptr,  # [C_in, C_out // G, K_h, K_w] (transposed conv weight layout)
    bias_ptr,  # [C_out] (optional)
    out_ptr,  # [B, C_out, H_out, W_out]
    # Dimensions
    B, C_in, C_out, G,  # Batch, channels, groups
    H_in, W_in,  # Input height/width
    H_out, W_out,  # Output height/width
    K_h, K_w,  # Kernel height/width
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
    stride_b, stride_o_c, stride_o_h, stride_o_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Check bounds for output
    out_h_mask = out_h < H_out
    out_w_mask = out_w < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Compute which output channel group this block handles
    c_per_group = C_out // G
    group_id = pid_c // c_per_group
    local_c = pid_c % c_per_group
    
    # For each input channel
    for ic in range(0, C_in, BLOCK_SIZE_K):
        ic_range = ic + tl.arange(0, BLOCK_SIZE_K)
        ic_mask = ic_range < C_in
        
        # Broadcast input channel mask across height/width
        ic_mask_h = ic_mask[:, None] if BLOCK_SIZE_K > 1 else ic_mask
        ic_mask_w = ic_mask[None, :] if BLOCK_SIZE_K > 1 else ic_mask
        
        # Load input data: x[B, ic, :, :]
        in_h = (out_h * stride_h - pad_h + dil_h * tl.arange(0, K_h)[None, None, :]) // dil_h
        in_w = (out_w * stride_w - pad_w + dil_w * tl.arange(0, K_w)[None, None, :]) // dil_w
        
        # We need to iterate over kernel positions
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input positions for this kernel position
                in_h_pos = out_h * stride_h - pad_h + dil_h * kh
                in_w_pos = out_w * stride_w - pad_w + dil_w * kw
                
                in_h_mask = (in_h_pos >= 0) & (in_h_pos < H_in)
                in_w_mask = (in_w_pos >= 0) & (in_w_pos < W_in)
                
                # Load input values
                x_offset = pid_b * stride_x_b + ic_range[:, None, None] * stride_x_c + \
                          in_h_pos[None, :, :] * stride_x_h + in_w_pos[None, :, :] * stride_x_w
                
                # Create proper masks for x
                x_mask = (ic_mask[:, None, None] & 
                         in_h_mask[None, :, :] & 
                         in_w_mask[None, :, :])
                
                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
                
                # Load weight values
                w_offset = ic_range[:, None, None] * stride_w_ic + \
                          local_c * stride_w_oc + \
                          kh * stride_w_kh + \
                          kw * stride_w_kw
                
                w_mask = ic_mask[:, None, None]
                w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
                
                # Accumulate
                acc += tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offset = pid_c * stride_b
        bias_val = tl.load(bias_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    out_offset = pid_b * stride_o_b + pid_c * stride_o_c + \
                out_h[:, None] * stride_o_h + out_w[None, :] * stride_o_w
    out_mask = out_h_mask[:, None] & out_w_mask[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_transposed_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    B, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_g, K_h, K_w = weight.shape
    C_out = C_out_per_g * groups
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_h - 1) + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_w - 1) + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Strides for x: [B, C_in, H_in, W_in]
    stride_x_b = x.stride(0)
    stride_x_c = x.stride(1)
    stride_x_h = x.stride(2)
    stride_x_w = x.stride(3)
    
    # Strides for weight: [C_in, C_out // G, K_h, K_w]
    stride_w_ic = weight.stride(0)
    stride_w_oc = weight.stride(1)
    stride_w_kh = weight.stride(2)
    stride_w_kw = weight.stride(3)
    
    # Strides for output: [B, C_out, H_out, W_out]
    stride_o_b = out.stride(0)
    stride_o_c = out.stride(1)
    stride_o_h = out.stride(2)
    stride_o_w = out.stride(3)
    
    # Strides for bias: [C_out]
    stride_bias = bias.stride(0) if bias is not None else 0
    
    # Set up kernel launch configuration
    # We use a 4D grid: [B, C_out // BLOCK_SIZE_M, H_out // BLOCK_SIZE_H, W_out // BLOCK_SIZE_W]
    BLOCK_SIZE_M = 16  # Channels per block
    BLOCK_SIZE_N = 1   # Batch per block (typically 1 for memory efficiency)
    BLOCK_SIZE_K = 8   # Input channels per block
    BLOCK_SIZE_H = 4   # Output height per block
    BLOCK_SIZE_W = 4   # Output width per block
    
    grid = (
        B,
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        H_in, W_in, H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
        stride_o_b, stride_o_c, stride_o_h, stride_o_w,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register the convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create the weight and bias parameters
        # Note: PyTorch uses [in_channels, out_channels // groups, *kernel_size] for ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom Triton kernel.
        """
        return triton_transposed_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )