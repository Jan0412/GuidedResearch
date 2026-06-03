import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor (batch, in_channels, H, W)
    w_ptr,  # Weight tensor (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor (out_channels,) - can be None
    out_ptr,  # Output tensor (batch, out_channels, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride, padding, output_padding,
    # Meta-parameters
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
    BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch
    pid_oc = tl.program_id(1)  # output channel block
    pid_h = tl.program_id(2)  # output height tile
    pid_w = tl.program_id(3)  # output width tile

    # Calculate output tile positions
    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W
    
    # Create output tile offsets
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_W)
    
    # Create output tile masks
    out_h_mask = out_h_offsets < out_h
    out_w_mask = out_w_offsets < out_w
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(0, in_channels, BLOCK_IC):
        for kh in range(0, k_h, BLOCK_KH):
            for kw in range(0, k_w, BLOCK_KW):
                # Calculate input position from output position
                # For transposed convolution: input_pos = (output_pos - (kernel_pos - 1 - padding)) // stride
                in_h_start = out_h_start - (kh - padding)
                in_w_start = out_w_start - (kw - padding)
                
                # Calculate input tile offsets
                in_h_offsets = in_h_start + out_h_offsets * stride
                in_w_offsets = in_w_start + out_w_offsets * stride
                
                # Check if input positions are valid
                in_h_valid = (in_h_offsets >= 0) & (in_h_offsets < in_h)
                in_w_valid = (in_w_offsets >= 0) & (in_w_offsets < in_w)
                
                # Create masks for input and output
                input_mask = (in_h_valid[:, None] & in_w_valid[None, :])
                output_mask = (out_h_mask[:, None] & out_w_mask[None, :])
                valid_mask = input_mask & output_mask
                
                # Load input values
                in_h_idx = in_h_offsets[:, None]
                in_w_idx = in_w_offsets[None, :]
                
                # Calculate input pointer offset
                input_offset = (pid_b * in_channels * in_h * in_w + 
                              ic * in_h * in_w + 
                              in_h_idx * in_w + in_w_idx)
                
                # Load input with mask
                x_val = tl.load(x_ptr + input_offset, mask=valid_mask, other=0.0)
                
                # Load weight values
                weight_offset = (ic * out_channels * k_h * k_w + 
                               pid_oc * BLOCK_OC * k_h * k_w + 
                               kh * k_w + kw)
                
                # We need to load weights for the output channel block
                # For simplicity, load one weight value at a time
                w_val = tl.load(w_ptr + weight_offset + tl.arange(0, BLOCK_OC) * k_h * k_w, 
                              mask=(pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC) < out_channels), 
                              other=0.0)
                
                # Accumulate: x * w for this channel and kernel position
                # Reshape for broadcasting
                x_val_expanded = x_val[:, :, None]  # (BLOCK_H, BLOCK_W, 1)
                w_val_expanded = w_val[None, None, :]  # (1, 1, BLOCK_OC)
                
                # Only accumulate if we're within the actual kernel bounds
                kh_valid = kh + tl.arange(0, BLOCK_KH) < k_h
                kw_valid = kw + tl.arange(0, BLOCK_KW) < k_w
                
                # For this implementation, we'll do a simpler approach
                # Load the specific weight for this kernel position
                w_single = tl.load(w_ptr + (ic * out_channels + pid_oc * BLOCK_OC) * k_h * k_w + 
                                 kh * k_w + kw,
                                 mask=pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC) < out_channels,
                                 other=0.0)
                
                # Accumulate
                acc += x_val * w_single[None, None, :] * valid_mask[:, :, None]
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC), 
                      mask=pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC) < out_channels)
        acc += bias[None, None, :]
    
    # Store result
    out_offset = (pid_b * out_channels * out_h * out_w + 
                 pid_oc * BLOCK_OC * out_h * out_w + 
                 out_h_offsets[:, None] * out_w + out_w_offsets[None, :])
    
    # Store with proper mask
    tl.store(out_ptr + out_offset, acc[:, :, 0], mask=output_mask)


# A more practical implementation using matrix multiplication approach
@triton.jit
def conv_transpose2d_simple_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w, stride, padding, output_padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    # This is a more practical implementation that handles the transposed convolution
    # by iterating over output positions and accumulating contributions from input and kernel
    
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Output position
    out_h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    out_h_mask = out_h_idx < out_h
    out_w_mask = out_w_idx < out_w
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels
    for ic in range(in_channels):
        # Iterate over kernel positions
        for kh in range(k_h):
            for kw in range(k_w):
                # Calculate corresponding input position
                in_h_idx = (out_h_idx - (kh - padding)) // stride
                in_w_idx = (out_w_idx - (kw - padding)) // stride
                
                # Check validity
                h_valid = (in_h_idx >= 0) & (in_h_idx < in_h) & ((out_h_idx - (kh - padding)) % stride == 0)
                w_valid = (in_w_idx >= 0) & (in_w_idx < in_w) & ((out_w_idx - (kw - padding)) % stride == 0)
                
                valid = h_valid[:, None] & w_valid[None, :]
                
                # Get input value
                in_pos = pid_b * in_channels * in_h * in_w + ic * in_h * in_w + in_h_idx * in_w + in_w_idx
                x_val = tl.load(x_ptr + in_pos, mask=valid, other=0.0)
                
                # Get weight value
                # Weight shape: (in_channels, out_channels, k_h, k_w)
                w_pos = ic * out_channels * k_h * k_w + pid_oc * k_h * k_w + kh * k_w + kw
                w_val = tl.load(w_ptr + w_pos)
                
                acc += x_val * w_val * valid
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_oc)
        acc += bias
    
    # Store output
    out_pos = pid_b * out_channels * out_h * out_w + pid_oc * out_h * out_w + out_h_idx * out_w + out_w_idx
    tl.store(out_ptr + out_pos, acc, mask=out_h_mask[:, None] & out_w_mask[None, :])


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of transposed 2D convolution
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_h, in_w = x.shape
    _, out_channels, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride - 2 * padding + k_h + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + k_w + output_padding
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_OC = 8
    BLOCK_IC = 8
    
    # Calculate grid dimensions
    grid = (batch_size, 
            (out_channels + BLOCK_OC - 1) // BLOCK_OC,
            (out_h + BLOCK_H - 1) // BLOCK_H,
            (out_w + BLOCK_W - 1) // BLOCK_W)
    
    # Launch kernel
    conv_transpose2d_simple_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w, stride, padding, output_padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with same parameters but we'll use our custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (same as PyTorch default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using our custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding,
            groups=self.groups
        )


import math