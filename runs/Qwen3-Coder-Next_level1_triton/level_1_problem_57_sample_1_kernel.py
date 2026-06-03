import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, h, w)
    w_ptr,  # Weight tensor (in_channels, out_channels, k_h, k_w)
    b_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (batch, out_channels, h_out, w_out)
    batch_size, in_channels, out_channels, 
    h_in, w_in, h_out, w_out,
    k_h, k_w,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 8,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)
    
    # Compute output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    h_mask = out_h < h_out
    w_mask = out_w < w_out
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_start in range(0, in_channels, BLOCK_SIZE_C):
        c_in_end = tl.minimum(c_in_start + BLOCK_SIZE_C, in_channels)
        
        # Compute input positions for this output position
        # For transposed convolution: input_h = (out_h - output_padding - k_h + 1 + stride * h_offset) / stride
        # But more efficiently: out_h = (input_h - 1) * stride - 2*padding + output_padding + k_h
        
        # Iterate over input positions that contribute to this output
        for kh in range(k_h):
            input_h = out_h - kh + padding - output_padding
            input_h = input_h[None, :]  # Make compatible with broadcasting
            
            # Check if input_h is valid (non-negative and within bounds)
            h_valid = (input_h >= 0) & (input_h < h_in) & (input_h % stride == 0)
            input_h = input_h // stride
            
            for kw in range(k_w):
                input_w = out_w - kw + padding - output_padding
                input_w = input_w[:, None]  # Make compatible with broadcasting
                
                # Check if input_w is valid
                w_valid = (input_w >= 0) & (input_w < w_in) & (input_w % stride == 0)
                input_w = input_w // stride
                
                # Combined validity mask
                valid_mask = h_valid & w_valid
                
                # Load input values
                # We need to handle the case where input_h and input_w might be out of bounds
                in_h_ptr = x_ptr + pid_b * (in_channels * h_in * w_in) + \
                          tl.arange(0, BLOCK_SIZE_H)[:, None] * (w_in * h_in) + \
                          input_h[:, :, None] * w_in + input_w[:, :, None]
                
                # Load weight values
                w_ptr_offset = c_in_start + tl.arange(0, BLOCK_SIZE_C)[:, None, None]
                w_ptr_offset = w_ptr_offset % in_channels  # Handle modulo for last block
                
                # Reshape weight to (BLOCK_SIZE_C, k_h, k_w)
                weight_ptr = w_ptr + \
                            tl.arange(0, BLOCK_SIZE_C)[:, None, None] * (out_channels * k_h * k_w) + \
                            pid_c * (k_h * k_w) + \
                            kh * k_w + kw
                
                # Load input and weight
                input_val = tl.load(in_h_ptr, mask=valid_mask[:, :, None], other=0.0)
                weight_val = tl.load(weight_ptr, mask=tl.arange(0, BLOCK_SIZE_C)[:, None, None] < (c_in_end - c_in_start), other=0.0)
                
                # Accumulate: acc += input * weight
                if c_in_start == 0:
                    acc += tl.sum(input_val * weight_val, axis=0)
                else:
                    acc += tl.sum(input_val * weight_val, axis=0)
    
    # Load bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c)
        acc += bias
    
    # Store result
    out_ptr_offset = pid_b * (out_channels * h_out * w_out) + \
                    pid_c * (h_out * w_out) + \
                    out_h[:, None] * w_out + out_w[None, :]
    
    tl.store(out_ptr + out_ptr_offset, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Triton-based transposed 2D convolution.
    """
    batch_size, in_channels, h_in, w_in = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Compute output dimensions
    h_out = (h_in - 1) * stride - 2 * padding + output_padding + k_h
    w_out = (w_in - 1) * stride - 2 * padding + output_padding + k_h
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, h_out, w_out, device=x.device, dtype=x.dtype)
    
    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        triton.cdiv(h_out, meta["BLOCK_SIZE_H"]),
        triton.cdiv(w_out, meta["BLOCK_SIZE_W"]),
        out_channels // meta.get("BLOCK_SIZE_C", 1),
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        k_h, k_w,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        num_groups=groups,
        BLOCK_SIZE_H=8,
        BLOCK_SIZE_W=8,
        BLOCK_SIZE_C=4,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel using Triton.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )