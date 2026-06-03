import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,                # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,                # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,                # Bias tensor: (C_out,) or None
    out_ptr,              # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out, 
    H_in, W_in, 
    H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    # Meta-parameters
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    out_c_id = tl.program_id(1)
    out_h_id = tl.program_id(2)
    out_w_id = tl.program_id(3)
    
    # Output pointers
    out_offset = (batch_id * C_out * H_out * W_out + 
                  out_c_id * H_out * W_out + 
                  out_h_id * W_out + 
                  out_w_id)
    
    # Accumulate over C_in, K_h, K_w
    acc = 0.0
    
    # Iterate over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Load input values for this batch and input channels
        # For each input channel, we need to find which input positions contribute to this output
        # In transposed convolution, each output (b, oc, oh, ow) depends on:
        # input positions: ih = oh - pad_h - (kh - 1) * dil_h - stride_h * k_offset
        #                  iw = ow - pad_w - (kw - 1) * dil_w - stride_w * k_offset
        
        # For simplicity, iterate over kernel positions
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                ih = out_h_id - pad_h - kh * dil_h
                iw = out_w_id - pad_w - kw * dil_w
                
                # Check if this input position is valid
                if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                    # Load weights: w[c_in, c_out, kh, kw]
                    w_offset = (c_in_offsets * C_out * K_h * K_w + 
                               out_c_id * K_h * K_w + 
                               kh * K_w + 
                               kw)
                    w_vals = tl.load(w_ptr + w_offset, mask=c_in_mask, other=0.0)
                    
                    # Load inputs: x[b, c_in, ih, iw]
                    x_offset = (batch_id * C_in * H_in * W_in + 
                               c_in_offsets * H_in * W_in + 
                               ih * W_in + 
                               iw)
                    x_vals = tl.load(x_ptr + x_offset, mask=c_in_mask, other=0.0)
                    
                    # Accumulate: x * w
                    acc += tl.sum(x_vals * w_vals)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_id)
        acc += bias
    
    # Store output
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 2D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Set kernel block sizes (tunable parameters)
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_C_IN = 8
    BLOCK_SIZE_K = 1
    
    # Define grid: (batch, out_channels, out_height, out_width)
    grid = (B, C_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Register hyperparameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias using PyTorch's default initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias,
                                      stride=self.stride, 
                                      padding=self.padding, 
                                      dilation=self.dilation)