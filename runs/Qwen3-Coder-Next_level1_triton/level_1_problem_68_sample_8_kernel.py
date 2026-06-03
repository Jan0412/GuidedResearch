import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, D_in, W_in, H_in]
    w_ptr,  # [C_in, C_out, K_d, K_w, K_h]
    bias_ptr,  # [C_out] (optional, can be nullptr)
    out_ptr,  # [B, C_out, D_out, W_out, H_out]
    # Dimensions
    B, C_in, D_in, W_in, H_in,
    C_out, K_d, K_w, K_h,
    D_out, W_out, H_out,
    # Stride parameters
    stride_d, stride_w, stride_h,
    pad_d, pad_w, pad_h,
    out_pad_d, out_pad_w, out_pad_h,
    # Strides in memory
    stride_x_b, stride_x_c, stride_x_d, stride_x_w, stride_x_h,
    stride_w_ci, stride_w_co, stride_w_kd, stride_w_kw, stride_w_kh,
    stride_out_b, stride_out_c, stride_out_d, stride_out_w, stride_out_h,
    # Block sizes
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get output indices
    out_c_index = tl.program_id(0)
    out_d = tl.program_id(1)
    out_w = tl.program_id(2)
    out_h = tl.program_id(3)
    batch_idx = tl.program_id(4)
    
    # Compute output channel range for this block
    out_c_offsets = out_c_index * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    out_c_mask = out_c_offsets < C_out
    
    # Accumulator for the output
    acc = tl.zeros([BLOCK_SIZE_C_OUT], dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_idx in range(C_in):
        for kd in range(K_d):
            for kw in range(K_w):
                for kh in range(K_h):
                    # Calculate corresponding input position
                    in_d = out_d - kd + pad_d
                    in_w = out_w - kw + pad_w
                    in_h = out_h - kh + pad_h
                    
                    # Check if input position is valid
                    if in_d >= 0 and in_d < D_in and in_w >= 0 and in_w < W_in and in_h >= 0 and in_h < H_in:
                        # Load input value
                        x_offset = (batch_idx * stride_x_b + 
                                   c_in_idx * stride_x_c + 
                                   in_d * stride_x_d + 
                                   in_w * stride_x_w + 
                                   in_h * stride_x_h)
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight value
                        w_offset = (c_in_idx * stride_w_ci + 
                                   out_c_offsets * stride_w_co + 
                                   kd * stride_w_kd + 
                                   kw * stride_w_kw + 
                                   kh * stride_w_kh)
                        w_val = tl.load(w_ptr + w_offset, mask=out_c_mask, other=0.0)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Apply bias if available
    if bias_ptr is not None:
        bias_offsets = out_c_offsets
        bias_val = tl.load(bias_ptr + bias_offsets, mask=out_c_mask, other=0.0)
        acc += bias_val
    
    # Store result
    out_offset = (batch_idx * stride_out_b + 
                 out_c_offsets * stride_out_c + 
                 out_d * stride_out_d + 
                 out_w * stride_out_w + 
                 out_h * stride_out_h)
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=out_c_mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of ConvTranspose3d for groups=1
    """
    assert groups == 1, "Only groups=1 is supported in this implementation"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, W_in, H_in = x.shape
    C_in_w, C_out, K_d, K_w, K_h = weight.shape
    assert C_in == C_in_w, f"Input channels mismatch: {C_in} vs {C_in_w}"
    
    # Compute output dimensions
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    out_pad_d, out_pad_w, out_pad_h = output_padding
    
    D_out = (D_in - 1) * stride_d - 2 * pad_d + K_d + out_pad_d
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, W_out, H_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_SIZE_C_OUT = 32
    BLOCK_SIZE_K = 8
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(C_out, BLOCK_SIZE_C_OUT),  # out_c blocks
        D_out,                                 # out_d
        W_out,                                 # out_w
        H_out,                                 # out_h
        B                                      # batch
    )
    
    # Calculate strides
    stride_x_b, stride_x_c, stride_x_d, stride_x_w, stride_x_h = x.stride()
    stride_w_ci, stride_w_co, stride_w_kd, stride_w_kw, stride_w_kh = weight.stride()
    stride_out_b, stride_out_c, stride_out_d, stride_out_w, stride_out_h = out.stride()
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D_in, W_in, H_in,
        C_out, K_d, K_w, K_h,
        D_out, W_out, H_out,
        stride_d, stride_w, stride_h,
        pad_d, pad_w, pad_h,
        out_pad_d, out_pad_w, out_pad_h,
        stride_x_b, stride_x_c, stride_x_d, stride_x_w, stride_x_h,
        stride_w_ci, stride_w_co, stride_w_kd, stride_w_kw, stride_w_kh,
        stride_out_b, stride_out_c, stride_out_d, stride_out_w, stride_out_h,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton kernel for ConvTranspose3d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weight and bias parameters"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # same as PyTorch default
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution using Triton kernel.
        """
        # For groups=1, use Triton kernel
        if self.groups == 1:
            return triton_conv_transpose3d(
                x, self.weight, self.bias,
                stride=self.stride, 
                padding=self.padding, 
                output_padding=self.output_padding,
                groups=self.groups
            )
        else:
            # Fallback to PyTorch for groups > 1 (not implemented in Triton kernel yet)
            return nn.functional.conv_transpose3d(
                x, self.weight, self.bias,
                stride=self.stride, 
                padding=self.padding, 
                output_padding=self.output_padding,
                groups=self.groups
            )