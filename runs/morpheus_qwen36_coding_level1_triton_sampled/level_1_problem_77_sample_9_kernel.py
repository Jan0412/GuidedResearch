import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, C_in, D_in, H_in, W_in,
    C_out, kD, kH, kW,
    stride, padding, dilation,
    D_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr
):
    # Decode program ID to output coordinates
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    d = tl.program_id(2)
    h = tl.program_id(3)
    w = tl.program_id(4)
    
    acc = 0.0
    
    # Precompute base offsets for weights and inputs
    w_base = c_out * C_in * kD * kH * kW
    x_base = n * C_in * D_in * H_in * W_in
    
    # Precompute base spatial offsets for input
    x_d_base = d * stride - padding
    x_h_base = h * stride - padding
    x_w_base = w * stride - padding
    
    # Loop over kernel dimensions
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input spatial coordinates
                x_d = x_d_base + kd * dilation
                x_h = x_h_base + kh * dilation
                x_w = x_w_base + kw * dilation
                
                # Check bounds for input tensor
                if x_d >= 0 and x_d < D_in and x_h >= 0 and x_h < H_in and x_w >= 0 and x_w < W_in:
                    # Vectorized load over channel dimension c_in
                    offsets_c = tl.arange(0, BLOCK_SIZE)
                    mask_c = offsets_c < C_in
                    
                    # Load weights: w is [C_out, C_in, kD, kH, kW]
                    # Index varies with c_in by kD * kH * kW
                    w_offset_base = kd * kH * kW + kh * kW + kw
                    w_idx = w_base + w_offset_base + offsets_c * (kD * kH * kW)
                    w_vals = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
                    
                    # Load inputs: x is [N, C_in, D_in, H_in, W_in]
                    # Index varies with c_in by D_in * H_in * W_in
                    x_offset_base = x_d * H_in * W_in + x_h * W_in + x_w
                    x_idx = x_base + x_offset_base + offsets_c * (D_in * H_in * W_in)
                    x_vals = tl.load(x_ptr + x_idx, mask=mask_c, other=0.0)
                    
                    # Accumulate dot product
                    acc += tl.sum(w_vals * x_vals)
    
    # Add bias if present
    if bias_ptr != 0:
        acc += tl.load(bias_ptr + c_out)
        
    # Store output
    # out is [N, C_out, D_out, H_out, W_out]
    out_idx = n * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None, 
                            stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton kernel for 3D transposed convolution.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kD, kH, kW = w.shape
    
    # Calculate output spatial dimensions
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kD - 1) + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kW - 1) + 1
    
    out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Grid configuration: one thread per output element
    grid = (N, C_out, D_out, H_out, W_out)
    
    # Tunable block size for vectorization over input channels
    BLOCK_SIZE = 16
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, w, bias if bias is not None else 0, out,
        N, C_in, D_in, H_in, W_in,
        C_out, kD, kH, kW,
        stride, padding, dilation,
        D_out, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )