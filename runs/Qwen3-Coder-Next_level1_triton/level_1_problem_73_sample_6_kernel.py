import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_conv_transpose3d_kernel(
    # Pointers to tensors
    X,  # Input tensor: (B, C_in, D, H, W)
    W,  # Weight tensor: (C_in, C_out // G, Kd, Kh, Kw)
    B,  # Bias tensor: (C_out,) - optional
    Y,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B_size, C_in, D_in, H_in, W_in,
    C_out, Kd, Kh, Kw,
    D_out, H_out, W_out,
    stride, padding, groups,
    # Meta-parameters
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block_id = tl.program_id(1)
    d_id = tl.program_id(2)
    h_id = tl.program_id(3)
    w_id = tl.program_id(4)
    
    # Initialize output accumulator
    c_out_start = c_out_block_id * BLOCK_C_OUT
    output_sum = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Process over input channels
    for c_in_start in range(0, C_in, BLOCK_C_IN):
        # Process over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Compute input position
                    d_in = d_id * stride + kd - padding
                    h_in = h_id * stride + kh - padding
                    w_in = w_id * stride + kw - padding
                    
                    # Check if input position is valid
                    valid = (d_in >= 0) & (d_in < D_in) & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                    
                    # Compute input pointer offset
                    if valid:
                        # Get input value
                        x_ptr = X + batch_id * (C_in * D_in * H_in * W_in) + \
                                tl.arange(0, BLOCK_C_IN)[None, :] * (D_in * H_in * W_in) + \
                                d_in * (H_in * W_in) + \
                                h_in * W_in + \
                                w_in
                        
                        # Get weight value for each output channel
                        w_ptr = W + (c_in_start + tl.arange(0, BLOCK_C_IN)[:, None]) * (C_out * Kd * Kh * Kw) + \
                                c_out_start * Kd * Kh * Kw + \
                                kd * (Kh * Kw) + \
                                kh * Kw + \
                                kw
                        
                        # Load data
                        x_val = tl.load(x_ptr, mask=(tl.arange(0, BLOCK_C_IN) < C_in - c_in_start)[None, :], other=0.0)
                        w_val = tl.load(w_ptr, mask=(tl.arange(0, BLOCK_C_IN) < C_in - c_in_start)[:, None], other=0.0)
                        
                        # Accumulate
                        output_sum += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if provided
    if B is not None:
        bias_ptr = B + c_out_start + tl.arange(0, BLOCK_C_OUT)
        bias = tl.load(bias_ptr, mask=(tl.arange(0, BLOCK_C_OUT) < C_out - c_out_start), other=0.0)
        output_sum += bias
    
    # Store result
    y_ptr = Y + batch_id * (C_out * D_out * H_out * W_out) + \
            c_out_start * (D_out * H_out * W_out) + \
            d_id * (H_out * W_out) + \
            h_id * W_out + \
            w_id
    
    tl.store(y_ptr, output_sum, mask=(tl.arange(0, BLOCK_C_OUT) < C_out - c_out_start))

def triton_conv_transpose3d(x, weight, bias, stride, padding, groups):
    """
    Custom Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Stride of the convolution
        padding: Padding applied to input
        groups: Number of groups for convolution
    
    Returns:
        Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
    """
    # Get dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_g, Kd, Kh, Kw = weight.shape
    assert C_in == C_in_w, f"Input channels must match: {C_in} vs {C_in_w}"
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride - 2 * padding + (Kd - 1) + 1 + 1
    H_out = (H_in - 1) * stride - 2 * padding + (Kh - 1) + 1 + 1
    W_out = (W_in - 1) * stride - 2 * padding + (Kw - 1) + 1 + 1
    
    # Ensure correct tensor layout
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    C_out = weight.shape[1] * groups
    y = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define kernel launch configuration
    BLOCK_B = 1
    BLOCK_C_OUT = 8
    BLOCK_C_IN = 8
    BLOCK_D = 1
    BLOCK_H = 1
    BLOCK_W = 1
    
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta["BLOCK_C_OUT"]),
        triton.cdiv(D_out, meta["BLOCK_D"]),
        triton.cdiv(H_out, meta["BLOCK_H"]),
        triton.cdiv(W_out, meta["BLOCK_W"]),
    )
    
    # Launch kernel
    triton_conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, D_in, H_in, W_in,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride, padding, groups,
        BLOCK_B=BLOCK_B,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return y

class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Create weight tensor
        # For group convolution, weight shape is (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        
        # Create bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.groups
        )