import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # Input tensor: (B, C_in, D, H, W)
    W_ptr,  # Weight tensor: (C_in, C_out // groups, Kd, Kh, Kw)
    B_ptr,  # Bias tensor: (C_out,)
    Y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out, groups,
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_Kd: tl.constexpr,
    BLOCK_SIZE_Kh: tl.constexpr,
    BLOCK_SIZE_Kw: tl.constexpr,
    # Strides for memory access
    stride_X_b, stride_X_c, stride_X_d, stride_X_h, stride_X_w,
    stride_W_cin, stride_W_cout, stride_W_kd, stride_W_kh, stride_W_kw,
    stride_Y_b, stride_Y_c, stride_Y_d, stride_Y_h, stride_Y_w,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Calculate input position that corresponds to this output position
    # For transposed convolution: in_d = (out_d - out_pad_d) // stride_d - pad_d
    # Only process if this is a valid input position
    in_d_start = out_d - out_pad_d
    in_h_start = out_h - out_pad_h
    in_w_start = out_w - out_pad_w
    
    # Initialize accumulator for this output element
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c_in_offset in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_range = c_in_offset + tl.arange(0, BLOCK_SIZE_C_in)
        mask_c_in = c_in_range < C_in
        
        # Get input value (if valid position)
        in_d = in_d_start + stride_d * tl.arange(0, 1)
        in_h = in_h_start + stride_h * tl.arange(0, 1)
        in_w = in_w_start + stride_w * tl.arange(0, 1)
        
        # We need to handle multiple kernel positions that contribute to this output
        # For transposed convolution, each output position receives contributions from
        # multiple input positions based on kernel size
        
    # Actually, let's restructure: for each output position, we need to accumulate
    # over all kernel positions and input channels
    # This is the standard approach for transposed convolution
    
    # Reset accumulator
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # For transposed convolution, each output position (out_d, out_h, out_w) is computed as:
    # sum over c_in, k_d, k_h, k_w of X[b, c_in, out_d - k_d, out_h - k_h, out_w - k_w] * W[c_in, c_out, k_d, k_h, k_w]
    # with appropriate handling of padding and strides
    
    # Loop over kernel dimensions
    for k_d in range(Kd):
        in_d = out_d - k_d
        if in_d >= 0 and in_d < D:
            for k_h in range(Kh):
                in_h = out_h - k_h
                if in_h >= 0 and in_h < H:
                    for k_w in range(Kw):
                        in_w = out_w - k_w
                        if in_w >= 0 and in_w < W:
                            # Load input value at position (b, c_in, in_d, in_h, in_w)
                            # and accumulate with corresponding kernel weights
                            for c_in_offset in range(0, C_in, BLOCK_SIZE_C_in):
                                c_in_range = c_in_offset + tl.arange(0, BLOCK_SIZE_C_in)
                                mask_c_in = c_in_range < C_in
                                
                                # Load input
                                x_offset = (pid_b * stride_X_b + 
                                           c_in_range * stride_X_c + 
                                           in_d * stride_X_d + 
                                           in_h * stride_X_h + 
                                           in_w * stride_X_w)
                                x = tl.load(X_ptr + x_offset, mask=mask_c_in, other=0.0)
                                
                                # Load kernel weights
                                # W shape: (C_in, C_out // groups, Kd, Kh, Kw)
                                # We need to access W[c_in, c_out_block, k_d, k_h, k_w]
                                # For simplicity, we'll process all output channels in the block
                                w_offset = (c_in_range[:, None] * stride_W_cin + 
                                           tl.arange(0, BLOCK_SIZE_C_out)[None, :] * stride_W_cout +
                                           k_d * stride_W_kd +
                                           k_h * stride_W_kh +
                                           k_w * stride_W_kw)
                                w = tl.load(W_ptr + w_offset, mask=(mask_c_in[:, None] & True), other=0.0)
                                
                                # Accumulate: x[c_in] * w[c_in, c_out]
                                acc += tl.sum(x[:, None] * w, axis=0)
    
    # Handle bias if provided
    if B_ptr is not None:
        b_offset = pid_c_out * stride_Y_c
        bias = tl.load(B_ptr + b_offset)
        acc += bias
    
    # Store result
    y_offset = (pid_b * stride_Y_b + 
               (pid_c_out + tl.arange(0, BLOCK_SIZE_C_out)) * stride_Y_c + 
               out_d * stride_Y_d + 
               out_h * stride_Y_h + 
               out_w * stride_Y_w)
    tl.store(Y_ptr + y_offset, acc.to(tl.float16), mask=pid_c_out + tl.arange(0, BLOCK_SIZE_C_out) < C_out)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_in_w, C_out_per_group, Kd, Kh, Kw = weight.shape
    C_out = C_out_per_group * groups
    
    # Calculate output dimensions
    D_out = (D - 1) * stride[0] - 2 * padding[0] + Kd + output_padding[0]
    H_out = (H - 1) * stride[1] - 2 * padding[1] + Kh + output_padding[1]
    W_out = (W - 1) * stride[2] - 2 * padding[2] + Kw + output_padding[2]
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Prepare output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set up block sizes for tiling
    BLOCK_SIZE_C_out = 8  # Tile size for output channels
    BLOCK_SIZE_C_in = 8   # Tile size for input channels
    BLOCK_SIZE_Kd = 1
    BLOCK_SIZE_Kh = 1
    BLOCK_SIZE_Kw = 1
    
    # Calculate grid dimensions
    grid = (B, triton.cdiv(C_out, BLOCK_SIZE_C_out), D_out, H_out, W_out)
    
    # Calculate strides
    stride_X = x.stride()
    stride_W = weight.stride()
    stride_Y = y.stride()
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, groups,
        D, H, W,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_Kd=BLOCK_SIZE_Kd,
        BLOCK_SIZE_Kh=BLOCK_SIZE_Kh,
        BLOCK_SIZE_Kw=BLOCK_SIZE_Kw,
        stride_X_b=stride_X[0], stride_X_c=stride_X[1], stride_X_d=stride_X[2], 
        stride_X_h=stride_X[3], stride_X_w=stride_X[4],
        stride_W_cin=stride_W[0], stride_W_cout=stride_W[1], 
        stride_W_kd=stride_W[2], stride_W_kh=stride_W[3], stride_W_kw=stride_W[4],
        stride_Y_b=stride_Y[0], stride_Y_c=stride_Y[1], stride_Y_d=stride_Y[2], 
        stride_Y_h=stride_Y[3], stride_Y_w=stride_Y[4],
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and create the weight/bias tensors manually
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Calculate actual kernel dimensions
        Kd, Kh, Kw = kernel_size
        
        # Initialize weights with proper shape: (in_channels, out_channels // groups, Kd, Kh, Kw)
        # This matches PyTorch's ConvTranspose3d weight layout
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, Kd, Kh, Kw))
        
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
        Performs the 3D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, output_padding={self.output_padding}, '
                f'groups={self.groups}, bias={self.bias is not None}')


# Import math for initialization
import math