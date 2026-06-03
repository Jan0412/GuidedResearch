import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # [batch, in_channels, D, H, W]
    W_ptr,  # [in_channels, out_channels, kD, kH, kW]
    B_ptr,  # [out_channels] or None
    Y_ptr,  # [batch, out_channels, D_out, H_out, W_out]
    # Dimensions
    batch_size, in_channels, out_channels,
    D, H, W,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    # Output dimensions
    D_out, H_out, W_out,
    # Strides for memory access
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_c, stride_y_d, stride_y_h, stride_y_w,
    # Block sizes
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs for output spatial positions and channels
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate input spatial positions corresponding to output position
    # For transposed conv: d_in = pid_d * stride_d - padding_d + pid_kd * dilation_d
    # But we need to iterate over kernel positions
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for pid_c_in in range(in_channels):
        # Iterate over kernel positions
        for pid_kd in range(BLOCK_SIZE_KD):
            for pid_kh in range(BLOCK_SIZE_KH):
                for pid_kw in range(BLOCK_SIZE_KW):
                    # Calculate corresponding input position
                    d_in = pid_d * stride_d - padding_d + pid_kd * dilation_d
                    h_in = pid_h * stride_h - padding_h + pid_kh * dilation_h
                    w_in = pid_w * stride_w - padding_w + pid_kw * dilation_w
                    
                    # Check if input position is valid
                    mask_d = (tl.arange(0, BLOCK_SIZE_D) + pid_d * BLOCK_SIZE_D < D) & (d_in >= 0) & (d_in < D)
                    mask_h = (tl.arange(0, BLOCK_SIZE_H) + pid_h * BLOCK_SIZE_H < H) & (h_in >= 0) & (h_in < H)
                    mask_w = (tl.arange(0, BLOCK_SIZE_W) + pid_w * BLOCK_SIZE_W < W) & (w_in >= 0) & (w_in < W)
                    
                    # Create 3D mask
                    mask_d_2d = mask_d[:, None, None]
                    mask_h_2d = mask_h[None, :, None]
                    mask_w_2d = mask_w[None, None, :]
                    mask = mask_d_2d & mask_h_2d & mask_w_2d
                    
                    # Load input data
                    x_offset = pid_b * stride_x_b + pid_c_in * stride_x_c + d_in * stride_x_d + h_in * stride_x_h + w_in * stride_x_w
                    x_val = tl.load(X_ptr + x_offset, mask=mask, other=0.0)
                    
                    # Load weight data
                    w_offset = pid_c_in * stride_w_ic + pid_c_out * stride_w_oc + pid_kd * stride_w_kd + pid_kh * stride_w_kh + pid_kw * stride_w_kw
                    w_val = tl.load(W_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Apply bias if present
    if B_ptr is not None:
        bias = tl.load(B_ptr + pid_c_out)
        acc += bias
    
    # Store result
    y_offset = pid_b * stride_y_b + pid_c_out * stride_y_c + pid_d * stride_y_d + pid_h * stride_y_h + pid_w * stride_y_w
    tl.store(Y_ptr + y_offset, acc.to(tl.float32), mask=(pid_b < batch_size) & (pid_c_out < out_channels) & 
             (pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D) < D_out) & 
             (pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H) < H_out) & 
             (pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W) < W_out))


def triton_conv_transpose3d(x, weight, bias, stride, padding, dilation):
    """
    Perform 3D transposed convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, D, H, W = x.shape
    ic, out_channels, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kD - 1) + 1
    H_out = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kH - 1) + 1
    W_out = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kW - 1) + 1
    
    # Allocate output tensor
    y = torch.empty(batch_size, out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_y = y.stride()
    
    # Define grid dimensions
    # Grid: [batch_size, out_channels, D_out, H_out, W_out]
    # But we need to be careful about block sizes
    BLOCK_SIZE_D = 2
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    BLOCK_SIZE_KD = 2
    BLOCK_SIZE_KH = 2
    BLOCK_SIZE_KW = 2
    
    grid = (batch_size, out_channels, (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D, 
            (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        D, H, W,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        D_out, H_out, W_out,
        stride_x[0], stride_x[1], stride_x[2], stride_x[3], stride_x[4],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3], stride_w[4],
        stride_y[0], stride_y[1], stride_y[2], stride_y[3], stride_y[4],
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        # Create the weight and bias parameters manually
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )