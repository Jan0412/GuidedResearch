import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # [B, IC, D, H, W]
    W_ptr,  # [IC, OC, kD, kH, kW]
    B_ptr,  # [OC] - optional bias
    Y_ptr,  # [B, OC, D_out, H_out, W_out]
    # Dimensions
    B, IC, OC,
    D, H, W,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes for tiling
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs for output tensor dimensions
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1) // (H_out * W_out)
    pid_h = (tl.program_id(1) % (H_out * W_out)) // W_out
    pid_w = tl.program_id(1) % W_out
    
    # Compute the starting indices for the output element
    out_d = tl.program_id(2)
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    
    # Loop over input channels with blocking for better cache utilization
    for ic_start in range(0, IC, BLOCK_IC):
        ic_end = tl.minimum(ic_start + BLOCK_IC, IC)
        
        # Loop over kernel dimensions
        for kd in range(kD):
            # Compute corresponding input depth
            in_d = (out_d - kd * dil_d + pad_d) // stride_d
            # Check if input depth is valid
            if in_d >= 0 and in_d < D:
                for kh in range(kH):
                    # Compute corresponding input height
                    in_h = (out_h - kh * dil_h + pad_h) // stride_h
                    if in_h >= 0 and in_h < H:
                        for kw in range(kW):
                            # Compute corresponding input width
                            in_w = (out_w - kw * dil_w + pad_w) // stride_w
                            if in_w >= 0 and in_w < W:
                                # Compute actual positions
                                actual_out_d = out_d
                                actual_out_h = out_h
                                actual_out_w = out_w
                                
                                # Check output padding constraints
                                if actual_out_d >= D_out or actual_out_h >= H_out or actual_out_w >= W_out:
                                    continue
                                    
                                # Compute input indices
                                input_d = in_d
                                input_h = in_h
                                input_w = in_w
                                
                                # Calculate offsets for input tensor
                                x_offset = (
                                    pid_b * (IC * D * H * W) +
                                    tl.arange(0, BLOCK_IC)[:, None] * (D * H * W) +
                                    input_d * (H * W) +
                                    input_h * W +
                                    input_w
                                ) % (B * IC * D * H * W)
                                
                                # Calculate offsets for weight tensor
                                w_offset = (
                                    tl.arange(0, BLOCK_IC)[:, None] * (OC * kD * kH * kW) +
                                    tl.arange(0, BLOCK_OC)[None, :] * (kD * kH * kW) +
                                    kd * (kH * kW) +
                                    kh * kW +
                                    kw
                                ) % (IC * OC * kD * kH * kW)
                                
                                # Load input values
                                x_mask = (tl.arange(0, BLOCK_IC)[:, None] < (ic_end - ic_start)) & \
                                         (tl.arange(0, BLOCK_IC)[:, None] >= 0)
                                x_val = tl.load(X_ptr + x_offset, mask=x_mask, other=0.0)
                                
                                # Load weight values
                                w_mask = (tl.arange(0, BLOCK_IC)[:, None] < (ic_end - ic_start)) & \
                                         (tl.arange(0, BLOCK_OC)[None, :] < OC)
                                w_val = tl.load(W_ptr + w_offset, mask=w_mask, other=0.0)
                                
                                # Accumulate
                                acc += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if B_ptr is not None:
        b_offset = tl.arange(0, BLOCK_OC)
        b_mask = b_offset < OC
        bias = tl.load(B_ptr + b_offset, mask=b_mask, other=0.0)
        acc += bias
    
    # Store result
    y_offset = (
        pid_b * (OC * D_out * H_out * W_out) +
        tl.arange(0, BLOCK_OC) * (D_out * H_out * W_out) +
        out_d * (H_out * W_out) +
        out_h * W_out +
        out_w
    )
    y_mask = tl.arange(0, BLOCK_OC) < OC
    tl.store(Y_ptr + y_offset, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose3d for FP32 tensors.
    This implementation uses tiling and blocking for optimal performance.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    B, IC, D, H, W = x.shape
    OC, IC_w, kD, kH, kW = weight.shape
    
    # Verify input compatibility
    assert IC == IC_w, f"Input channels mismatch: {IC} vs {IC_w}"
    assert groups == 1, "Only groups=1 is supported in this Triton kernel"
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + dilation * (kD - 1) + output_padding + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (kW - 1) + output_padding + 1
    
    # Create output tensor
    output = torch.empty(B, OC, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tiling (tunable parameters for performance)
    BLOCK_OC = 8
    BLOCK_IC = 16
    BLOCK_D = 4
    BLOCK_H = 8
    BLOCK_W = 8
    
    # Grid dimensions: [batch, (out_h * out_w), out_d]
    grid = (B, H_out * W_out, D_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, output,
        B, IC, OC,
        D, H, W,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        dilation, dilation, dilation,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Register buffers for parameters
        self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for initialization
import math