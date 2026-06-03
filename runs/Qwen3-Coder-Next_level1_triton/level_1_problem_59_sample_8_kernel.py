import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (B, C_in, H, W, D)
    w_ptr,  # (C_out, C_in, kH, kW, 1) - kernel is always 1 in depth
    bias_ptr,  # (C_out,) or None
    out_ptr,  # (B, C_out, H_out, W_out, D)
    # Shapes
    B, C_in, H, W, D,
    C_out, kH, kW,
    # Strides
    stride_xB, stride_xC, stride_xH, stride_xW, stride_xD,
    stride_wCout, stride_wCin, stride_wkH, stride_wkW, stride_wkD,
    stride_outB, stride_outC, stride_outH, stride_outW, stride_outD,
    # Convolution parameters
    stride_h, stride_w, stride_d,
    pad_h, pad_w, pad_d,
    dil_h, dil_w, dil_d,
    # Groups
    groups,
    # Block sizes for tiling
    BLOCK_SIZE_Cout: tl.constexpr,
    BLOCK_SIZE_Cin: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_d = tl.program_id(4)
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    D_out = D  # Since kernel depth is 1 and stride_d=1, D_out = D
    
    # Check bounds
    if pid_b >= B or pid_cout >= C_out or pid_h >= H_out or pid_w >= W_out or pid_d >= D_out:
        return
    
    # Compute output pointer offset
    out_ptr += pid_b * stride_outB + pid_cout * stride_outC + pid_h * stride_outH + pid_w * stride_outW + pid_d * stride_outD
    
    # Compute the starting input position corresponding to this output position
    # For output at (pid_h, pid_w, pid_d), the corresponding input position for the kernel center
    h_start = pid_h * stride_h - pad_h
    w_start = pid_w * stride_w - pad_w
    d_start = pid_d  # Since kernel depth is 1, d_start = pid_d
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_Cout,), dtype=tl.float32)
    
    # Handle groups: each group processes a subset of input channels
    group_size_cin = C_in // groups
    group_size_cout = C_out // groups
    group_id = pid_cout // group_size_cout
    cout_start = group_id * group_size_cout
    cin_start = group_id * group_size_cin
    
    # Loop over input channels in the group
    for pid_cin in range(group_size_cin):
        cin_idx = cin_start + pid_cin
        
        # Loop over kernel height
        for kh in range(kH):
            h = h_start + kh * dil_h
            if h < 0 or h >= H:
                continue
                
            # Loop over kernel width
            for kw in range(kW):
                w = w_start + kw * dil_w
                if w < 0 or w >= W:
                    continue
                    
                # Load input value: x[pid_b, cin_idx, h, w, pid_d]
                x_offset = pid_b * stride_xB + cin_idx * stride_xC + h * stride_xH + w * stride_xW + pid_d * stride_xD
                x_val = tl.load(x_ptr + x_offset)
                
                # Load weight: w[pid_cout, cin_idx, kh, kw, 0]
                w_offset = pid_cout * stride_wCout + cin_idx * stride_wCin + kh * stride_wkH + kw * stride_wkW
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_cout)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr, acc.to(out_ptr.dtype.element_ty))


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Performs 3D convolution with kernel shape (k, k, 1) using Triton.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H, W, D = x.shape
    C_out, _, kH, kW, kD = weight.shape
    
    # Validate kernel depth is 1
    assert kD == 1, "Kernel depth must be 1"
    
    # Compute output dimensions
    stride_h = stride_w = stride_d = stride
    pad_h = pad_w = pad_d = padding
    dil_h = dil_w = dil_d = dilation
    
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    D_out = D
    
    # Create output tensor
    out = torch.empty(B, C_out, H_out, W_out, D_out, dtype=x.dtype, device=x.device)
    
    # Compute strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Grid configuration
    # We'll use 5D grid: [B, C_out, H_out, W_out, D_out]
    # But for better performance, we'll tile across C_out and H/W
    
    # Tiling parameters (tunable for performance)
    BLOCK_SIZE_Cout = min(32, C_out)
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    BLOCK_SIZE_D = 1
    
    # Adjust grid dimensions based on tiling
    grid = (
        B,
        (C_out + BLOCK_SIZE_Cout - 1) // BLOCK_SIZE_Cout,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W, D,
        C_out, kH, kW,
        *stride_x, *stride_w, *stride_out,
        stride_h, stride_w, stride_d,
        pad_h, pad_w, pad_d,
        dil_h, dil_w, dil_d,
        groups,
        BLOCK_SIZE_Cout=BLOCK_SIZE_Cout,
        BLOCK_SIZE_Cin=1,  # Process one input channel per iteration
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, 
                                (kernel_size, kernel_size, 1), 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using optimized Triton kernel.
        """
        # Extract parameters from the original conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        
        # Use our optimized Triton implementation
        return triton_conv3d(x, weight, bias,
                            stride=self.conv3d.stride[0],
                            padding=self.conv3d.padding[0],
                            dilation=self.conv3d.dilation[0],
                            groups=self.conv3d.groups)