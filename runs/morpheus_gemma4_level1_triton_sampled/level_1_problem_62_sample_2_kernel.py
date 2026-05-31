import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H, W,
    C_out, 
    S, P, D, G,
    x_sB, x_sC, x_sH, x_sW,
    w_sOC, w_sIC, w_sKH, w_sKW,
    out_sB, out_sOC, out_sOH, out_sOW,
    H_out, W_out,
    kH: tl.constexpr, kW: tl.constexpr, C_in_per_group: tl.constexpr,
):
    # Map program ID to output coordinates
    pid = tl.program_id(0)
    
    ow = pid % W_out
    pid //= W_out
    oh = pid % H_out
    pid //= H_out
    oc = pid % C_out
    b = pid // C_out
    
    # Grouping logic: determine which input channels this output channel connects to
    start_ic = (oc // (C_out // G)) * C_in_per_group
    
    acc = 0.0
    # Iterate over the input channels within the group
    for ic_offset in range(0, C_in_per_group):
        ic = start_ic + ic_offset
        # Iterate over the kernel window
        for kh in range(0, kH):
            h_in = oh * S + kh * D - P
            if h_in >= 0 and h_in < H:
                for kw in range(0, kW):
                    w_in = ow * S + kw * D - P
                    if w_in >= 0 and w_in < W:
                        # Load input element and weight element
                        # Input: [batch, channel, height, width]
                        x_val = tl.load(x_ptr + b * x_sB + ic * x_sC + h_in * x_sH + w_in * x_sW)
                        # Weight: [out_channel, in_channel_per_group, kH, kW]
                        w_val = tl.load(w_ptr + oc * w_sOC + ic_offset * w_sIC + kh * w_sKH + kw * w_sKW)
                        acc += x_val * w_val
    
    # Add bias if it exists
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
        
    # Store the final result in the output tensor
    tl.store(out_ptr + b * out_sB + oc * out_sOC + oh * out_sOH + ow * out_sOW, acc)

def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, H, W = x.shape
    C_out, C_in_per_group, kH, kW = weight.shape
    
    # Handle potential tuple inputs for stride, padding, dilation
    S = stride if isinstance(stride, int) else stride[0]
    P = padding if isinstance(padding, int) else padding[0]
    D = dilation if isinstance(dilation, int) else dilation[0]
    G = groups

    # Calculate output dimensions
    H_out = ((H + 2 * P - D * (kH - 1) - 1) // S) + 1
    W_out = ((W + 2 * P - D * (kW - 1) - 1) // S) + 1
    
    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Get strides for pointer arithmetic
    x_sB, x_sC, x_sH, x_sW = x.stride()
    w_sOC, w_sIC, w_sKH, w_sKW = weight.stride()
    out_sB, out_sOC, out_sOH, out_sOW = out.stride()
    
    # Grid is one program per output element
    grid = (B * C_out * H_out * W_out,)
    
    conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out,
        S, P, D, G,
        x_sB, x_sC, x_sH, x_sW,
        w_sOC, w_sIC, w_sKH, w_sKW,
        out_sB, out_sOC, out_sOH, out_sOW,
        H_out, W_out,
        kH=kH, kW=kW, C_in_per_group=C_in_per_group
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.Conv2d to initialize and manage weights and bias
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Store parameters for the Triton kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace the standard nn.Conv2d forward pass with the custom Triton kernel
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )