import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,          # Input pointer
    w_ptr,          # Weight pointer
    b_ptr,          # Bias pointer
    out_ptr,        # Output pointer
    B, IC, L,       # Input dims
    OC, K,          # Weight dims
    S, D,           # Stride, Dilation
    L_out,          # Output length
    BLOCK_L: tl.constexpr,
):
    # Grid: (B * OC, ceil(L_out / BLOCK_L))
    pid_batch_oc = tl.program_id(0)
    pid_l = tl.program_id(1)

    batch_idx = pid_batch_oc // OC
    oc_idx = pid_batch_oc % OC

    # Calculate output spatial offsets
    l_start = pid_l * BLOCK_L
    l_offsets = l_start + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L_out

    # Accumulator for the output block
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Convolution loop
    # We iterate over input channels and kernel width
    # Weights for a specific output channel (oc_idx) are constant for this program
    for ic in range(0, IC):
        for k in range(0, K):
            # Load weight: w[oc_idx, ic, k]
            w_offset = oc_idx * (IC * K) + ic * K + k
            weight = tl.load(w_ptr + w_offset)

            # Load input: x[batch_idx, ic, l_offsets * S + k * D]
            # Calculate the spatial indices for the input tensor
            x_spatial_offsets = l_offsets * S + k * D
            # Mask for input bounds (though L_out is calculated to fit, safety first)
            mask_x = x_spatial_offsets < L
            
            x_ptr_offset = batch_idx * (IC * L) + ic * L + x_spatial_offsets
            x_vals = tl.load(x_ptr + x_ptr_offset, mask=mask_x, other=0.0)
            
            acc += x_vals * weight

    # Add bias if it exists
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc_idx)
        acc += bias

    # Store the result
    out_ptr_offset = batch_idx * (OC * L_out) + oc_idx * L_out + l_offsets
    tl.store(out_ptr + out_ptr_offset, acc, mask=mask_l)


def triton_conv1d(x, weight, bias, stride, dilation):
    # Input shapes
    # x: (B, IC, L)
    # weight: (OC, IC, K)
    # bias: (OC,)
    B, IC, L = x.shape
    OC, IC_w, K = weight.shape
    
    # Calculate output length
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, OC, L_out), device=x.device, dtype=x.dtype)
    
    BLOCK_L = 64
    # Grid: (B * OC, ceil(L_out / BLOCK_L))
    grid = (B * OC, triton.cdiv(L_out, BLOCK_L))
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, IC, L,
        OC, K,
        stride, dilation,
        L_out,
        BLOCK_L=BLOCK_L,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the nn.Conv1d to maintain the parameter state and initialization
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel.
        """
        # Extract parameters from the nn.Conv1d layer
        weight = self.conv1d.weight
        bias = self.conv1d.bias if self.conv1d.bias is not None else None
        stride = self.conv1d.stride[0]
        dilation = self.conv1d.dilation[0]
        
        return triton_conv1d(x, weight, bias, stride, dilation)