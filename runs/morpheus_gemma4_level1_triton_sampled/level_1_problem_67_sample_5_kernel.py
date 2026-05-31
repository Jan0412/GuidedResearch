import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    B, Cin, Cout, L, Lout, K, S, P, D, G,
    BLOCK_SIZE: tl.constexpr,
):
    # pid_0: batch_idx * Cout + out_channel_idx
    # pid_1: out_spatial_idx_block
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    b = pid_0 // Cout
    co = pid_0 % Cout
    
    lo_start = pid_1 * BLOCK_SIZE
    lo_offsets = lo_start + tl.arange(0, BLOCK_SIZE)
    
    # Load bias for the current output channel
    bias = tl.load(B_ptr + co)
    acc = tl.full([BLOCK_SIZE], bias, dtype=tl.float32)
    
    cin_per_group = Cin // G
    group_idx = co // (Cout // G)
    cin_offset = group_idx * cin_per_group
    
    # Loop over input channels in the group
    ci = 0
    while ci < cin_per_group:
        # Loop over the kernel window
        k = 0
        while k < K:
            # Weight W[co, ci, k]
            # W layout: (Cout, Cin_per_group, K)
            w = tl.load(W_ptr + co * (cin_per_group * K) + ci * K + k)
            
            # Input X[b, cin_offset + ci, lo * S - P + k * D]
            # X layout: (B, Cin, L)
            input_lo_offsets = lo_offsets * S - P + k * D
            x_ptr_base = X_ptr + b * (Cin * L) + (cin_offset + ci) * L
            
            # Mask for padding and boundary checks
            mask = (lo_offsets < Lout) & (input_lo_offsets >= 0) & (input_lo_offsets < L)
            x = tl.load(x_ptr_base + input_lo_offsets, mask=mask, other=0.0)
            
            acc += x * w
            k += 1
        ci += 1
    
    # Store result to output tensor
    # Out layout: (B, Cout, Lout)
    out_ptr_base = Out_ptr + b * (Cout * Lout) + co * Lout
    tl.store(out_ptr_base + lo_offsets, acc, mask=lo_offsets < Lout)


def triton_conv1d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        # Create a dummy zero bias if none provided to simplify kernel
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)

    B, Cin, L = x.shape
    Cout, Cin_per_group, K = weight.shape
    
    # Calculate output length
    Lout = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    # Grid: (B * Cout) programs for channels, (Lout / BLOCK_SIZE) for spatial dimension
    grid = (B * Cout, (Lout + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, L, Lout, K, stride, padding, dilation, groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Weights: (out_channels, in_channels // groups, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )