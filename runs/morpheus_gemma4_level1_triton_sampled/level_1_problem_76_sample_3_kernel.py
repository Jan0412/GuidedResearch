import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, L, C_out, K, stride, dilation, L_out,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B * C_out, tl.cdiv(L_out, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    batch_id = pid_m // C_out
    oc_id = pid_m % C_out
    
    # Output length offsets
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < L_out
    
    # Accumulator for the output block
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Iterate over input channels and kernel width
    # Triton requires while loops for non-constexpr bounds
    ic = 0
    while ic < C_in:
        k = 0
        while k < K:
            # Calculate input index: batch, channel, and the sliding window position
            # x_idx = batch_id * (C_in * L) + ic * L + (n_offsets * stride + k * dilation)
            x_idx_base = (batch_id * C_in * L) + (ic * L)
            x_idx = x_idx_base + (n_offsets * stride + k * dilation)
            
            # Mask to prevent out-of-bounds access on the input length
            x_mask = mask_n & (n_offsets * stride + k * dilation < L)
            x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
            
            # Weight index: oc_id, ic, k
            w_idx = (oc_id * C_in * K) + (ic * K) + k
            w_val = tl.load(w_ptr + w_idx)
            
            acc += x_val * w_val
            k += 1
        ic += 1
    
    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc_id)
        acc += bias_val
        
    # Store the result in the output tensor
    out_idx = (batch_id * C_out * L_out) + (oc_id * L_out) + n_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_n)

def triton_conv1d(x, weight, bias, stride, dilation):
    # x: (B, C_in, L)
    # weight: (C_out, C_in, K)
    # bias: (C_out,)
    B, C_in, L = x.shape
    C_out, _, K = weight.shape
    
    # Calculate output length
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Tuning parameter
    BLOCK_N = 128
    
    # Grid: Parallelize over (Batch * OutChannels) and blocks of OutputLength
    grid = (B * C_out, triton.cdiv(L_out, BLOCK_N))
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, L, C_out, K, stride, dilation, L_out,
        BLOCK_N=BLOCK_N
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized 1D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # We use nn.Conv1d to manage parameters (weights and bias)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel.
        """
        # Extract weight and bias from the nn.Conv1d module
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        
        return triton_conv1d(x, weight, bias, self.stride, self.dilation)