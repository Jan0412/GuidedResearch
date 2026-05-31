import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, Lin, Cout, K, S, D, Lout,
    has_bias,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Parallelize over (batch * out_channels) and (Lout / BLOCK_L)
    batch_id = pid_0 // Cout
    oc_id = pid_0 % Cout
    
    l_start = pid_1 * BLOCK_L
    l_offsets = l_start + tl.arange(0, BLOCK_L)
    l_mask = l_offsets < Lout

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Iterate over the kernel size
    for k in range(K):
        # Iterate over input channels in blocks
        for c_start in range(0, Cin, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < Cin
            
            # Load weights: shape (BLOCK_C,)
            # w_ptr is (Cout, Cin, K)
            w_off = oc_id * (Cin * K) + c_offsets * K + k
            w_val = tl.load(w_ptr + w_off, mask=c_mask, other=0.0)
            
            # Load inputs: shape (BLOCK_L, BLOCK_C)
            # x_ptr is (B, Cin, Lin)
            # Index: batch_id * (Cin * Lin) + c_offsets * Lin + (l_offsets * S + k * D)
            x_off = (batch_id * Cin * Lin) + (c_offsets[None, :] * Lin) + (l_offsets[:, None] * S + k * D)
            x_val = tl.load(x_ptr + x_off, mask=l_mask[:, None] & c_mask[None, :], other=0.0)
            
            # Element-wise multiply and sum across the channel dimension
            acc += tl.sum(x_val * w_val[None, :], axis=1)

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + oc_id)
        out_val = acc + bias_val
    else:
        out_val = acc

    # Store result in output tensor: shape (B, Cout, Lout)
    out_off = batch_id * (Cout * Lout) + oc_id * Lout + l_offsets
    tl.store(out_ptr + out_off, out_val, mask=l_mask)


def triton_conv1d(x, weight, bias, stride, dilation):
    """
    Triton wrapper for 1D convolution.
    """
    B, Cin, Lin = x.shape
    Cout, _, K = weight.shape
    
    # Calculate output length (padding=0)
    Lout = (Lin - dilation * (K - 1) - 1) // stride + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    # Hyperparameters for Triton kernel
    BLOCK_L = 128
    BLOCK_C = 32
    
    # Grid: (B * Cout) programs for the first dimension, (Lout / BLOCK_L) for the second
    grid = (B * Cout, triton.cdiv(Lout, BLOCK_L))
    
    has_bias = 1 if bias is not None else 0
    b_ptr = bias if bias is not None else None

    conv1d_kernel[grid](
        x, weight, b_ptr, out,
        B, Cin, Lin, Cout, K, stride, dilation, Lout,
        has_bias,
        BLOCK_L=BLOCK_L,
        BLOCK_C=BLOCK_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv1d to manage parameters (weights and bias)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel.
        """
        # Ensure input is on CUDA and FP32 as required
        x = x.to(torch.float32)
        weight = self.conv1d.weight.to(torch.float32)
        bias = self.conv1d.bias.to(torch.float32) if self.conv1d.bias is not None else None
        
        return triton_conv1d(x, weight, bias, self.stride, self.dilation)