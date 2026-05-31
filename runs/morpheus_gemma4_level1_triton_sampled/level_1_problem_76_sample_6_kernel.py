import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, L, C_out, K, stride, dilation, L_out,
    has_bias,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Program IDs
    pid_batch_out = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Calculate batch and output channel indices
    batch_id = pid_batch_out // C_out
    oc_id = pid_batch_out % C_out
    
    # Output length offsets
    l_offsets = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = l_offsets < L_out
    
    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)
    
    # Loop over input channels in blocks to optimize memory access
    for ic_start in range(0, C_in, BLOCK_SIZE_C):
        ic_offsets = ic_start + tl.arange(0, BLOCK_SIZE_C)
        mask_c = ic_offsets < C_in
        
        # Loop over the kernel size (K is typically small)
        for k in range(K):
            # Load weights: shape (BLOCK_SIZE_C,)
            # w_ptr is (C_out, C_in, K)
            w_off = oc_id * (C_in * K) + ic_offsets * K + k
            w = tl.load(w_ptr + w_off, mask=mask_c, other=0.0)
            
            # Load input x: shape (BLOCK_SIZE_L, BLOCK_SIZE_C)
            # x_ptr is (B, C_in, L)
            l_expanded = l_offsets[:, None]
            ic_expanded = ic_offsets[None, :]
            
            # Calculate indices for x
            x_off = batch_id * (C_in * L) + ic_expanded * L + (l_expanded * stride + k * dilation)
            
            # Mask for x to prevent out-of-bounds access on the length dimension
            mask_x = (l_expanded * stride + k * dilation < L)[:, None] & mask_c[None, :]
            x = tl.load(x_ptr + x_off, mask=mask_x, other=0.0)
            
            # Compute dot product across input channels for the current kernel element
            # x: (BLOCK_SIZE_L, BLOCK_SIZE_C), w: (BLOCK_SIZE_C,)
            acc += tl.sum(x * w[None, :], axis=1)
    
    # Add bias if applicable
    if has_bias:
        bias = tl.load(b_ptr + oc_id)
        acc += bias
    
    # Store the final result: out_ptr is (B, C_out, L_out)
    out_off = batch_id * (C_out * L_out) + oc_id * L_out + l_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_l)

def triton_conv1d(x, weight, bias, stride, dilation):
    # Ensure tensors are contiguous and on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, L = x.shape
    C_out, _, K = weight.shape
    
    # Calculate output length (padding=0)
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    out = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Hyperparameters for Triton kernel
    BLOCK_SIZE_L = 128
    BLOCK_SIZE_C = 32
    
    # Grid: (Batch * OutChannels, OutputLength / BLOCK_SIZE_L)
    grid = (B * C_out, triton.cdiv(L_out, BLOCK_SIZE_L))
    
    has_bias = 1 if bias is not None else 0
    b_ptr = bias if bias is not None else torch.zeros(1, device=x.device)
    
    conv1d_kernel[grid](
        x, weight, b_ptr, out,
        B, C_in, L, C_out, K, stride, dilation, L_out,
        has_bias,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized 1D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        
        # Use nn.Parameter to store weights and bias to maintain compatibility with torch.nn.Module
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel implementation.
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)