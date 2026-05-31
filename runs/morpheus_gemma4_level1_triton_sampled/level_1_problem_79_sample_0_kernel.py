import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    L_in, L_out,
    stride, padding, dilation,
    C_in, C_out, K_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over (batch * C_out) and L_out
    pid = tl.program_id(0)
    batch_idx = pid // C_out
    cout_idx = pid % C_out
    
    # Offset for the current block of output elements
    off_j = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_j = off_j < L_out
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    # Since C_in and K_size are typically small, we can loop over them
    for cin in range(C_in):
        for k in range(K_size):
            # Transposed convolution mapping: j = i * stride - padding + k * dilation
            # Solving for i: i = (j + padding - k * dilation) / stride
            num = off_j + padding - k * dilation
            
            # The input index i must be an integer and within [0, L_in - 1]
            mask_i = (num >= 0) & (num % stride == 0) & (num // stride < L_in)
            i = num // stride
            
            # Load input x[batch, cin, i]
            # x shape: (batch, C_in, L_in)
            x_off = batch_idx * (C_in * L_in) + cin * L_in + i
            val_x = tl.load(x_ptr + x_off, mask=mask_i & mask_j, other=0.0)
            
            # Load weight w[cin, cout, k]
            # w shape: (C_in, C_out, K_size)
            w_off = cin * (C_out * K_size) + cout_idx * K_size + k
            val_w = tl.load(w_ptr + w_off)
            
            acc += val_x * val_w
            
    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + cout_idx)
        acc += bias_val
        
    # Store the final result in the output tensor
    # out shape: (batch, C_out, L_out)
    out_off = batch_idx * (C_out * L_out) + cout_idx * L_out + off_j
    tl.store(out_ptr + out_off, acc, mask=mask_j)

def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation, C_in, C_out, K_size):
    assert x.is_cuda, "Tensors must be on CUDA."
    
    batch, _, L_in = x.shape
    # Calculate output length for ConvTranspose1d
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_size - 1) + 1
    
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    if bias is not None:
        bias = bias.contiguous().float()
        
    out = torch.empty((batch, C_out, L_out), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 128
    # Grid: (batch * out_channels, ceil(L_out / BLOCK_SIZE))
    grid = (batch * C_out, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        L_in, L_out,
        stride, padding, dilation,
        C_in, C_out, K_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized version of the transposed 1D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # We use nn.ConvTranspose1d to manage the learnable parameters
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace the standard PyTorch forward pass with the Triton implementation
        return triton_conv_transpose1d(
            x, 
            self.conv1d_transpose.weight, 
            self.conv1d_transpose.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.in_channels, 
            self.out_channels, 
            self.kernel_size
        )