import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, Cout, Lin, Lout, K, S, P, D,
    stride_xb, stride_xc, stride_xl,
    stride_wc, stride_wo, stride_wk,
    stride_ob, stride_oo, stride_ol,
    BLOCK_L: tl.constexpr,
    C_IN: tl.constexpr,
    K_SIZE: tl.constexpr,
):
    # pid_0: batch * Cout
    # pid_1: Lout block
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    b = pid_0 // Cout
    cout = pid_0 % Cout
    lout_start = pid_1 * BLOCK_L
    
    # Load bias for the current output channel
    # b_ptr is expected to be a pointer to a tensor of shape (Cout,)
    acc_bias = tl.load(b_ptr + cout)
    
    lout_offsets = lout_start + tl.arange(0, BLOCK_L)
    mask_lout = lout_offsets < Lout
    
    # Initialize accumulator for the block of Lout
    acc = tl.full([BLOCK_L], acc_bias, dtype=tl.float32)
    
    # Transposed Convolution Logic:
    # For each k in kernel_size, we find which input index lin contributes to lout
    # lout = lin * stride - padding + k * dilation
    # => lin = (lout + padding - k * dilation) / stride
    for k in range(K_SIZE):
        # Calculate the corresponding input indices for the current block of lout
        lin_offsets = (lout_offsets + P - k * D) // S
        
        # Check if the mapping is valid (must be divisible by stride and within bounds)
        valid_lin = (lout_offsets + P - k * D) % S == 0
        valid_lin &= (lin_offsets >= 0) & (lin_offsets < Lin)
        
        # Sum over input channels
        for cin in range(C_IN):
            # Load x[b, cin, lin]
            # x shape: (B, Cin, Lin)
            x_val = tl.load(
                x_ptr + b * stride_xb + cin * stride_xc + lin_offsets * stride_xl, 
                mask=mask_lout & valid_lin, 
                other=0.0
            )
            # Load w[cin, cout, k]
            # w shape: (Cin, Cout, K)
            w_val = tl.load(w_ptr + cin * stride_wc + cout * stride_wo + k * stride_wk)
            
            acc += x_val * w_val
            
    # Store the result in the output tensor
    # out shape: (B, Cout, Lout)
    tl.store(
        out_ptr + b * stride_ob + cout * stride_oo + lout_offsets * stride_ol, 
        acc, 
        mask=mask_lout
    )

def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # x: (B, Cin, Lin)
    # weight: (Cin, Cout, K)
    # bias: (Cout,)
    B, Cin, Lin = x.shape
    Cin_w, Cout, K = weight.shape
    
    Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        bias = torch.zeros(Cout, device=x.device, dtype=x.dtype)
        
    # Strides
    stride_xb, stride_xc, stride_xl = x.stride()
    stride_wc, stride_wo, stride_wk = weight.stride()
    stride_ob, stride_oo, stride_ol = out.stride()
    
    BLOCK_L = 256
    grid = (B * Cout, (Lout + BLOCK_L - 1) // BLOCK_L)
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, Lin, Lout, K, stride, padding, dilation,
        stride_xb, stride_xc, stride_xl,
        stride_wc, stride_wo, stride_wk,
        stride_ob, stride_oo, stride_ol,
        BLOCK_L=BLOCK_L,
        C_IN=Cin,
        K_SIZE=K
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # We use the standard ConvTranspose1d to manage parameters
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton wrapper instead of the PyTorch operator
        return triton_conv_transpose1d(
            x, 
            self.conv1d_transpose.weight, 
            self.conv1d_transpose.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )