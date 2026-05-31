import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_x_b, stride_x_c, stride_x_l,
    stride_w_o, stride_w_c, stride_w_k,
    stride_out_b, stride_out_c, stride_out_l,
    B, C_in, C_out, L_in, L_out,
    kernel_size, dilation,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    l_start = pid_l * BLOCK_SIZE_L
    l_end = min(l_start + BLOCK_SIZE_L, L_out)

    K_eff = dilation * (kernel_size - 1) + 1

    # Load weights for current output channel
    # Weights shape: (C_out, C_in, K_eff)
    w_ptrs = w_ptr + pid_c * stride_w_o
    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    offsets_k = tl.arange(0, K_eff)
    mask_c = offsets_c < C_in
    mask_k = offsets_k < K_eff

    # Load weights into registers
    # w_vals shape: (C_in, K_eff)
    w_vals = tl.load(
        w_ptrs + offsets_c[:, None] * stride_w_c + offsets_k[None, :] * stride_w_k,
        mask=mask_c[:, None] & mask_k[None, :],
        other=0.0
    )

    # Initialize output accumulation
    # out_vals shape: (BLOCK_SIZE_L,)
    out_vals = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)

    # Iterate over effective kernel size
    for k in range(K_eff):
        # Compute input positions for current k
        # l_in = l_val - dilation * k
        l_val = l_start + tl.arange(0, BLOCK_SIZE_L)
        l_in = l_val[:, None] - dilation * k

        # Mask for valid input positions
        mask_l_in = (l_in >= 0) & (l_in < L_in)

        # Load input values
        # x shape: (B, C_in, L_in)
        x_ptrs = x_ptr + pid_b * stride_x_b
        # x_vals shape: (BLOCK_SIZE_L, C_in)
        x_vals = tl.load(
            x_ptrs + offsets_c[None, :] * stride_x_c + l_in * stride_x_l,
            mask=mask_l_in & mask_c[None, :],
            other=0.0
        )

        # Compute dot product over C_in and accumulate
        # w_vals[:, k] is (C_in,), x_vals is (BLOCK_SIZE_L, C_in)
        # We need sum over C_in: sum_c w[c, k] * x[c, l]
        # This results in shape (BLOCK_SIZE_L,)
        w_k = w_vals[:, k]
        out_vals += tl.sum(w_k[None, :] * x_vals, axis=1)

    # Add bias
    # b shape: (C_out,)
    b_val = tl.load(b_ptr + pid_c)
    out_vals += b_val

    # Store output
    # out shape: (B, C_out, L_out)
    out_ptrs = out_ptr + pid_b * stride_out_b + pid_c * stride_out_c + l_val * stride_out_l
    mask_l = l_val < l_end
    tl.store(out_ptrs, out_vals, mask=mask_l)


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, L_in = x.shape
    C_out, _, kernel_size = weight.shape
    dilation = (weight.shape[2] - 1) // (kernel_size - 1) if kernel_size > 1 else 1
    
    # Compute output length
    L_out = (L_in - 1) + dilation * (kernel_size - 1) + 1
    
    if bias is None:
        bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)
    bias = bias.contiguous()
    
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Strides
    stride_x_b, stride_x_c, stride_x_l = x.stride()
    stride_w_o, stride_w_c, stride_w_k = weight.stride()
    stride_out_b, stride_out_c, stride_out_l = out.stride()
    
    BLOCK_SIZE_L = 128
    BLOCK_SIZE_C = C_in
    
    grid = (B, C_out, (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        stride_x_b, stride_x_c, stride_x_l,
        stride_w_o, stride_w_c, stride_w_k,
        stride_out_b, stride_out_c, stride_out_l,
        B, C_in, C_out, L_in, L_out,
        kernel_size, dilation,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters to reconstruct kernel later
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(x, self.weight, self.bias)