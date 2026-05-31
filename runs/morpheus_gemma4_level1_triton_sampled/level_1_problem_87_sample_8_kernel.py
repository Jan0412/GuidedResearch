import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_kernel(
    x_ptr,      # Pointer to input tensor (B, C_in, H, W)
    w_ptr,      # Pointer to weight tensor (C_out, C_in)
    b_ptr,      # Pointer to bias tensor (C_out)
    out_ptr,    # Pointer to output tensor (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride_x_b, stride_x_c, stride_x_s, # Strides for x
    stride_out_b, stride_out_c, stride_out_s, # Strides for out
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Range of indices for this block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # N = B * H * W
    n_elements = B * H * W
    
    # Calculate batch and spatial indices for the current block of M
    # m = b * (H * W) + spatial_idx
    b_idx = rm // (H * W)
    spatial_idx = rm % (H * W)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension C_in
    for k in range(0, C_in, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # Load X block: (BLOCK_M, BLOCK_K)
        # x[b, c, h, w] -> b * stride_x_b + c * stride_x_c + spatial_idx
        x_offsets = (b_idx[:, None] * stride_x_b) + (rk[None, :] * stride_x_c) + spatial_idx[:, None]
        x_mask = (rm[:, None] < n_elements) & (rk[None, :] < C_in)
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Load W block: (BLOCK_K, BLOCK_N)
        # w[c_out, c_in] -> c_out * C_in + c_in
        w_offsets = (rn[None, :] * C_in) + rk[:, None]
        w_mask = (rk[:, None] < C_in) & (rn[None, :] < C_out)
        w = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # Matrix multiplication
        acc += tl.dot(x, w)

    # Load bias and add to result
    bias_offsets = rn
    bias_mask = rn < C_out
    bias = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
    acc += bias[None, :]

    # Store result: (BLOCK_M, BLOCK_N)
    # out[b, c_out, h, w] -> b * stride_out_b + c_out * stride_out_c + spatial_idx
    out_offsets = (b_idx[:, None] * stride_out_b) + (rn[None, :] * stride_out_c) + spatial_idx[:, None]
    out_mask = (rm[:, None] < n_elements) & (rn[None, :] < C_out)
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)

def triton_pointwise_conv(x, weight, bias):
    # x: (B, C_in, H, W)
    # weight: (C_out, C_in, 1, 1)
    # bias: (C_out)
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Flatten weight to (C_out, C_in)
    weight = weight.view(C_out, C_in).contiguous()
    x = x.contiguous()
    
    out = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)
    
    # Strides for indexing
    stride_x_b = C_in * H * W
    stride_x_c = H * W
    
    stride_out_b = C_out * H * W
    stride_out_c = H * W
    
    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    n_elements = B * H * W
    grid = (triton.cdiv(n_elements, BLOCK_M), triton.cdiv(C_out, BLOCK_N))
    
    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, H, W,
        stride_x_b, stride_x_c, 0, # stride_x_s not used explicitly
        stride_out_b, stride_out_c, 0, # stride_out_s not used explicitly
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use nn.Parameter to maintain parity with nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle bias for the Triton kernel
        bias_tensor = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        
        return triton_pointwise_conv(x, self.weight, bias_tensor)