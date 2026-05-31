import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N, C_in, C_out, D, H, W,
    K, stride, padding, groups,
    out_D, out_H, out_W,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Output coordinates
    d_out_start = pid_d * BLOCK_D
    h_out_start = pid_h * BLOCK_H
    w_out_start = pid_w * BLOCK_W

    d_out_offsets = d_out_start + tl.arange(0, BLOCK_D)
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_H)
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_W)

    # Masks for output block
    mask_d = d_out_offsets < out_D
    mask_h = h_out_offsets < out_H
    mask_w = w_out_offsets < out_W
    mask_out = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Accumulator for the output block
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Determine input channel range based on groups
    c_in_base = (pid_c // groups) * (C_in // groups)
    c_in_end = c_in_base + (C_in // groups)

    # Loop over input channels in chunks
    for c_in_start in range(c_in_base, c_in_end, BLOCK_C_IN):
        c_in_end_chunk = min(c_in_start + BLOCK_C_IN, c_in_end)
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
        mask_c = c_in_offsets < C_in

        # Input tile offsets
        d_in_start = d_out_start * stride + padding
        h_in_start = h_out_start * stride + padding
        w_in_start = w_out_start * stride + padding

        # We need to load a tile of input that covers the receptive field
        # Receptive field size is K
        d_in_offsets = d_in_start + tl.arange(0, BLOCK_D + K - 1)
        h_in_offsets = h_in_start + tl.arange(0, BLOCK_H + K - 1)
        w_in_offsets = w_in_start + tl.arange(0, BLOCK_W + K - 1)

        # Masks for input tile
        mask_d_in = d_in_offsets < D
        mask_h_in = h_in_offsets < H
        mask_w_in = w_in_offsets < W
        mask_in_tile = mask_d_in[:, None, None] & mask_h_in[None, :, None] & mask_w_in[None, None, :]

        # Load input tile
        # Input shape: (N, C_in, D, H, W)
        # We load (BLOCK_C_IN, BLOCK_D+K-1, BLOCK_H+K-1, BLOCK_W+K-1)
        # However, Triton loads are vectorized. We can load chunk by chunk or use masking.
        # To simplify, we load the whole tile with masking.
        # Note: Loading large tiles might be heavy, but masking handles bounds.
        # We assume memory is contiguous.
        # x_ptr offset: pid_n * C_in * D * H * W + c_in_offsets * D * H * W + ...
        # This is complex. Better to compute strides.
        # Strides for NCDHW: (C_in*D*H*W, D*H*W, H*W, W, 1)
        stride_n = C_in * D * H * W
        stride_c = D * H * W
        stride_d = H * W
        stride_h = W
        stride_w = 1

        x_offsets = (pid_n * stride_n + 
                     c_in_offsets[:, None, None, None] * stride_c + 
                     d_in_offsets[None, :, None, None] * stride_d + 
                     h_in_offsets[None, None, :, None] * stride_h + 
                     w_in_offsets[None, None, None, :] * stride_w)
        
        x_tile = tl.load(x_ptrs, mask=mask_c[:, None, None, None] & mask_in_tile[None, :, :, :], other=0.0)

        # Weight tile
        # Weight shape: (C_out, C_in, K, K, K)
        # We need weights for pid_c and c_in_offsets
        # w_offsets: pid_c * K^3 + c_in_offsets * K^3 + k_d * K^2 + k_h * K + k_w
        stride_w_c = K * K * K
        stride_w_d = K * K
        stride_w_h = K
        stride_w_w = 1

        k_d_offsets = tl.arange(0, K)
        k_h_offsets = tl.arange(0, K)
        k_w_offsets = tl.arange(0, K)

        w_offsets = (pid_c * stride_w_c + 
                     c_in_offsets[:, None, None, None] * stride_w_c + 
                     k_d_offsets[None, :, None, None] * stride_w_d + 
                     k_h_offsets[None, None, :, None] * stride_w_h + 
                     k_w_offsets[None, None, None, :] * stride_w_w)
        
        w_tile = tl.load(w_ptrs, mask=mask_c[:, None, None, None], other=0.0)

        # Compute partial sum
        # x_tile shape: (BLOCK_C_IN, BLOCK_D+K-1, BLOCK_H+K-1, BLOCK_W+K-1)
        # w_tile shape: (BLOCK_C_IN, K, K, K)
        # We need to sum over c_in, k_d, k_h, k_w
        # Result shape: (BLOCK_D+K-1, BLOCK_H+K-1, BLOCK_W+K-1)
        # Then we need to extract the valid output region (BLOCK_D, BLOCK_H, BLOCK_W)
        
        # This reduction is complex in Triton without loops.
        # We can loop over k_d, k_h, k_w.
        # Or use tl.reduce.
        # Given the complexity, a loop over k is often clearer and efficient enough.
        
        for k_d in range(K):
            for k_h in range(K):
                for k_w in range(K):
                    # x indices: d_out_start + k_d to d_out_start + k_d + BLOCK_D - 1?
                    # No, x_tile has extra K-1.
                    # x_tile[d, h, w] corresponds to d_in = d_in_start + d.
                    # We want d_in = d_out_start * stride + k_d + d - padding?
                    # Actually, d_out = (d_in - padding - k_d) / stride?
                    # Standard conv: out[d] = sum_k x[d + k - padding] * w[k]
                    # So x index for k_d is d_out + k_d - padding?
                    # Wait, d_out_start is the start of the output block.
                    # For output element d_out, we sum x[d_out + k_d - padding] * w[k_d].
                    # So x index is d_out_start + d + k_d - padding.
                    # But d_in_start = d_out_start * stride + padding.
                    # If stride=1, d_in_start = d_out_start + padding.
                    # x index = d_out_start + d + k_d - padding = d_in_start - padding + d + k_d - padding?
                    # This is getting confusing.
                    # Let's use the direct mapping.
                    # x index for output (d, h, w) and kernel (kd, kh, kw) is:
                    # d_in = d * stride + kd - padding
                    # h_in = h * stride + kh - padding
                    # w_in = w * stride + kw - padding
                    
                    # We have d_out_offsets, h_out_offsets, w_out_offsets.
                    # We can compute d_in_offsets for each k.
                    # d_in = d_out_offsets * stride + kd - padding
                    # This creates a new offset tensor.
                    
                    d_in = d_out_offsets * stride + k_d - padding
                    h_in = h_out_offsets * stride + k_h - padding
                    w_in = w_out_offsets * stride + k_w - padding
                    
                    # Load x values
                    # x shape: (N, C_in, D, H, W)
                    # We need to load x[pid_n, c_in, d_in, h_in, w_in]
                    # This requires masking against D, H, W.
                    
                    mask_d_in = d_in < D
                    mask_h_in = h_in < H
                    mask_w_in = w_in < W
                    mask_in = mask_d_in[:, None, None] & mask_h_in[None, :, None] & mask_w_in[None, None, :] & mask_c[:, None, None]
                    
                    # x_offsets calculation again for this specific k
                    x_k_offsets = (pid_n * stride_n + 
                                   c_in_offsets[:, None, None] * stride_c + 
                                   d_in[:, None, None] * stride_d + 
                                   h_in[None, :, None] * stride_h + 
                                   w_in[None, None, :] * stride_w)
                    
                    x_k = tl.load(x_k_offsets, mask=mask_in, other=0.0)
                    
                    # w_k is w_tile[c_in, k_d, k_h, k_w]
                    # w_tile shape: (BLOCK_C_IN, K, K, K)
                    # w_k = w_tile[c_in, k_d, k_h, k_w]
                    # This is a vector of size BLOCK_C_IN.
                    w_k = w_tile[:, k_d, k_h, k_w]
                    
                    # Accumulate
                    acc += x_k * w_k[:, None, None]

    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c)
        acc += bias

    # Store output
    out_offsets = (pid_n * C_out * out_D * out_H * out_W + 
                   pid_c * out_D * out_H * out_W + 
                   d_out_offsets[:, None, None] * out_H * out_W + 
                   h_out_offsets[None, :, None] * out_W + 
                   w_out_offsets[None, None, :])
    
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                  stride: int, padding: int, groups: int) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    N, C_in, D, H, W = x.shape
    C_out, C_in_w, K, _, _ = w.shape
    assert C_in == C_in_w
    
    # Output dimensions
    out_D = (D + 2 * padding - K) // stride + 1
    out_H = (H + 2 * padding - K) // stride + 1
    out_W = (W + 2 * padding - K) // stride + 1
    
    out = torch.empty((N, C_out, out_D, out_H, out_W), device=x.device, dtype=x.dtype)
    
    # Block sizes
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    BLOCK_C_IN = 8
    
    grid = (N, C_out, triton.cdiv(out_D, BLOCK_D), triton.cdiv(out_H, BLOCK_H), triton.cdiv(out_W, BLOCK_W))
    
    conv3d_kernel[grid](
        x, w, b, out,
        N, C_in, C_out, D, H, W,
        K, stride, padding, groups,
        out_D, out_H, out_W,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C_IN=BLOCK_C_IN
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv3d.weight
        b = self.conv3d.bias
        return triton_conv3d(x, w, b, self.stride, self.padding, self.groups)