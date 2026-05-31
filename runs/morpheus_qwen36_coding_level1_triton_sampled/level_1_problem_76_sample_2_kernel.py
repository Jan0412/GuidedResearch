import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, C_in, L_in, C_out, K, stride, dilation, L_out,
    has_bias,
    BLOCK_SIZE_L: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    block_l_idx = tl.program_id(2)

    out_pos_start = block_l_idx * BLOCK_SIZE_L
    out_pos_offsets = out_pos_start + tl.arange(0, BLOCK_SIZE_L)
    mask_out = out_pos_offsets < L_out

    # Load weights for this output channel
    # Weight shape: (C_out, C_in, K)
    # We load C_in * K elements and reshape
    offsets_w = tl.arange(0, C_in * K)
    mask_w = offsets_w < C_in * K
    W_flat = tl.load(weight_ptr + out_ch_idx * C_in * K + offsets_w, mask=mask_w, other=0.0)
    W = tl.reshape(W_flat, (C_in, K))

    # Load bias if present
    bias_val = 0.0
    if has_bias:
        bias_val = tl.load(bias_ptr + out_ch_idx)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)

    # Loop over kernel positions
    for k in range(K):
        # Compute input position for each output position
        # l_in = l_out * stride - dilation * k
        l_in = out_pos_offsets * stride - k * dilation
        
        # Mask for valid input positions
        mask_l = (l_in >= 0) & (l_in < L_in)
        
        if tl.sum(mask_l) > 0:
            # Load input block for this kernel position
            # x shape: (B, C_in, L_in)
            # We want x[batch_idx, :, l_in]
            # Base pointer for batch
            base_ptr = x_ptr + batch_idx * C_in * L_in
            # Offsets for channels: c_in * L_in
            c_in_offsets = tl.arange(0, C_in)
            # Memory addresses: base + c_in * L_in + l_in
            ptr = base_ptr + c_in_offsets * L_in + l_in
            X_block = tl.load(ptr, mask=mask_l, other=0.0)
            
            # Weight column for this k: W[:, k]
            W_col = W[:, k]
            
            # Dot product: sum over C_in
            acc += tl.sum(X_block * W_col, axis=0)

    # Add bias and store
    acc += bias_val
    # Output shape: (B, C_out, L_out)
    # ptr_out = out_ptr + batch_idx * C_out * L_out + out_ch_idx * L_out + out_pos
    ptr_out = out_ptr + batch_idx * C_out * L_out + out_ch_idx * L_out + out_pos_offsets
    tl.store(ptr_out, acc, mask=mask_out)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: int = 1, dilation: int = 1) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, L_in = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w
    
    if bias is not None:
        bias = bias.contiguous()
        assert bias.shape[0] == C_out
        has_bias = True
    else:
        has_bias = False
        
    # Calculate output length
    L_out = (L_in - dilation * (K - 1) - 1) // stride + 1
    
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_L = 128
    num_blocks_L = (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    
    grid = (B, C_out, num_blocks_L)
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, L_in, C_out, K, stride, dilation, L_out,
        has_bias,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.has_bias = bias
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)