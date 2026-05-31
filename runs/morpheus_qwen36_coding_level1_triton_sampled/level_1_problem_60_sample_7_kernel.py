import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out, W, H, D, K_W, K_H, K_D,
    S_W, S_H, S_D, P_W, P_H, P_D,
    D_W, D_H, D_D, G,
    stride_w, stride_h, stride_d,
    n_elements, BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Calculate output dimensions
    W_out = (W + 2 * P_W - D_W * (K_W - 1) - 1) // S_W + 1
    H_out = (H + 2 * P_H - D_H * (K_H - 1) - 1) // S_H + 1
    D_out = (D + 2 * P_D - D_D * (K_D - 1) - 1) // S_D + 1

    # Program ID mapping
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # Calculate base index for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode output coordinates for each element in the block
    # Flattened index -> (b, c_out, w_out, h_out, d_out)
    idx = offsets
    d_out = idx % D_out
    idx = idx // D_out
    h_out = idx % H_out
    idx = idx // H_out
    w_out = idx % W_out
    idx = idx // W_out
    c_out = idx % C_out
    b = idx // C_out

    # Pointer arithmetic for inputs and weights
    # x: (B, C_in, W, H, D)
    # w: (C_out, C_in, K_W, K_H, K_D)
    # out: (B, C_out, W_out, H_out, D_out)
    
    x_base = b * C_in * W * H * D + c_out * W * H * D  # Simplified base, groups handled below
    # Note: Groups affect channel mapping. For group G, c_in belongs to group c_out // G.
    # Input channel range for this output channel:
    c_in_start = (c_out // G) * (C_in // G)
    c_in_end = c_in_start + (C_in // G)
    
    w_base = c_out * C_in * K_W * K_H * K_D
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Bias addition
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out, mask=mask, other=0.0)
        acc = acc + bias

    # Tiled loop over input channels
    for c_in in range(c_in_start, c_in_end, BLOCK_C):
        c_in_block = tl.arange(0, BLOCK_C)
        c_in_mask = (c_in + c_in_block) < c_in_end
        
        # Load input tile
        # Input spatial coordinates for the receptive field
        # w_in = w_out * S_W - P_W + k_w * D_W
        # We need to load the volume corresponding to the kernel
        # To optimize, we load the input volume for the current block of output elements
        # and the corresponding weight tile.
        
        # Calculate input spatial offsets for the kernel
        # This part is complex for 3D. A direct approach:
        # For each element in the output block, we need to access input channels.
        # We can vectorize over the kernel elements if small, or loop.
        # Given constraints, we'll loop over kernel elements but vectorize over channels.
        
        # Load weights for current channel block
        w_offsets = c_in + c_in_block
        w_mask = w_offsets < C_in
        w_ptrs = w_ptr + w_base + w_offsets[:, None, None, None, None] * K_W * K_H * K_D + \
                 tl.zeros((BLOCK_SIZE, K_W, K_H, K_D), dtype=tl.int64)
        # This pointer math is tricky in Triton for 5D. 
        # Alternative: Load weights in a flattened manner or use tl.gather.
        # For functional correctness and compilation, we use a simpler access pattern.
        
        # Simplified weight loading:
        # w shape is (C_out, C_in, K_W, K_H, K_D).
        # We want w[c_out, c_in_block, k_w, k_h, k_d].
        # We can load this into a register block if BLOCK_C and K are small.
        # Otherwise, we load incrementally.
        
        # Let's assume we load weights for the current c_in_block for all k.
        # This requires reshaping or careful indexing.
        # For this implementation, we'll perform the dot product element-wise in a loop 
        # to ensure correctness and compilation, optimizing memory access via masking.
        
        for k_w in range(K_W):
            for k_h in range(K_H):
                for k_d in range(K_D):
                    # Input spatial coordinates
                    w_in = w_out * S_W - P_W + k_w * D_W
                    h_in = h_out * S_H - P_H + k_h * D_H
                    d_in = d_out * S_D - P_D + k_d * D_D
                    
                    # Mask for valid input coordinates
                    valid_in = (w_in >= 0) & (w_in < W) & (h_in >= 0) & (h_in < H) & (d_in >= 0) & (d_in < D)
                    
                    # Load input
                    x_idx = b * C_in * W * H * D + (c_in + c_in_block) * W * H * D + w_in * H * D + h_in * D + d_in
                    x_vals = tl.load(x_ptr + x_idx, mask=valid_in[:, None], other=0.0)
                    
                    # Load weight
                    w_idx = c_out * C_in * K_W * K_H * K_D + (c_in + c_in_block) * K_W * K_H * K_D + k_w * K_H * K_D + k_h * K_D + k_d
                    w_vals = tl.load(w_ptr + w_idx, mask=c_in_mask, other=0.0)
                    
                    # Accumulate
                    acc = acc + tl.sum(x_vals * w_vals, axis=0)

    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None,
                  stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
        
    B, C_in, W, H, D = x.shape
    C_out, C_in_w, K_W, K_H, K_D = w.shape
    assert C_in == C_in_w
    assert C_in % groups == 0
    assert C_out % groups == 0
    
    S_W, S_H, S_D = stride
    P_W, P_H, P_D = padding
    D_W, D_H, D_D = dilation
    
    W_out = (W + 2 * P_W - D_W * (K_W - 1) - 1) // S_W + 1
    H_out = (H + 2 * P_H - D_H * (K_H - 1) - 1) // S_H + 1
    D_out = (D + 2 * P_D - D_D * (K_D - 1) - 1) // S_D + 1
    
    out = torch.empty((B, C_out, W_out, H_out, D_out), dtype=x.dtype, device=x.device)
    
    n_elements = out.numel()
    BLOCK_SIZE = 128
    BLOCK_C = 4  # Tunable block size for channels
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv3d_kernel[grid](
        x, w, b, out,
        B, C_in, C_out, W, H, D, K_W, K_H, K_D,
        S_W, S_H, S_D, P_W, P_H, P_D,
        D_W, D_H, D_D, groups,
        S_W, S_H, S_D,
        n_elements, BLOCK_SIZE, BLOCK_C
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias if self.bias is not None else None
        return triton_conv3d(
            x, self.weight, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )