import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, C_out, H, W, K_h, K_w,
    H_out, W_out,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid indices
    pid0 = tl.program_id(0)  # N
    pid1 = tl.program_id(1)  # C_out
    pid2 = tl.program_id(2)  # Block H
    pid3 = tl.program_id(3)  # Block W
    
    h_start = pid2 * BLOCK_H
    w_start = pid3 * BLOCK_W
    
    # Output indices within the block
    h_idx = h_start + tl.arange(0, BLOCK_H)
    w_idx = w_start + tl.arange(0, BLOCK_W)
    
    # Output mask
    h_mask = h_idx < H_out
    w_mask = w_idx < W_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Pointer to kernel weights for current c_out
    w_c_out_ptr = w_ptr + pid1 * C_in * K_h * K_w
    
    # Loop over input channels
    for c_in in range(C_in):
        # Pointer to input for current n and c_in
        x_c_in_ptr = x_ptr + pid0 * C_in * H * W + c_in * H * W
        
        # Load input patch
        # Patch size: (BLOCK_H + K_h - 1) x (BLOCK_W + K_w - 1)
        patch_h = BLOCK_H + K_h - 1
        patch_w = BLOCK_W + K_w - 1
        
        # Create 2D offsets for patch
        row_offsets = h_start + tl.arange(0, patch_h)
        col_offsets = w_start + tl.arange(0, patch_w)
        
        # Flatten offsets to 1D for loading
        # Memory layout is row-major: offset = row * W + col
        patch_offsets = row_offsets[:, None] * W + col_offsets[None, :]
        
        # Patch mask
        row_mask = row_offsets < H
        col_mask = col_offsets < W
        patch_mask = row_mask[:, None] & col_mask[None, :]
        
        # Load patch
        patch = tl.load(x_c_in_ptr + patch_offsets, mask=patch_mask, other=0.0)
        
        # Loop over kernel dimensions
        for kh in range(K_h):
            for kw in range(K_w):
                # Load kernel weight
                w_val = tl.load(w_c_out_ptr + c_in * K_h * K_w + kh * K_w + kw)
                
                # Compute shifted indices for patch
                # For output (h, w), we need patch at (h - h_start + kh, w - w_start + kw)
                # This is equivalent to adding kh, kw to the base indices
                # We can compute the offset in the patch array
                # patch_row = h - h_start + kh
                # patch_col = w - w_start + kw
                # offset = patch_row * patch_w + patch_col
                
                # We can compute this for all h, w in the block
                # h_idx and w_idx are 1D arrays of shape (BLOCK_H,) and (BLOCK_W,)
                # We need to broadcast them
                h_idx_2d = h_idx[:, None]
                w_idx_2d = w_idx[None, :]
                
                patch_row = h_idx_2d - h_start + kh
                patch_col = w_idx_2d - w_start + kw
                
                # Compute linear offset in patch
                patch_offset = patch_row * patch_w + patch_col
                
                # Load from patch
                # We need a mask for patch access? The patch is already loaded with mask, so out-of-bound values are 0.
                # But we need to ensure we don't access out of patch bounds?
                # The patch is loaded with size patch_h x patch_w, and we access within this range because kh, kw are within kernel size.
                # So patch_offset is within [0, patch_h * patch_w - 1].
                # So no additional mask needed.
                
                val = tl.load(patch + patch_offset)
                
                # Accumulate
                acc += val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid1)
        acc += bias
    
    # Store output
    out_ptr_offset = pid0 * C_out * H_out * W_out + pid1 * H_out * W_out
    tl.store(out_ptr + out_ptr_offset + h_idx_2d * W_out + w_idx_2d, acc, mask=mask)


def triton_transposed_conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """
    Performs transposed 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        w: Weight tensor of shape (C_out, C_in, K_h, K_w)
        b: Bias tensor of shape (C_out,)
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape
    
    # Compute output dimensions
    H_out = H + K_h - 1
    W_out = W + K_w - 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Grid
    grid = (N, C_out, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    # Launch kernel
    transposed_conv_kernel[grid](
        x, w, b, out,
        N, C_in, C_out, H, W, K_h, K_w,
        H_out, W_out,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for kernel launch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous() if self.bias is not None else None
        
        # Call Triton kernel
        return triton_transposed_conv(x, w, b)