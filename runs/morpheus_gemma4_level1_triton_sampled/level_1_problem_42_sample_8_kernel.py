import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    B, C, H, W,
    oH, oW,
    S_C, S_H,
    out_S_C, out_S_H,
    stride, padding, dilation,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Each program handles a block of output spatial dimensions for a specific (batch, channel)
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Calculate output coordinates
    oh_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for output boundaries
    oh_mask = oh_offsets < oH
    ow_mask = ow_offsets < oW
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    # Pointers to the input and output slices for this (batch, channel)
    x_base_ptr = x_ptr + pid_bc * S_C
    out_base_ptr = out_ptr + pid_bc * out_S_C

    # Initialize max values to negative infinity
    max_val = tl.full((BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)

    # Loop over the pooling window
    for i in range(K):
        # Calculate height index in input
        h_idx = oh_offsets * stride - padding + i * dilation
        h_mask = (h_idx >= 0) & (h_idx < H)
        
        for j in range(K):
            # Calculate width index in input
            w_idx = ow_offsets * stride - padding + j * dilation
            w_mask = (w_idx >= 0) & (w_idx < W)
            
            # Combined mask for valid input access
            load_mask = h_mask[:, None] & w_mask[None, :]
            
            # Calculate pointer for the window element
            # x_base_ptr + h_idx * S_H + w_idx
            ptr = x_base_ptr + h_idx[:, None] * S_H + w_idx[None, :]
            
            # Load and update max
            vals = tl.load(ptr, mask=load_mask, other=-float('inf'))
            max_val = tl.maximum(max_val, vals)

    # Store the resulting max values
    out_ptr_final = out_base_ptr + oh_offsets[:, None] * out_S_H + ow_offsets[None, :]
    tl.store(out_ptr_final, max_val, mask=out_mask)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    
    # Calculate output dimensions
    oH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    oW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((B, C, oH, oW), device=x.device, dtype=x.dtype)
    
    # Strides
    S_C = H * W
    S_H = W
    out_S_C = oH * oW
    out_S_H = oW
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        B * C, 
        (oH + BLOCK_H - 1) // BLOCK_H, 
        (oW + BLOCK_W - 1) // BLOCK_W
    )
    
    maxpool2d_kernel[grid](
        x, out,
        B, C, H, W,
        oH, oW,
        S_C, S_H,
        out_S_C, out_S_H,
        stride, padding, dilation,
        K=kernel_size,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)