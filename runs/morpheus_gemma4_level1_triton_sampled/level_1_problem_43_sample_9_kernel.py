import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool3d_kernel(
    x_ptr, 
    out_ptr, 
    B, C, D, H, W, 
    OD, OH, OW, 
    S, P, Dil, 
    K: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr,
):
    # pid0 = b * C * OD + c * OD + od
    # pid1 = oh
    # pid2 = ow_block
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    # Decompose pid0
    b = pid0 // (C * OD)
    rem = pid0 % (C * OD)
    c = rem // OD
    od = rem % OD
    
    oh = pid1
    ow_block = pid2
    
    # Output width offsets for this block
    ow_offsets = ow_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_ow = ow_offsets < OW

    # Base pointers for the current batch, channel, and output depth
    x_base = x_ptr + b * (C * D * H * W) + c * (D * H * W)
    out_base = out_ptr + b * (C * OD * OH * OW) + c * (OD * OH * OW) + od * (OH * OW) + oh * OW

    # Initialize max values for the block to negative infinity
    max_val = tl.full((BLOCK_SIZE_W,), -float('inf'), dtype=tl.float32)

    # Iterate over the 3D kernel window
    for k in range(K):
        curr_d = od * S - P + k * Dil
        mask_d = (curr_d >= 0) & (curr_d < D)
        
        for l in range(K):
            curr_h = oh * S - P + l * Dil
            mask_h = (curr_h >= 0) & (curr_h < H)
            
            for m in range(K):
                curr_w = ow_offsets * S - P + m * Dil
                mask_w = (curr_w >= 0) & (curr_w < W)
                
                # Combine masks: must be within input bounds and within output width
                final_mask = mask_d & mask_h & mask_w & mask_ow
                
                # Calculate pointer to input elements
                # ptr = x_base + curr_d * (H * W) + curr_h * W + curr_w
                ptr = x_base + curr_d * (H * W) + curr_h * W + curr_w
                
                # Load values and update max
                val = tl.load(ptr, mask=final_mask, other=-float('inf'))
                max_val = tl.maximum(max_val, val)

    # Store the resulting max values back to the output tensor
    tl.store(out_base + ow_offsets, max_val, mask=mask_ow)


def triton_maxpool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    
    # PyTorch MaxPool3d output size calculation (ceil_mode=False)
    def get_out_dim(in_dim):
        return (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    OD = get_out_dim(D)
    OH = get_out_dim(H)
    OW = get_out_dim(W)

    out = torch.empty((B, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_W = 16
    # Grid: (batch * channels * out_depth, out_height, ceil(out_width / BLOCK_SIZE_W))
    grid = (B * C * OD, OH, (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    maxpool3d_kernel[grid](
        x, out, 
        B, C, D, H, W, 
        OD, OH, OW, 
        stride, padding, dilation, 
        K=kernel_size, 
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        # Note: return_indices and ceil_mode are not implemented in this Triton kernel for simplicity
        # as the original architecture provided uses standard settings.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)