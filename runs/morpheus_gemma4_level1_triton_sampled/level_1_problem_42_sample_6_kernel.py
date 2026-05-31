import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool_kernel(
    input_ptr, 
    output_ptr,
    N, C, H, W,
    OH, OW,
    S, P, D,
    BLOCK_SIZE_W: tl.constexpr,
    K: tl.constexpr,
):
    # Program IDs
    pid_h = tl.program_id(0)  # maps to (batch * channels * OH)
    pid_w = tl.program_id(1)  # maps to OW block

    # Decompose pid_h to get batch, channel, and output height indices
    batch_idx = pid_h // (C * OH)
    chan_idx = (pid_h // OH) % C
    oh_idx = pid_h % OH

    # Output width offsets for the current block
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_ow = ow_offsets < OW

    # Initialize max value to negative infinity
    max_val = tl.full([BLOCK_SIZE_W], float("-inf"), dtype=tl.float32)

    # Iterate over the pooling window
    for i in range(K):
        h_idx = oh_idx * S - P + i * D
        # Check if the height index is within the input boundaries
        if h_idx >= 0 and h_idx < H:
            # Pointer to the start of the row in the input tensor for this batch and channel
            # Input layout: (N, C, H, W)
            row_ptr = input_ptr + (batch_idx * C * H * W) + (chan_idx * H * W) + (h_idx * W)
            
            for j in range(K):
                # Calculate width indices for the block
                w_idx = ow_offsets * S - P + j * D
                # Mask to ensure width indices are within bounds and within the output width
                mask_w = (w_idx >= 0) & (w_idx < W) & mask_ow
                
                # Load values from the input tensor
                vals = tl.load(row_ptr + w_idx, mask=mask_w, other=float("-inf"))
                # Update the running maximum
                max_val = tl.maximum(max_val, vals)

    # Calculate output pointer offset
    # Output layout: (N, C, OH, OW)
    out_offset = (batch_idx * C * OH * OW) + (chan_idx * OH * OW) + (oh_idx * OW) + ow_offsets
    tl.store(output_ptr + out_offset, max_val, mask=mask_ow)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton wrapper for MaxPool2d.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_W = 128
    # Grid: (batch * channels * OH, ceil(OW / BLOCK_SIZE_W))
    grid = (N * C * OH, triton.cdiv(OW, BLOCK_SIZE_W))
    
    maxpool_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        stride, padding, dilation,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        K=kernel_size
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using the Triton implementation.
        """
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )