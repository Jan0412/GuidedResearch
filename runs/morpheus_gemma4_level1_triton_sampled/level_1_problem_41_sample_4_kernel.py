import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    L,
    L_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Parallelize over (batch * channels) and output sequence length
    pid_bc = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Sequence offsets for this block
    l_offsets = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    l_mask = l_offsets < L_out

    # Initialize max values to -infinity
    max_vals = tl.full((BLOCK_SIZE_L,), -float('inf'), dtype=tl.float32)

    # Iterate over the kernel window
    for k in range(kernel_size):
        # Calculate the input index for each element in the block
        # Formula: input_idx = output_idx * stride - padding + k * dilation
        idx = l_offsets * stride - padding + k * dilation
        
        # Mask for valid input indices (handling padding)
        idx_mask = l_mask & (idx >= 0) & (idx < L)
        
        # Load values from the input tensor
        # Offset = (batch_idx * C * L) + (channel_idx * L) + input_idx
        # pid_bc already represents (batch_idx * C + channel_idx)
        vals = tl.load(x_ptr + pid_bc * L + idx, mask=idx_mask, other=-float('inf'))
        
        # Compute the running maximum
        max_vals = tl.maximum(max_vals, vals)

    # Store the final max values to the output tensor
    # Offset = (batch_idx * C * L_out) + (channel_idx * L_out) + output_idx
    tl.store(out_ptr + pid_bc * L_out + l_offsets, max_vals, mask=l_mask)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    # Ensure input is contiguous and on GPU
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, L = x.shape
    
    # Calculate output length
    L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_L = 128
    
    # Grid: (B * C, ceil(L_out / BLOCK_SIZE_L))
    grid = (B * C, (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L)

    maxpool1d_kernel[grid](
        x, out,
        B, C, L, L_out,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        # Keep the original layer as a fallback for return_indices=True
        if self.return_indices:
            self.maxpool = nn.MaxPool1d(
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                return_indices=return_indices
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.
        """
        if self.return_indices:
            return self.maxpool(x)
        
        # Use the optimized Triton kernel for the common case (return_indices=False)
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )