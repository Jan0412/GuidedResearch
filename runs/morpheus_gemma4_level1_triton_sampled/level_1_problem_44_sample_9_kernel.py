import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    L,
    L_out,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    # Program ID for batch * channel and the output length dimension
    pid_bc = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Calculate the base offset for the current batch and channel
    # x shape: (B, C, L)
    # out shape: (B, C, L_out)
    x_base_offset = pid_bc * L
    out_base_offset = pid_bc * L_out

    # Range of output indices handled by this block
    l_offsets = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = l_offsets < L_out

    # Accumulator for the sum of the window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the kernel window
    for k in range(KERNEL_SIZE):
        # Calculate the input indices for the current window offset k
        # index = l * stride - padding + k
        load_offsets = l_offsets * stride - padding + k
        
        # Mask for valid input indices (handling padding)
        load_mask = mask & (load_offsets >= 0) & (load_offsets < L)
        
        # Load values from input; out-of-bounds (padding) are treated as 0.0
        val = tl.load(x_ptr + x_base_offset + load_offsets, mask=load_mask, other=0.0)
        acc += val

    # Compute the average and store the result
    # PyTorch AvgPool1d with count_include_pad=True divides by the full kernel size
    out = acc / KERNEL_SIZE
    tl.store(out_ptr + out_base_offset + l_offsets, out, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Ensure input is contiguous on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, L = x.shape
    # Calculate output length
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Tuning parameter
    BLOCK_SIZE = 256
    
    # Grid: parallelize over (B * C) and the output length
    grid = (B * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the Triton kernel
    avg_pool1d_kernel[grid](
        x, 
        out, 
        B, 
        C, 
        L, 
        L_out, 
        stride, 
        padding, 
        BLOCK_SIZE=BLOCK_SIZE, 
        KERNEL_SIZE=kernel_size
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)