import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    L,
    L_out,
    stride,
    padding,
    kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the output length for one (batch, channel) pair
    pid_bc = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Pointers to the start of the current batch/channel
    x_ptr_base = x_ptr + pid_bc * L
    out_ptr_base = out_ptr + pid_bc * L_out

    # Output offsets for this block
    out_offsets = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = out_offsets < L_out

    # Accumulator for the sum
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate through the pooling window
    # Note: kernel_size is passed as a value, Triton handles the loop
    for k in range(kernel_size):
        # Calculate input indices for the current window element
        in_offsets = out_offsets * stride - padding + k
        # Mask to handle padding (values outside [0, L-1] are treated as 0)
        mask_in = (in_offsets >= 0) & (in_offsets < L)
        # Load values and add to accumulator
        vals = tl.load(x_ptr_base + in_offsets, mask=mask_in, other=0.0)
        acc += vals

    # Compute average
    out = acc / kernel_size
    # Store the result
    tl.store(out_ptr_base + out_offsets, out, mask=mask_out)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton wrapper for 1D Average Pooling.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, L = x.shape
    # Calculate output length
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Hyperparameters
    BLOCK_SIZE = 256
    
    # Grid: (Batch * Channels, Output Length / BLOCK_SIZE)
    grid = (B * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    avg_pool1d_kernel[grid](
        x, 
        out, 
        L, 
        L_out, 
        stride, 
        padding, 
        kernel_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using Triton kernels.
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
        Applies 1D Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)