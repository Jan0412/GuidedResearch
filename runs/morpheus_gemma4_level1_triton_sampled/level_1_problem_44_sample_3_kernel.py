import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr, 
    out_ptr, 
    B, C, L_in, L_out, 
    kernel_size, stride, padding, 
    BLOCK_SIZE: tl.constexpr
):
    # Program ID for the batch and channel dimension
    pid_bc = tl.program_id(0)
    # Program ID for the output length dimension
    pid_l = tl.program_id(1)

    # Pointers to the start of the current sequence (B, C)
    x_base_ptr = x_ptr + pid_bc * L_in
    out_base_ptr = out_ptr + pid_bc * L_out

    # Range of output elements this program handles
    l_out_offsets = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_l_out = l_out_offsets < L_out

    # Accumulator for the sum of elements in the pooling window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the kernel window
    for i in range(kernel_size):
        # Calculate the input index for each output element in the block
        input_offsets = l_out_offsets * stride - padding + i
        # Create mask for input bounds (handle padding as 0)
        input_mask = (input_offsets >= 0) & (input_offsets < L_in) & mask_l_out
        # Load values from the input tensor
        vals = tl.load(x_base_ptr + input_offsets, mask=input_mask, other=0.0)
        acc += vals

    # Compute average (PyTorch AvgPool1d default: count_include_pad=True)
    out = acc / kernel_size
    # Store the result in the output tensor
    tl.store(out_base_ptr + l_out_offsets, out, mask=mask_l_out)

def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton wrapper for 1D Average Pooling.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, L_in = x.shape
    # Calculate output length
    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    # Grid: (Batch * Channels, Output Length Blocks)
    grid = (B * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    avg_pool1d_kernel[grid](
        x, out, 
        B, C, L_in, L_out, 
        kernel_size, stride, padding, 
        BLOCK_SIZE=BLOCK_SIZE
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
        Applies 1D Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)