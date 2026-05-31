import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr, 
    out_ptr,
    batch_size, 
    in_channels, 
    input_length, 
    output_length,
    kernel_size, 
    stride, 
    padding,
    stride_x_batch, 
    stride_x_channel, 
    stride_x_len,
    stride_out_batch, 
    stride_out_channel, 
    stride_out_len,
    BLOCK_SIZE: tl.constexpr,
):
    # pid 0 handles the combination of batch and channel
    pid_bc = tl.program_id(0)
    # pid 1 handles the output length dimension in blocks
    pid_l = tl.program_id(1)

    batch_idx = pid_bc // in_channels
    channel_idx = pid_bc % in_channels

    # Compute the range of output indices this program handles
    offsets = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length

    # Pointers to the start of the current batch and channel
    x_base = x_ptr + batch_idx * stride_x_batch + channel_idx * stride_x_channel
    out_base = out_ptr + batch_idx * stride_out_batch + channel_idx * stride_out_channel

    # Accumulator for the sum of the window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop through the kernel window
    for k in range(kernel_size):
        # Calculate the input index for each element in the output block
        # formula: output_idx * stride - padding + offset_in_kernel
        load_offsets = offsets * stride - padding + k
        
        # Mask to handle padding (zero-padding) and boundaries
        load_mask = mask & (load_offsets >= 0) & (load_offsets < input_length)
        
        # Load values from input; out-of-bounds are treated as 0.0
        val = tl.load(x_base + load_offsets * stride_x_len, mask=load_mask, other=0.0)
        acc += val

    # Compute average (PyTorch AvgPool1d divides by kernel_size by default)
    out = acc / kernel_size
    
    # Store the result
    tl.store(out_base + offsets * stride_out_len, out, mask=mask)

def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)

    # Strides for input and output tensors
    stride_x_batch, stride_x_channel, stride_x_len = x.stride()
    stride_out_batch, stride_out_channel, stride_out_len = out.stride()

    BLOCK_SIZE = 256
    # Grid: (Batch * Channels, Output Length / BLOCK_SIZE)
    grid = (batch_size * in_channels, triton.cdiv(output_length, BLOCK_SIZE))

    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        stride_x_batch, stride_x_channel, stride_x_len,
        stride_out_batch, stride_out_channel, stride_out_len,
        BLOCK_SIZE=BLOCK_SIZE,
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