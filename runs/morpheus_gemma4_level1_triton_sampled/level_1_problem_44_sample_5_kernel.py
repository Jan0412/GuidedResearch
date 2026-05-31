import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    B,              # Combined batch and channel dimension (batch_size * in_channels)
    L,              # Input length
    L_out,          # Output length
    k,              # Kernel size
    s,              # Stride
    p,              # Padding
    stride_x_b,     # Stride for batch dimension in x
    stride_x_l,     # Stride for length dimension in x
    stride_out_b,   # Stride for batch dimension in out
    stride_out_l,   # Stride for length dimension in out
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID: pid_b handles the batch*channel dimension, pid_l handles the output length
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Calculate the range of output indices this block is responsible for
    offsets = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < L_out

    # Pointers to the start of the current row (batch/channel)
    x_row_ptr = x_ptr + pid_b * stride_x_b
    out_row_ptr = out_ptr + pid_b * stride_out_b

    # Accumulator for the sum of elements in the pooling window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the kernel window size
    for i in range(k):
        # For each output element j, the corresponding input index is j * stride - padding + offset_i
        input_offsets = offsets * s - p + i
        # Mask to ensure we are within the bounds of the input tensor
        input_mask = mask & (input_offsets >= 0) & (input_offsets < L)
        # Load the input value (padding is treated as 0)
        val = tl.load(x_row_ptr + input_offsets * stride_x_l, mask=input_mask, other=0.0)
        acc += val

    # Compute the average (nn.AvgPool1d default: count_include_pad=True)
    out = acc / k

    # Store the result in the output tensor
    tl.store(out_row_ptr + offsets * stride_out_l, out, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton wrapper for 1D Average Pooling.
    """
    # Input shape: (batch_size, in_channels, input_length)
    N, C, L = x.shape
    B = N * C
    
    # Calculate output length
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    
    # Ensure input is contiguous and reshape for easier indexing in the kernel
    x = x.contiguous().view(B, L)
    out = torch.empty((B, L_out), device=x.device, dtype=x.dtype)
    
    # Strides
    stride_x_b = L
    stride_x_l = 1
    stride_out_b = L_out
    stride_out_l = 1
    
    # Tuning parameter
    BLOCK_SIZE = 256
    
    # Grid: (batch_size * in_channels, ceil(L_out / BLOCK_SIZE))
    grid = (B, triton.cdiv(L_out, BLOCK_SIZE))
    
    avg_pool1d_kernel[grid](
        x, out,
        B, L, L_out,
        kernel_size, stride, padding,
        stride_x_b, stride_x_l,
        stride_out_b, stride_out_l,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to (batch_size, in_channels, output_length)
    return out.view(N, C, L_out)


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        # Ensure the input is on GPU for Triton
        if not x.is_cuda:
            x = x.cuda()
            
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)