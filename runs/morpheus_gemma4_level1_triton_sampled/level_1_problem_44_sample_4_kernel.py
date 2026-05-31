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
    S,
    P,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the combined (batch * channel) dimension
    pid_bc = tl.program_id(0)
    # Program ID for the output length dimension
    pid_l = tl.program_id(1)

    # Calculate the range of output indices this block handles
    offsets_out = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offsets_out < L_out

    # Pointers to the start of the current (batch, channel) slice
    # x is (N, C, L), out is (N, C, L_out)
    x_slice_ptr = x_ptr + pid_bc * L
    out_slice_ptr = out_ptr + pid_bc * L_out

    # Accumulator for the sum of elements in the pooling window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the kernel window
    for k in range(K):
        # For each output index i, the window starts at i * S - P
        # The k-th element in the window is at index i * S - P + k
        input_offsets = offsets_out * S - P + k
        
        # Create a mask to handle padding (values outside [0, L-1] are treated as 0)
        mask_in = (input_offsets >= 0) & (input_offsets < L)
        
        # Load values from input, using 0.0 for padded regions
        vals = tl.load(x_slice_ptr + input_offsets, mask=mask_in, other=0.0)
        acc += vals

    # Average pooling: divide by the kernel size (standard PyTorch behavior with count_include_pad=True)
    out = acc / K
    
    # Store the result in the output tensor
    tl.store(out_slice_ptr + offsets_out, out, mask=mask_out)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton wrapper for 1D Average Pooling.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 256
    
    # Grid: (batch * channels, ceil(output_length / BLOCK_SIZE))
    grid = (batch_size * in_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    avg_pool1d_kernel[grid](
        x, 
        out, 
        input_length, 
        output_length, 
        stride, 
        padding, 
        K=kernel_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using custom Triton kernels.
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