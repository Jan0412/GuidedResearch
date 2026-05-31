import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, 
    out_ptr,
    batch_size, 
    features, 
    input_length, 
    output_length,
    kernel_size, 
    stride, 
    padding, 
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the output sequence for a specific batch and feature
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Map pid_0 to batch and feature indices
    batch = pid_0 // features
    channel = pid_0 % features

    # Pointers to the start of the current batch and channel
    # x shape: (batch_size, features, input_length)
    # out shape: (batch_size, features, output_length)
    x_ptr_base = x_ptr + batch * features * input_length + channel * input_length
    out_ptr_base = out_ptr + batch * features * output_length + channel * output_length

    # Calculate output offsets for this block
    out_offsets = pid_1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < output_length

    # Initialize max_val with negative infinity
    max_val = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)

    # Iterate over the kernel window
    for k in range(kernel_size):
        # Calculate input index for the k-th element of the kernel
        # Formula: input_idx = out_idx * stride + k * dilation - padding
        input_idx = out_offsets * stride + k * dilation - padding
        
        # Load values from input, using mask to handle padding and boundaries
        # If the index is out of bounds [0, input_length), it is treated as padding (-inf)
        load_mask = mask & (input_idx >= 0) & (input_idx < input_length)
        val = tl.load(x_ptr_base + input_idx, mask=load_mask, other=float("-inf"))
        
        # Update the maximum value
        max_val = tl.maximum(max_val, val)

    # Store the final max values into the output tensor
    tl.store(out_ptr_base + out_offsets, max_val, mask=mask)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton wrapper for MaxPool1d.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, features, input_length = x.shape
    
    # Calculate output length based on PyTorch formula
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, features, output_length), device=x.device, dtype=x.dtype)

    # Tunable parameter for block size
    BLOCK_SIZE = 1024

    # Grid: (batch * features) x (output_length / BLOCK_SIZE)
    grid = (batch_size * features, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)

    maxpool1d_kernel[grid](
        x, out,
        batch_size, features, input_length, output_length,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        # Note: The custom kernel currently supports return_indices=False.
        # If return_indices=True were required, a separate kernel would be needed to track indices.
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )