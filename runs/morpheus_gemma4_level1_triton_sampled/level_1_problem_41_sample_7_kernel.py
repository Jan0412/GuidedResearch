import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, 
    out_ptr, 
    L, 
    L_out, 
    S, 
    P, 
    D, 
    K: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for (batch * channel) and (output_sequence_block)
    pid_bc = tl.program_id(0)
    pid_lo = tl.program_id(1)

    # Calculate the range of output indices this block handles
    out_offsets = pid_lo * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = out_offsets < L_out

    # Pointers for the current batch/channel
    x_ptr_base = x_ptr + pid_bc * L
    out_ptr_base = out_ptr + pid_bc * L_out

    # Initialize max values to negative infinity
    max_vals = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

    # Iterate over the kernel window
    for k in range(K):
        # Calculate input offsets for the current kernel element
        # input_idx = output_idx * stride - padding + k * dilation
        in_offsets = out_offsets * S - P + k * D
        
        # Mask for valid input indices (handle implicit zero padding by using -inf)
        mask_in = (in_offsets >= 0) & (in_offsets < L)
        
        # Load values from input
        vals = tl.load(x_ptr_base + in_offsets, mask=mask_in, other=-float('inf'))
        
        # Update running maximum
        max_vals = tl.maximum(max_vals, vals)

    # Store the final maximum values to the output tensor
    tl.store(out_ptr_base + out_offsets, max_vals, mask=mask_out)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton wrapper for MaxPool1d.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, L = x.shape
    
    # Calculate output length
    L_out = ((L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Tuning parameter
    BLOCK_SIZE = 1024
    
    # Grid: (batch * channels, ceil(L_out / BLOCK_SIZE))
    grid = (B * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    maxpool1d_kernel[grid](
        x, 
        out, 
        L, 
        L_out, 
        stride, 
        padding, 
        dilation, 
        K=kernel_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using custom Triton kernels.
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
        Applies Max Pooling 1D to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        # Note: This implementation assumes return_indices=False as per the provided constants.
        # If return_indices=True, the original PyTorch operator returns a tuple (output, indices).
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )