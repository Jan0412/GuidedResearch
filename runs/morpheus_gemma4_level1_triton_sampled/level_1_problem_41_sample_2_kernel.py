import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, 
    out_ptr, 
    B, C, L, L_out, 
    K, S, P, D, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID 0 handles a specific (batch, channel) pair
    pid_0 = tl.program_id(0)
    # Program ID 1 handles a block of the output sequence
    pid_1 = tl.program_id(1)

    batch_idx = pid_0 // C
    chan_idx = pid_0 % C

    # Calculate pointers to the start of the current channel's data
    x_base = x_ptr + batch_idx * C * L + chan_idx * L
    out_base = out_ptr + batch_idx * C * L_out + chan_idx * L_out

    # Indices for the output sequence block
    out_offsets = pid_1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = out_offsets < L_out

    # Initialize max values to negative infinity
    max_vals = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)

    # Loop over the kernel window
    for k in range(K):
        # Calculate the input indices for the k-th element of the kernel
        # formula: j = out_idx * stride - padding + k * dilation
        in_offsets = out_offsets * S - P + k * D
        
        # Mask for valid input indices (handle padding)
        mask_in = (in_offsets >= 0) & (in_offsets < L) & mask_out
        
        # Load values from input; use -inf for values outside the boundary (padding)
        vals = tl.load(x_base + in_offsets, mask=mask_in, other=float("-inf"))
        
        # Update running maximum
        max_vals = tl.maximum(max_vals, vals)

    # Store the resulting maximums to the output tensor
    tl.store(out_base + out_offsets, max_vals, mask=mask_out)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure input is contiguous
    x = x.contiguous()
    B, C, L = x.shape
    
    # Calculate output sequence length
    # L_out = floor((L + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    l_out = ((L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, l_out), device=x.device, dtype=x.dtype)
    
    # Triton kernel configuration
    BLOCK_SIZE = 256 
    grid = (B * C, (l_out + BLOCK_SIZE - 1) // BLOCK_SIZE)

    maxpool1d_kernel[grid](
        x, out, 
        B, C, L, l_out, 
        kernel_size, stride, padding, dilation, 
        BLOCK_SIZE=BLOCK_SIZE
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
        
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in the Triton implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using the optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )