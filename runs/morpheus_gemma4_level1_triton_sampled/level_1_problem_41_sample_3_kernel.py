import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, out_ptr,
    B, C, L, L_out,
    stride, padding, dilation,
    S_C, S_L,
    OS_C, OS_L,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    # Map program_id(0) to the combined batch and channel dimension
    bc_idx = tl.program_id(0)
    # Map program_id(1) to the output sequence dimension in blocks
    l_start = tl.program_id(1) * BLOCK_SIZE
    l_offsets = l_start + tl.arange(0, BLOCK_SIZE)
    mask = l_offsets < L_out

    # Create offsets for the pooling window
    k_offsets = tl.arange(0, KERNEL_SIZE)
    
    # Calculate input indices for the window: (BLOCK_SIZE, KERNEL_SIZE)
    # input_indices[l, k] = l * stride - padding + k * dilation
    input_indices = (l_offsets[:, None] * stride - padding) + (k_offsets[None, :] * dilation)
    
    # Mask indices that fall outside the input sequence boundaries
    input_mask = (input_indices >= 0) & (input_indices < L)
    
    # Calculate the pointer to the start of the current channel
    # Since the tensor is (B, C, L), the stride for the combined (B*C) dimension is S_C = L
    channel_ptr = x_ptr + bc_idx * S_C
    
    # Load values from the input tensor. Out-of-bounds are treated as -inf for max pooling.
    # Pointer arithmetic: channel_ptr + indices * S_L
    vals = tl.load(channel_ptr + input_indices * S_L, mask=input_mask, other=-float('inf'))
    
    # Compute max over the kernel dimension (axis 1)
    res = tl.max(vals, axis=1)
    
    # Store the result in the output tensor
    out_channel_ptr = out_ptr + bc_idx * OS_C
    tl.store(out_channel_ptr + l_offsets * OS_L, res, mask=mask)

def triton_maxpool1d(x, kernel_size, stride, padding, dilation):
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, L = x.shape
    
    # Handle default stride
    if stride is None:
        stride = kernel_size
        
    # Calculate output length
    L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Strides for indexing
    S_C = x.stride(1)
    S_L = x.stride(2)
    OS_C = out.stride(1)
    OS_L = out.stride(2)
    
    BLOCK_SIZE = 128
    
    # Grid: (B * C, number of blocks for L_out)
    grid = (B * C, triton.cdiv(L_out, BLOCK_SIZE))
    
    maxpool1d_kernel[grid](
        x, out,
        B, C, L, L_out,
        stride, padding, dilation,
        S_C, S_L,
        OS_C, OS_L,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # return_indices is not supported by the custom Triton kernel implementation
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported by the optimized Triton kernel.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using the Triton kernel.

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