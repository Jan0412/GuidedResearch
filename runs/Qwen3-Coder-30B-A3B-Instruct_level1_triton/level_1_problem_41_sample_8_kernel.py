import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    batch_size,
    features,
    input_len,
    output_len,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and feature index for this program
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and feature
    input_base = batch_idx * features * input_len + feature_idx * input_len
    output_base = batch_idx * features * output_len + feature_idx * output_len
    
    # Process each output element
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < output_len
    
    # For each output position, compute max over the kernel window
    for i in range(tl.cdiv(kernel_size, BLOCK_SIZE)):
        # Calculate starting position in input for current kernel window
        start_pos = output_idx * stride - padding
        # Calculate actual positions in input considering dilation
        actual_pos = start_pos + tl.arange(0, BLOCK_SIZE) * dilation
        
        # Load input values with proper masking
        input_vals = tl.load(input_ptr + input_base + actual_pos, mask=(actual_pos >= 0) & (actual_pos < input_len), other=-float('inf'))
        
        # Compute max for this window
        if i == 0:
            max_val = input_vals
            if return_indices:
                max_idx = actual_pos
        else:
            # Update max with new values
            new_max_mask = input_vals > max_val
            max_val = tl.where(new_max_mask, input_vals, max_val)
            if return_indices:
                max_idx = tl.where(new_max_mask, actual_pos, max_idx)
        
        # Break early if all positions are processed
        if (i + 1) * BLOCK_SIZE >= kernel_size:
            break
    
    # Store results
    tl.store(output_ptr + output_base + output_idx, max_val, mask=mask)
    if return_indices:
        tl.store(indices_ptr + output_base + output_idx, max_idx, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Max Pooling 1D using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        batch_size, features, seq_len = x.shape
        
        # Calculate output length
        output_len = (seq_len + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Ensure tensors are on GPU and contiguous
        x = x.contiguous().cuda()
        
        # Initialize output tensor
        output = torch.empty(batch_size, features, output_len, dtype=torch.float32, device=x.device)
        
        # Initialize indices tensor if needed
        indices = None
        if self.return_indices:
            indices = torch.empty(batch_size, features, output_len, dtype=torch.int64, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 32
        
        # Grid configuration
        grid = (
            batch_size,
            features,
            triton.cdiv(output_len, BLOCK_SIZE)
        )
        
        # Launch kernel
        maxpool1d_kernel[grid](
            x,
            output,
            indices,
            batch_size,
            features,
            seq_len,
            output_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output if not self.return_indices else (output, indices)