import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Input tensor pointer (B, C, L_in)
    out_ptr,  # Output tensor pointer (B, C, L_out)
    batch_size,  # B
    num_features,  # C
    input_length,  # L_in
    output_length,  # L_out
    kernel_size,  # K
    stride,  # S
    padding,  # P
    dilation,  # D
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (one position for one feature)
    # We can process multiple (batch, feature) pairs per block for better occupancy
    
    # Calculate which output position this program handles
    out_idx = tl.program_id(0)
    
    if out_idx >= output_length:
        return
    
    # Calculate the starting input index for this output position
    # Input position = out_idx * stride - padding
    start_idx = out_idx * stride - padding
    
    # Initialize max to -inf
    max_val = -tl.float32("inf")
    
    # Iterate through the kernel window
    for k in range(kernel_size):
        # Calculate the actual input index with dilation
        in_idx = start_idx + k * dilation
        
        # Check if within bounds
        if in_idx >= 0 and in_idx < input_length:
            # For each (batch, feature) pair, we need to load the corresponding element
            # This requires iterating through batch and feature dimensions
            # But Triton programs are 1D, so we'll use a different approach
            
            # Actually, let's restructure: each program handles (batch, feature, out_idx)
            # We'll use a 3D grid in the launcher
            pass
    
    # Since we need to handle 3D indexing properly, let's rewrite with proper indexing
    pass


@triton.jit
def maxpool1d_kernel_v2(
    x_ptr,  # Input tensor pointer (B, C, L_in)
    out_ptr,  # Output tensor pointer (B, C, L_out)
    batch_size,  # B
    num_features,  # C
    input_length,  # L_in
    output_length,  # L_out
    kernel_size,  # K
    stride,  # S
    padding,  # P
    dilation,  # D
    BLOCK_SIZE_BATCH_FEATURES: tl.constexpr,
):
    # 1D grid: each program handles one output position for one (batch, feature) pair
    program_id = tl.program_id(0)
    
    # Total number of output elements = B * C * L_out
    total_elements = batch_size * num_features * output_length
    
    if program_id >= total_elements:
        return
    
    # Decode the program_id into (batch, feature, out_idx)
    temp = program_id
    out_idx = temp % output_length
    temp = temp // output_length
    feature_idx = temp % num_features
    batch_idx = temp // num_features
    
    # Calculate the starting input index for this output position
    start_idx = out_idx * stride - padding
    
    # Initialize max to -inf
    max_val = -tl.float32("inf")
    
    # Iterate through the kernel window
    for k in range(kernel_size):
        # Calculate the actual input index with dilation
        in_idx = start_idx + k * dilation
        
        # Check if within bounds
        if in_idx >= 0 and in_idx < input_length:
            # Calculate input pointer offset
            # Input layout: (batch, feature, length) - contiguous in row-major order
            input_offset = (batch_idx * num_features * input_length + 
                          feature_idx * input_length + in_idx)
            
            # Load the value
            val = tl.load(x_ptr + input_offset)
            
            # Update max
            max_val = tl.maximum(max_val, val)
    
    # Calculate output pointer offset
    output_offset = program_id  # Since we flattened the output
    
    # Store the result
    tl.store(out_ptr + output_offset, max_val)


class TritonMaxPool1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, dilation, return_indices):
        # Ensure input is contiguous
        x = x.contiguous()
        
        batch_size, num_features, input_length = x.shape
        
        # Calculate output length: L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        # Prepare output tensor
        out = torch.empty(batch_size, num_features, output_length, dtype=x.dtype, device=x.device)
        
        # Total number of output elements
        total_elements = batch_size * num_features * output_length
        
        # Set block size
        BLOCK_SIZE_BATCH_FEATURES = 256
        
        # Calculate grid size
        grid = lambda meta: ((total_elements + meta["BLOCK_SIZE_BATCH_FEATURES"] - 1) // meta["BLOCK_SIZE_BATCH_FEATURES"],)
        
        # Launch kernel
        maxpool1d_kernel_v2[grid](
            x, out,
            batch_size, num_features, input_length, output_length,
            kernel_size, stride, padding, dilation,
            BLOCK_SIZE_BATCH_FEATURES=BLOCK_SIZE_BATCH_FEATURES
        )
        
        # Save for backward (we don't need indices for this simplified version)
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_length = input_length
        ctx.output_length = output_length
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, we'll use PyTorch's native backward for now
        # A full custom backward would require storing argmax indices
        x, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Use PyTorch's native MaxPool1d for backward (since we didn't store indices)
        # In a production implementation, we would implement the backward pass properly
        return torch.nn.functional.max_pool1d(
            x, kernel_size=kernel_size, stride=stride, padding=padding, 
            dilation=dilation, return_indices=False
        ) * grad_output, None, None, None, None, None


def triton_maxpool1d(x, kernel_size, stride, padding, dilation, return_indices):
    if return_indices:
        # For now, use native implementation if indices are requested
        return torch.nn.functional.max_pool1d(
            x, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, return_indices=True
        )
    
    return TritonMaxPool1d.apply(x, kernel_size, stride, padding, dilation, False)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using custom Triton kernels.
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
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)