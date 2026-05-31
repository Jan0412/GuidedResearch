import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Input tensor pointer (batch_size, num_features, sequence_length)
    out_ptr,  # Output tensor pointer (batch_size, num_features, output_sequence_length)
    # Dimensions and parameters
    batch_size, num_features, seq_len, out_seq_len,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
    NUM_FEATURES_PER_BLOCK: tl.constexpr
):
    # Batch and feature index
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    
    # Compute the starting position in the sequence for this block
    # We process NUM_FEATURES_PER_BLOCK features at once for better memory access
    
    # For each output position in the sequence
    out_seq_idx = tl.program_id(2)
    
    # Compute input sequence position corresponding to this output position
    in_seq_start = out_seq_idx * stride - padding
    # Compute effective kernel positions
    kernel_positions = tl.arange(0, BLOCK_SIZE)
    
    # Compute the kernel size we actually use (not all positions may be valid due to padding/dilation)
    # We precompute offsets for kernel positions
    offsets = in_seq_start + kernel_positions * dilation
    # Mask to handle padding: only consider valid positions within [0, seq_len)
    mask = (offsets >= 0) & (offsets < seq_len)
    
    # Load the input values for this kernel position
    # Since we're doing max pool, we want to load values and compute max
    # We'll load up to BLOCK_SIZE elements at once, but we need to be careful about masking
    
    # Initialize max value to -inf
    max_val = -tl.math.INFINITY
    
    # Iterate over the kernel window, handling padding with masking
    # Since BLOCK_SIZE may be larger than kernel_size, we need to mask appropriately
    for i in range(0, kernel_size):
        # Compute offset for this kernel element
        k_offset = in_seq_start + i * dilation
        # Compute actual index into sequence
        seq_offset = k_offset
        # Mask if this position is outside valid range
        k_mask = (seq_offset >= 0) & (seq_offset < seq_len)
        
        if k_mask:
            # Compute pointer to this position
            ptr = x_ptr + batch_id * num_features * seq_len + feature_id * seq_len + seq_offset
            val = tl.load(ptr)
            max_val = tl.maximum(max_val, val)
    
    # Store result
    out_ptr_ = out_ptr + batch_id * num_features * out_seq_len + feature_id * out_seq_len + out_seq_idx
    tl.store(out_ptr_, max_val)


def triton_maxpool1d(
    x: torch.Tensor, 
    kernel_size: int, 
    stride: int = None, 
    padding: int = 0, 
    dilation: int = 1
) -> torch.Tensor:
    """
    Triton implementation of MaxPool1d.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, seq_len = x.shape
    
    # Calculate output sequence length
    if stride is None:
        stride = kernel_size
    
    # Formula for output length: floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    out_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, num_features, out_seq_len, dtype=x.dtype, device=x.device)
    
    # Configure grid
    # Grid: (batch_size, num_features, out_seq_len)
    # But for efficiency, we can group features together
    
    BLOCK_SIZE = 64  # Max size of kernel window to process at once
    NUM_FEATURES_PER_BLOCK = 1  # We'll process one feature at a time for simplicity
    
    # Adjust grid dimensions
    grid = (batch_size, num_features, out_seq_len)
    
    # Launch kernel
    maxpool1d_kernel[grid](
        x, out,
        batch_size, num_features, seq_len, out_seq_len,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_FEATURES_PER_BLOCK=NUM_FEATURES_PER_BLOCK
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for MaxPool1d.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        # We ignore return_indices=True as Triton implementation doesn't support it yet
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )