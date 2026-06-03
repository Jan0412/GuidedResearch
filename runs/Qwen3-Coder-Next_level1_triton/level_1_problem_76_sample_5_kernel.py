import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,              # Input tensor: (batch_size, in_channels, length)
    w_ptr,              # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,              # Bias tensor: (out_channels,) or None
    out_ptr,            # Output tensor: (batch_size, out_channels, out_length)
    batch_size, 
    in_channels,
    out_channels,
    length,             # Input length
    out_length,         # Output length
    kernel_size,
    stride,
    dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch_size dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for out_channels dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels dimension
    BLOCK_SIZE_L: tl.constexpr,  # Block size for kernel_size dimension
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Compute output position in the sequence dimension
    # We'll process one output position per program to reduce coordination complexity
    # and then do the dot product across all input channels and kernel positions
    # But let's do a more efficient approach: process multiple output positions in parallel
    
    # Actually, let's implement a simpler approach: one block per (batch, out_channel, out_pos)
    # For efficiency, we'll process multiple output positions per block
    
    # Offset for this batch
    batch_offset = pid_batch * in_channels * length
    # Offset for this output channel
    out_ch_offset = pid_out_ch * kernel_size * in_channels
    
    # Compute output position range for this program
    # We'll do a simple 1D loop over output positions
    # For simplicity and to avoid complex tiling, we'll process one output position per kernel call
    # and parallelize across batch and out_ch dimensions
    
    # But to be more efficient, let's use a different strategy:
    # We'll use 3D grid: (batch, out_channel, out_pos_block)
    # However, for now, let's implement a straightforward but efficient version
    
    # Calculate the base output position for this program
    out_pos = tl.program_id(2) * BLOCK_SIZE_L
    out_pos_end = tl.minimum(out_pos + BLOCK_SIZE_L, out_length)
    
    # Pointer to start of output for this batch and output channel
    out_ptr_batch_ch = out_ptr + pid_batch * out_channels * out_length + pid_out_ch * out_length
    
    # Initialize accumulator
    if tl.program_id(2) == 0:
        acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    else:
        acc = tl.load(out_ptr_batch_ch + out_pos + tl.arange(0, BLOCK_SIZE_L), 
                      mask=tl.arange(0, BLOCK_SIZE_L) < (out_pos_end - out_pos), 
                      other=0.0).to(tl.float32)
    
    # Loop over input channels
    for in_ch in range(in_channels):
        # Pointer to this input channel
        x_ptr_ch = x_ptr + batch_offset + in_ch * length
        # Pointer to this weight (out_ch, in_ch, :)
        w_ptr_ch = w_ptr + out_ch_offset + in_ch * kernel_size
        
        # Load kernel weights
        w_offsets = tl.arange(0, kernel_size)
        w = tl.load(w_ptr_ch + w_offsets, mask=w_offsets < kernel_size, other=0.0)
        
        # Loop over output positions
        for pos_offset in range(0, BLOCK_SIZE_L, 8):  # Process in chunks to fit in registers
            pos = out_pos + pos_offset
            if pos >= out_pos_end:
                break
                
            # Compute input position for this output position
            # input_pos = pos * stride - dilation * (kernel_size - 1) + dilation * k
            # which means k = (input_pos - pos * stride + dilation * (kernel_size - 1)) / dilation
            
            # Instead, compute input positions for all kernel positions
            k_offsets = tl.arange(0, kernel_size)
            input_pos = pos * stride - dilation * (kernel_size - 1) + dilation * k_offsets
            
            # Check if input positions are valid
            mask_input = (input_pos >= 0) & (input_pos < length)
            
            # Load input values
            x_vals = tl.load(x_ptr_ch + input_pos, mask=mask_input, other=0.0)
            
            # Compute dot product
            acc_update = tl.sum(w * x_vals, axis=0)
            
            # Accumulate
            if pos_offset == 0:
                acc = acc_update
            else:
                acc = acc + acc_update
    
    # Apply bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_ch)
        acc = acc + bias
    
    # Store result
    out_offsets = tl.arange(0, BLOCK_SIZE_L) + out_pos
    mask = out_offsets < out_length
    tl.store(out_ptr_batch_ch + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution
        dilation: Dilation for convolution
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Configure grid
    # We'll use a 3D grid: (batch_size, out_channels, out_length_blocks)
    BLOCK_SIZE_L = 16  # Number of output positions per block
    
    grid = lambda meta: (
        batch_size,
        out_channels,
        triton.cdiv(out_length, meta['BLOCK_SIZE_L'])
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        length, out_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=1,  # Not used in current implementation
        BLOCK_SIZE_N=1,  # Not used in current implementation
        BLOCK_SIZE_K=1,  # Not used in current implementation
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as original Model
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, dilation=dilation, bias=bias)
        # But we'll replace the forward pass with our Triton kernel
        # So we don't need to keep the original conv1d for computation, 
        # but we keep it for parameter initialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Copy parameters from the original conv1d if it exists
        if hasattr(self, 'conv1d'):
            self.weight = self.conv1d.weight
            self.bias = self.conv1d.bias if bias else None
            # Remove the original conv1d since we won't use it for computation
            del self.conv1d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using our custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use our Triton implementation
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, dilation=self.dilation)