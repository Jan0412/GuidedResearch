import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    L_in,
    L_out,
    kernel_size,
    stride,
    padding,
    dilation,
    stride_xb,
    stride_xc,
    stride_xl,
    stride_ob,
    stride_oc,
    stride_ol,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the batch and channel dimension
    pid_bc = tl.program_id(0)
    # Program ID for the sequence dimension
    pid_l = tl.program_id(1)

    # Calculate batch and channel indices
    b = pid_bc // C
    c = pid_bc % C

    # Calculate sequence offsets for the current block
    l_start = pid_l * BLOCK_SIZE
    l_offsets = l_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds access on the output sequence
    mask = l_offsets < L_out

    # Initialize max_val with a very small number
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

    # Iterate through the kernel window
    for k in range(kernel_size):
        # Calculate the input index for each element in the block
        # Formula: input_idx = output_idx * stride + k * dilation - padding
        in_idx = l_offsets * stride + k * dilation - padding
        
        # Mask for valid input indices (handle implicit zero/negative-inf padding)
        in_mask = (in_idx >= 0) & (in_idx < L_in)
        
        # Load values from input tensor
        # Offset = b * stride_xb + c * stride_xc + in_idx * stride_xl
        val = tl.load(
            x_ptr + b * stride_xb + c * stride_xc + in_idx * stride_xl, 
            mask=in_mask, 
            other=-float('inf')
        )
        
        # Update max value
        max_val = tl.maximum(max_val, val)

    # Store the result in the output tensor
    # Offset = b * stride_ob + c * stride_oc + l_offsets * stride_ol
    tl.store(
        out_ptr + b * stride_ob + c * stride_oc + l_offsets * stride_ol, 
        max_val, 
        mask=mask
    )

def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    # Ensure input is contiguous and on GPU
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, L_in = x.shape
    
    # Calculate output length
    # L_out = floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    L_out = ((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Strides for input and output
    stride_xb = C * L_in
    stride_xc = L_in
    stride_xl = 1
    
    stride_ob = C * L_out
    stride_oc = L_out
    stride_ol = 1
    
    # Block size for the sequence dimension
    BLOCK_SIZE = 128
    
    # Grid: (Batch * Channels, Sequence_Out / BLOCK_SIZE)
    grid = (B * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    maxpool1d_kernel[grid](
        x, out,
        B, C, L_in, L_out,
        kernel_size, stride, padding, dilation,
        stride_xb, stride_xc, stride_xl,
        stride_ob, stride_oc, stride_ol,
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
            raise NotImplementedError("Triton kernel for return_indices=True is not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using the Triton implementation.

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