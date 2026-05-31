import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, 
    out_ptr, 
    L_in, 
    L_out, 
    stride, 
    padding, 
    dilation, 
    kernel_size, 
    BLOCK_SIZE: tl.constexpr,
):
    # pid_nc represents the (batch_size * num_features) dimension
    pid_nc = tl.program_id(0)
    # pid_l represents the block index along the output sequence length
    pid_l_block = tl.program_id(1)

    # Calculate offsets for the output sequence length
    offsets_l = pid_l_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_l = offsets_l < L_out

    # Initialize max_val with negative infinity
    max_val = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)

    # Unroll the kernel loop to find the maximum in the window
    for k in range(kernel_size):
        # Calculate input indices for this kernel element k for all l in the block
        # Formula: input_idx = output_idx * stride - padding + k * dilation
        input_offsets = offsets_l * stride - padding + k * dilation
        
        # Calculate global pointers for the input tensor (N, C, L_in)
        # pid_nc is the flattened (batch * channel) index
        global_input_offsets = pid_nc * L_in + input_offsets
        
        # Mask to ensure we don't read out of bounds of the input sequence length
        mask_in = (input_offsets >= 0) & (input_offsets < L_in) & mask_l
        
        # Load values; use -inf for padded regions
        vals = tl.load(x_ptr + global_input_offsets, mask=mask_in, other=float("-inf"))
        
        # Element-wise maximum
        max_val = tl.maximum(max_val, vals)

    # Calculate global pointers for the output tensor (N, C, L_out)
    global_out_offsets = pid_nc * L_out + offsets_l
    tl.store(out_ptr + global_out_offsets, max_val, mask=mask_l)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton wrapper for MaxPool1d.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, L_in = x.shape
    
    # Calculate output sequence length
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, num_features, L_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    # Grid: (batch * channels, ceil(L_out / BLOCK_SIZE))
    grid = (batch_size * num_features, triton.cdiv(L_out, BLOCK_SIZE))
    
    maxpool1d_kernel[grid](
        x, out, 
        L_in, L_out, 
        stride, padding, dilation, kernel_size, 
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
        # PyTorch default: stride is kernel_size if not specified
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
        # The current Triton implementation focuses on return_indices=False
        # as per the provided architecture parameters.
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )