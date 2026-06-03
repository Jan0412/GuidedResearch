import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor pointer (batch_size, in_channels, length)
    w_ptr,  # Weight tensor pointer (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias pointer (out_channels, optional, can be nullptr)
    out_ptr,  # Output tensor pointer (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size,
    stride: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_out_pos = tl.program_id(2)

    # Compute output position in time
    out_pos = pid_out_pos
    if out_pos >= out_length:
        return

    # Calculate input position corresponding to this output position
    # For dilation: kernel element i is at position i * dilation
    # For stride: output position j corresponds to input position j * stride
    start_input_pos = out_pos * stride

    # Accumulator for this output
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    # Iterate over input channels and kernel positions in blocked fashion
    for k in range(0, in_channels * kernel_size, BLOCK_SIZE_K):
        # Get input channel and kernel index
        k_offset = k % kernel_size
        ch_offset = k // kernel_size
        in_ch = ch_offset

        # Compute input pointer offset for this batch, channel, and start position
        x_batch_offset = pid_batch * (in_channels * length)
        x_ch_offset = in_ch * length
        x_pos_offset = start_input_pos + k_offset * dilation

        # Check bounds for input position
        if x_pos_offset < length:
            x_ptr_offset = x_batch_offset + x_ch_offset + x_pos_offset
            # Load input value
            x_val = tl.load(x_ptr + x_ptr_offset)
        else:
            x_val = 0.0

        # Load weight for this output channel, input channel, and kernel position
        w_offset = pid_out_ch * (in_channels * kernel_size) + in_ch * kernel_size + k_offset
        w_val = tl.load(w_ptr + w_offset)

        # Accumulate
        acc += x_val * w_val

    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_ch)
        acc += bias

    # Store result
    out_batch_offset = pid_batch * (out_channels * out_length)
    out_ch_offset = pid_out_ch * out_length
    out_ptr_offset = out_batch_offset + out_ch_offset + out_pos
    tl.store(out_ptr + out_ptr_offset, acc.to(tl.float32))


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 1D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 3, "Input must be (batch_size, in_channels, length)"
    assert weight.dim() == 3, "Weight must be (out_channels, in_channels, kernel_size)"
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Compute output length
    out_length = (length - (kernel_size - 1) * dilation - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    # Use reasonable block sizes for matrix multiplication style operation
    BLOCK_SIZE_M = 1  # We process one output position at a time per thread block
    BLOCK_SIZE_N = 1  # One output channel per block
    BLOCK_SIZE_K = 256  # Block size forin_channels * kernel_size
    
    grid = (
        batch_size,
        out_channels,
        out_length
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernels for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create weight and bias parameters to match original model's interface
        # Note: We'll initialize these with the same shape but use our custom kernel
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)