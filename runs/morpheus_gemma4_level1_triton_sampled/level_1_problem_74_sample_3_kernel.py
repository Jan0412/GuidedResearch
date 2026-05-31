import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, Cout, L, Lout,
    stride, padding, dilation, kernel_size,
    stride_xb, stride_xc,
    stride_wc, stride_wo,
    stride_ob, stride_oc,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Program IDs
    batch_idx = tl.program_id(0)
    oc_idx = tl.program_id(1)
    n_block_idx = tl.program_id(2)

    # Output indices for the current block
    n_offsets = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n_offsets < Lout

    # Initialize accumulator with bias if available
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc_idx)
        acc += bias_val

    # Loop over the kernel size
    for k in range(kernel_size):
        # Calculate corresponding input index i
        # Formula: n = i * stride - padding + k * dilation
        # => i = (n + padding - k * dilation) / stride
        i_scaled = n_offsets + padding - k * dilation
        
        # Check if i is a valid integer index within the input length L
        # We use floor division and check the remainder for stride alignment
        i = i_scaled // stride
        mask_i = (i_scaled % stride == 0) & (i >= 0) & (i < L) & mask_n
        
        # Vectorized dot product over input channels (Cin)
        c_in_offsets = tl.arange(0, BLOCK_SIZE_C)
        mask_c = c_in_offsets < Cin
        
        # Load x: shape (BLOCK_SIZE_C, BLOCK_SIZE_N)
        # Offset = batch_idx * (Cin * L) + c_in * L + i
        x_ptr_offset = (batch_idx * stride_xb + 
                        c_in_offsets[:, None] * stride_xc + 
                        i[None, :])
        x_vals = tl.load(x_ptr_offset, mask=(mask_c[:, None] & mask_i[None, :]), other=0.0)
        
        # Load w: shape (BLOCK_SIZE_C,)
        # Weight shape is (Cin, Cout, K)
        # Offset = c_in * (Cout * K) + oc_idx * K + k
        w_ptr_offset = (c_in_offsets * stride_wc + 
                        oc_idx * stride_wo + 
                        k)
        w_vals = tl.load(w_ptr_offset, mask=mask_c, other=0.0)
        
        # Multiply and sum over the Cin dimension
        acc += tl.sum(x_vals * w_vals[:, None], axis=0)

    # Store the final result in the output tensor
    out_offset = batch_idx * stride_ob + oc_idx * stride_oc + n_offsets
    tl.store(out_ptr + out_offset, acc, mask=mask_n)

def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # Tensor shapes
    B, Cin, L = x.shape
    Cin_w, Cout, K = weight.shape
    
    # Calculate output length
    Lout = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Ensure tensors are contiguous and on GPU
    x = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    # Bias handling
    if bias is not None:
        bias = bias.contiguous().cuda()
    else:
        bias = None

    # Strides for pointer arithmetic
    stride_xb, stride_xc = Cin * L, L
    stride_wc, stride_wo = Cout * K, K
    stride_ob, stride_oc = Cout * Lout, Lout

    # Tuning parameters
    BLOCK_SIZE_N = 256
    # BLOCK_SIZE_C must be >= Cin and a power of 2
    BLOCK_SIZE_C = triton.next_power_of_2(Cin)

    # Grid: (Batch, OutChannels, OutputLengthBlocks)
    grid = (B, Cout, (Lout + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, L, Lout,
        stride, padding, dilation, K,
        stride_xb, stride_xc,
        stride_wc, stride_wo,
        stride_ob, stride_oc,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Transposed 1D Convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.ConvTranspose1d to manage parameters
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the PyTorch layer
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias if self.conv1d_transpose.bias is not None else None
        
        # Use the Triton implementation
        return triton_conv_transpose1d(
            x, weight, bias, 
            self.stride, self.padding, self.dilation
        )