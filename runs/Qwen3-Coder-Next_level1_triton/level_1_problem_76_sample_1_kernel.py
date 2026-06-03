import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor (B, C_in, L_in)
    w_ptr,  # Weight tensor (C_out, C_in, K)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, L_out)
    B, C_in, L_in, C_out, K, 
    stride, dilation,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    C_OUT_GROUP: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_c_out_group = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Compute the start and end of the output sequence this program instance handles
    start_l = pid_l * BLOCK_SIZE_L
    end_l = min(start_l + BLOCK_SIZE_L, (L_in - (dilation * (K - 1) + 1) + stride) // stride)
    
    # Compute output sequence index (actual input indices depend on stride and dilation)
    output_l = start_l + tl.arange(0, BLOCK_SIZE_L)
    input_l = output_l * stride + tl.arange(0, BLOCK_SIZE_L) * 0  # Will be broadcasted
    
    # For each output position, we need to compute: sum_{c_in, k} x[b, c_in, l*stride + k*dilation] * w[c_out, c_in, k]
    # We'll process C_out groups in parallel and accumulate over C_in and K
    
    # Initialize accumulator
    acc = tl.zeros((C_OUT_GROUP, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(C_in):
        # Load input data: shape (C_OUT_GROUP, BLOCK_SIZE_L, K) -> we'll handle C_OUT_GROUP * K per iteration
        # We'll load K values per output position
        # For each output position l_out, input position is l_out * stride + k * dilation
        
        # Create offsets for input positions: [l_out * stride + k * dilation for k in range(K)]
        k_offsets = tl.arange(0, BLOCK_SIZE_K)
        # Broadcast l offsets: (BLOCK_SIZE_L, 1) and k offsets: (1, K)
        l_idx = output_l[:, None]  # (BLOCK_SIZE_L, 1)
        k_idx = k_offsets[None, :]  # (1, BLOCK_SIZE_K)
        input_pos = l_idx * stride + k_idx * dilation  # (BLOCK_SIZE_L, BLOCK_SIZE_K)
        
        # Check if input positions are within bounds
        mask_input = input_pos < L_in  # (BLOCK_SIZE_L, BLOCK_SIZE_K)
        
        # Reshape to flatten for loading: (BLOCK_SIZE_L * BLOCK_SIZE_K,)
        input_pos_flat = input_pos.reshape([-1])
        mask_input_flat = mask_input.reshape([-1])
        
        # Compute index in x tensor: B, C_in, L_in
        # For current batch and input channel
        batch_offset = pid_batch * C_in * L_in
        channel_offset = c_in * L_in
        x_indices = batch_offset + channel_offset + input_pos_flat  # (BLOCK_SIZE_L * BLOCK_SIZE_K,)
        
        # Load x values: x[b, c_in, input_pos]
        x_vals = tl.load(x_ptr + x_indices, mask=mask_input_flat, other=0.0)
        x_vals = x_vals.reshape((BLOCK_SIZE_L, BLOCK_SIZE_K))  # (BLOCK_SIZE_L, K)
        
        # Load corresponding weight values: w[c_out_group_start + i, c_in, k] for i in [0, C_OUT_GROUP)
        c_out_group_start = pid_c_out_group * C_OUT_GROUP
        
        # Create weight indices
        # We'll load weights for C_OUT_GROUP output channels
        # w shape: (C_out, C_in, K)
        w_batch_offset = channel_offset * K  # Not used directly, but for clarity
        w_indices = tl.arange(0, C_OUT_GROUP)[:, None] * (C_in * K) + \
                    c_in * K + k_offsets[None, :]  # (C_OUT_GROUP, K)
        
        # Load weights: (C_OUT_GROUP, K)
        w_vals = tl.load(w_ptr + w_indices)
        
        # Compute outer product: x_vals (BLOCK_SIZE_L, K) * w_vals (C_OUT_GROUP, K) -> (C_OUT_GROUP, BLOCK_SIZE_L)
        # For each c_out, sum over k: x[b, c_in, pos] * w[c_out, c_in, k]
        acc += tl.dot(w_vals, x_vals, trans_a=True)  # (C_OUT_GROUP, BLOCK_SIZE_L)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out_group * C_OUT_GROUP + tl.arange(0, C_OUT_GROUP))
        acc += bias[:, None]
    
    # Store results
    out_indices = (pid_batch * C_out * ((L_in - (dilation * (K - 1) + 1) + stride) // stride) + 
                   pid_c_out_group * C_OUT_GROUP * ((L_in - (dilation * (K - 1) + 1) + stride) // stride) +
                   tl.arange(0, C_OUT_GROUP)[:, None] * ((L_in - (dilation * (K - 1) + 1) + stride) // stride) +
                   output_l[None, :])
    
    out_mask = (output_l[None, :] < end_l) & (tl.arange(0, C_OUT_GROUP)[:, None] < C_out - pid_c_out_group * C_OUT_GROUP)
    
    tl.store(out_ptr + out_indices, acc, mask=out_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 1D convolution.
    Note: This implementation assumes no padding, as the original model uses default padding=0.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    
    # Compute output length: L_out = floor((L_in - dilation * (K - 1) - 1) / stride) + 1
    # For padding=0: L_out = (L_in - dilation * (K - 1) + stride - 1) // stride
    L_out = (L_in - dilation * (K - 1) + stride - 1) // stride
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Define tuning parameters
    BLOCK_SIZE_L = 128  # Sequence length block size
    BLOCK_SIZE_K = 32   # Kernel size block size (should be >= K)
    C_OUT_GROUP = 8     # Number of output channels processed per block
    
    # Grid definition
    grid = lambda meta: (
        B,                                  # batch size
        (C_out + meta["C_OUT_GROUP"] - 1) // meta["C_OUT_GROUP"],  # number of output channel groups
        (L_out + meta["BLOCK_SIZE_L"] - 1) // meta["BLOCK_SIZE_L"],  # number of sequence blocks
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, L_in, C_out, K,
        stride, dilation,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        C_OUT_GROUP=C_OUT_GROUP
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for Conv1d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same as original model but without creating the native Conv1d layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights and bias as buffers to avoid gradient tracking during init
        # Use the same initialization as PyTorch's Conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our Triton convolution implementation
        return triton_conv1d(x, self.weight, self.bias, 
                             stride=self.stride, dilation=self.dilation)


# Import math for initialization calculations
import math