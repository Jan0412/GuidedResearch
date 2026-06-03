import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: (B, C_in, L_in)
    w_ptr,  # Weight tensor: (C_out, C_in // groups, kernel_size)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, L_out)
    # Dimensions
    B, C_in, C_out, L_in, L_out,
    kernel_size, stride, padding, dilation, groups,
    # Meta-parameters
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel block index
    pid_l = tl.program_id(2)  # output position block index

    # Compute output channel range
    c_out_start = pid_c_out * BLOCK_SIZE_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out

    # Compute input position (L_out index)
    l_out = pid_l * BLOCK_SIZE_L
    l_in_start = l_out * stride - padding
    l_in_offsets = tl.arange(0, BLOCK_SIZE_L)
    l_in = l_in_start + l_in_offsets * dilation

    # Compute valid mask for input positions
    l_in_mask = (l_in >= 0) & (l_in < L_in)

    # Initialize accumulator for output
    out_acc = tl.zeros((BLOCK_SIZE_C_OUT, BLOCK_SIZE_L), dtype=tl.float32)

    # Loop over input channels in groups
    for c_in_group in range(0, C_in, BLOCK_SIZE_C_IN):
        # Input channel offsets
        c_in_offsets = c_in_group + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in

        # Load input data: shape (BLOCK_SIZE_C_IN, BLOCK_SIZE_L)
        # We need to handle the case where different positions have different valid masks
        x_block = tl.zeros((BLOCK_SIZE_C_IN, BLOCK_SIZE_L), dtype=tl.float32)
        for i, (c_in_off, c_in_m) in enumerate(zip(c_in_offsets, c_in_mask[:, None])):
            for j, (l_in_j, l_in_m) in enumerate(zip(l_in, l_in_mask[None, :])):
                if c_in_m and l_in_m:
                    # Compute index: B * (C_in * L_in) + c_in_off * L_in + l_in_j
                    idx = pid_b * (C_in * L_in) + c_in_off * L_in + l_in_j
                    x_block = tl.atomic_add(x_block, (i, j), tl.load(x_ptr + idx, mask=tl.where((c_in_m & l_in_m), 1.0, 0.0)))
        
        # For each group, we need to load the corresponding weight slice
        # Weight shape: (C_out, C_in // groups, kernel_size)
        # We need to map c_in_group to the correct group
        group_size = C_in // groups
        group_id = c_in_group // group_size
        
        # Weight offsets
        w_c_out_start = pid_c_out * BLOCK_SIZE_C_OUT
        w_c_in_start = c_in_group % group_size
        w_offsets_c_out = w_c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
        w_offsets_c_in = w_c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
        w_offsets_kernel = tl.arange(0, kernel_size)
        
        w_mask_c_out = w_offsets_c_out < C_out
        w_mask_c_in = w_offsets_c_in < group_size
        
        # Load weights: shape (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, kernel_size)
        # We'll compute convolution by iterating over kernel positions
        for k in range(kernel_size):
            # Compute weight index: c_out * (C_in // groups * kernel_size) + c_in * kernel_size + k
            w_indices = (
                w_offsets_c_out[:, None, None] * (group_size * kernel_size) +
                w_offsets_c_in[None, :, None] * kernel_size +
                k
            )
            w_shape = (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, 1)
            w_vals = tl.load(
                w_ptr + w_indices,
                mask=(w_mask_c_out[:, None, None] & w_mask_c_in[None, :, None]),
                other=0.0
            )
            
            # Compute convolution contribution
            # x_block shape: (BLOCK_SIZE_C_IN, BLOCK_SIZE_L)
            # w_vals shape: (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, 1)
            # Result: (BLOCK_SIZE_C_OUT, BLOCK_SIZE_L)
            x_block_expanded = x_block[None, :, :]  # (1, BLOCK_SIZE_C_IN, BLOCK_SIZE_L)
            w_vals_expanded = w_vals[:, :, 0]  # (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN)
            
            # Use tl.dot-like operation via broadcasting and multiply-add
            conv_contribution = tl.sum(w_vals_expanded[:, :, None] * x_block_expanded, axis=1)
            
            # Update accumulator
            out_acc += conv_contribution
    
    # Apply bias if present
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        out_acc += b_vals[:, None]

    # Store output
    out_offsets_c_out = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    out_offsets_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    out_mask_c_out = out_offsets_c_out < C_out
    out_mask_l = out_offsets_l < L_out
    
    out_mask = out_mask_c_out[:, None] & out_mask_l[None, :]
    
    out_ptr_offset = pid_b * (C_out * L_out) + out_offsets_c_out[:, None] * L_out + out_offsets_l[None, :]
    
    tl.store(out_ptr + out_ptr_offset, out_acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Triton-based 1D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, L_in = x.shape
    C_out, _, kernel_size = weight.shape
    
    # Compute output length
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes for parallelization
    BLOCK_SIZE_L = 64
    BLOCK_SIZE_C_OUT = min(16, C_out)
    BLOCK_SIZE_C_IN = min(64, C_in)
    
    # Grid dimensions: (batch_size, num_c_out_blocks, num_l_out_blocks)
    grid = (
        B,
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
        (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L,
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, dilation, groups,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        HAS_BIAS=bias is not None,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for convolution
        return triton_conv1d(
            x, 
            self.conv1d.weight,
            self.conv1d.bias if self.conv1d.bias is not None else None,
            stride=self.conv1d.stride[0],
            padding=self.conv1d.padding[0],
            dilation=self.conv1d.dilation[0],
            groups=self.conv1d.groups
        )