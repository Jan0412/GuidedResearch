import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_x, stride_w, stride_b, stride_out,
    batch_size, in_channels, out_channels, kernel_size, length, out_length, groups, padding, dilation, stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr, NUM_BLOCKS_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = NUM_BLOCKS_M
    num_pid_n = NUM_BLOCKS_N
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block coordinates in output space (batch, out_channels, out_length)
    # We assume batch is handled by grid dimension or looped. 
    # For simplicity and efficiency, we often batch over batch size in the grid or handle it in the kernel.
    # Here we handle batch in the grid: grid = (batch_size * num_blocks_c * num_blocks_l,)
    # But to keep grid simple, we can use 1D grid over all blocks for all batches.
    # Let's restructure grid to be (batch_size, num_blocks_c, num_blocks_l) -> flattened.
    # Actually, standard practice: grid = (batch_size * num_blocks_c * num_blocks_l, )
    # pid corresponds to a specific batch, c_block, l_block.
    
    batch_idx = pid // (num_pid_m * num_pid_n)
    c_block_idx = (pid % (num_pid_m * num_pid_n)) // num_pid_n
    l_block_idx = (pid % (num_pid_m * num_pid_n)) % num_pid_n
    
    if batch_idx >= batch_size:
        return
        
    # Output block start indices
    m_start = c_block_idx * BLOCK_SIZE_M
    n_start = l_block_idx * BLOCK_SIZE_N
    
    # Offsets for output block
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for output block
    mask_m = m_offsets < out_channels
    mask_n = n_offsets < out_length
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels * kernel_size)
    # K dimension size: in_channels * kernel_size
    # We tile K
    num_k_blocks = tl.cdiv(in_channels * kernel_size, BLOCK_SIZE_K)
    
    for k in range(num_k_blocks):
        k_start = k * BLOCK_SIZE_K
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < (in_channels * kernel_size)
        
        # Compute input channel and kernel index from k_offset
        # k_offset = c_in_idx * kernel_size + k_idx
        # c_in_idx = k_offset // kernel_size
        # k_idx = k_offset % kernel_size
        c_in_idx = k_offsets // kernel_size
        k_idx = k_offsets % kernel_size
        
        # Handle groups
        # Each output channel group maps to an input channel group
        # out_channel_group = c_out // (out_channels // groups)
        # in_channel_group = in_channels // groups
        # c_in = out_channel_group * in_channel_group + c_in_idx
        
        # We need to compute c_in for each m_offset (output channel)
        # c_out = m_offsets
        # out_channels_per_group = out_channels // groups
        # in_channels_per_group = in_channels // groups
        
        # Vectorized computation for c_in
        # c_in = ((m_offsets // out_channels_per_group) * in_channels_per_group) + c_in_idx
        
        # However, c_in_idx varies with k, and m_offsets varies with c_out.
        # This creates a mismatch in shapes if not careful.
        # c_in_idx shape: (BLOCK_SIZE_K,)
        # m_offsets shape: (BLOCK_SIZE_M,)
        # We need to broadcast or reshape.
        
        # Let's compute c_in for the current tile.
        # c_in has shape (BLOCK_SIZE_M, BLOCK_SIZE_K) after broadcasting.
        # c_out has shape (BLOCK_SIZE_M, 1)
        
        out_channels_per_group = out_channels // groups
        in_channels_per_group = in_channels // groups
        
        # c_out for this block
        c_out_block = m_offsets[:, None]  # Shape (BLOCK_SIZE_M, 1)
        
        # Group index for output channels
        out_group = c_out_block // out_channels_per_group  # Shape (BLOCK_SIZE_M, 1)
        
        # Base input channel for this group
        base_in_channel = out_group * in_channels_per_group  # Shape (BLOCK_SIZE_M, 1)
        
        # c_in_idx from k_offsets: Shape (1, BLOCK_SIZE_K)
        c_in_idx_tile = c_in_idx[None, :]  # Shape (1, BLOCK_SIZE_K)
        
        # c_in: Shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        c_in = base_in_channel + c_in_idx_tile
        
        # k_idx: Shape (1, BLOCK_SIZE_K)
        k_idx_tile = k_idx[None, :]
        
        # Compute l_in for each output position
        # l_in = n_offsets * stride + k_idx * dilation - padding
        # n_offsets: (BLOCK_SIZE_N,)
        # k_idx_tile: (1, BLOCK_SIZE_K)
        # We need l_in for each (m, n, k) -> but l_in depends only on n and k.
        # l_in shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        
        l_in = n_offsets[:, None] * stride + k_idx_tile * dilation - padding  # Shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        
        # Load input x
        # x shape: (batch, in_channels, length)
        # stride_x: (stride_x_batch, stride_x_ch, stride_x_len)
        # x_ptr offset: batch_idx * stride_x[0] + c_in * stride_x[1] + l_in * stride_x[2]
        
        # Compute pointers for x
        # We need to handle masking for l_in bounds and c_in bounds.
        # c_in is guaranteed to be within [0, in_channels) by groups logic? 
        # c_in = base + c_in_idx. base < in_channels_per_group * groups = in_channels - in_channels_per_group.
        # c_in_idx < in_channels_per_group. So c_in < in_channels. OK.
        
        # l_in mask
        mask_l = (l_in >= 0) & (l_in < length)  # Shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        
        # Combine masks
        # mask_x = mask_l & mask_k (element-wise? No, mask_k is for k_offsets, mask_l is for l_in)
        # Actually, we need to mask based on valid l_in and valid k_offsets.
        # k_offsets mask is already handled by loop range? No, we use mask_k for k_offsets.
        # But k_offsets is used to compute c_in and k_idx.
        # If k_offsets >= in_channels * kernel_size, c_in and k_idx might be invalid.
        # So we should mask based on mask_k as well.
        
        # mask_x shape: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) ? No.
        # We are loading x with shape (BLOCK_SIZE_M, BLOCK_SIZE_K) for fixed n? 
        # No, x depends on l_in which depends on n.
        # So x tile shape should be (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)?
        # That's too large for register tile.
        # Standard approach: x tile is (BLOCK_SIZE_M, BLOCK_SIZE_K) and we iterate n?
        # Or x tile is (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) but we load in chunks.
        # Better: Use tl.dot with automatic tiling or structure loops differently.
        
        # Alternative structure:
        # Outer loops: batch, c_block, l_block.
        # Inner loop: k_block.
        # Inside k_block:
        #   Load x tile: shape (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) -> too big.
        #   Load w tile: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        #   We need to align dimensions for tl.dot.
        #   tl.dot(A, B) where A is (M, K), B is (K, N).
        #   Here output is (M, N).
        #   So we need A shape (M, K) and B shape (K, N).
        #   A corresponds to x? x depends on n. So x cannot be A.
        #   B corresponds to w? w is (out_ch, in_ch * k_size). w does not depend on n.
        #   So w can be B? w shape (M, K). Yes.
        #   x needs to be A? x shape (M, N, K) -> reshape to (M, N*K)? No.
        #   This suggests we should transpose the problem or use a different tiling.
        
        # Common Triton Conv pattern:
        # Use tl.dot with A=(M, K), B=(K, N).
        # Here K is in_channels * kernel_size.
        # A = x reshaped to (M, K) for each n? No.
        # Actually, for each output element (m, n), acc += x[m, n, k] * w[m, k].
        # x[m, n, k] depends on n and k.
        # w[m, k] depends on m and k.
        # This is not a standard GEMM because x varies with n.
        # However, we can view it as: for fixed n, x[:, n, :] is a matrix of shape (M, K).
        # Then acc_n = x[:, n, :] @ w.T? No.
        # acc[m, n] = sum_k x[m, n, k] * w[m, k].
        # This is element-wise product sum over k for each m, n.
        # This is like a batch of GEMMs? No.
        # It's a reduction over k.
        # We can compute this by loading x_tile (M, K) and w_tile (M, K) and doing element-wise multiply and reduce?
        # tl.dot requires matrix multiplication structure.
        # We can use tl.dot if we reshape.
        # Let A = x_tile with shape (M, K).
        # Let B = w_tile with shape (M, K).
        # We want sum_k A[m, k] * B[m, k].
        # This is not tl.dot. tl.dot does sum_k A[m, k] * B[k, n].
        # Here B depends on m, not n.
        # So we cannot use tl.dot directly in the standard way for this formulation.
        # We must use element-wise operations and reduce.
        # Or we can transpose w to (K, M) and use tl.dot?
        # If we transpose w to w_t = w.T, shape (K, M).
        # Then w_t[k, m] = w[m, k].
        # acc[m, n] = sum_k x[m, n, k] * w_t[k, m].
        # This looks like tl.dot(x_tile, w_t) but x_tile shape (M, K) and w_t shape (K, M).
        # tl.dot(x_tile, w_t) would give shape (M, M)? No.
        # tl.dot(A, B) -> A is (M, K), B is (K, N) -> (M, N).
        # Here B is w_t which is (K, M). So N=M? No.
        # We want output (M, N).
        # So we need B to have second dimension N.
        # But w does not depend on n.
        # This implies we cannot use tl.dot for the entire tile at once if x varies with n.
        # Solution: Iterate over n inside the kernel or use a different tiling.
        # Or accept that we compute using element-wise multiply and reduce.
        # Given the constraints, using element-wise multiply and reduce is safer for correctness.
        # Performance might be lower than tl.dot but it's correct.
        # However, for speedups, we should try to use tl.dot.
        # Can we restructure?
        # Output (B, M, N).
        # Reshape x to (B, M, N, K) -> (B, M, N*K).
        # Reshape w to (M, K).
        # This doesn't help directly.
        # Another approach: Im2col.
        # x_col shape (B, M, N, K) -> (B, M, N*K).
        # w shape (M, K).
        # This is still element-wise per M.
        # Actually, standard Conv GEMM:
        # x_col shape (B, M, N*K).
        # w shape (M, K).
        # We want y = x_col @ w^T? No.
        # y[m, n] = sum_k x_col[m, n, k] * w[m, k].
        # This is not matrix multiplication between x_col and w.
        # It's a reduction over K for each M, N.
        # So we must loop or use reduction.
        # In Triton, we can do:
        # acc = tl.sum(x_tile * w_tile, axis=1) ? No, axis depends.
        # x_tile shape (M, K), w_tile shape (M, K).
        # We want sum over K.
        # acc[m, n] = sum_k x[m, n, k] * w[m, k].
        # This requires x to have dimension N.
        # So x_tile must be (M, N, K).
        # w_tile must be (M, K).
        # We can broadcast w to (M, N, K).
        # Then element-wise multiply and reduce over K.
        # This is feasible.
        # x_tile shape (M, N, K).
        # w_tile shape (M, K) -> broadcast to (M, N, K).
        # prod = x_tile * w_tile.
        # acc = tl.sum(prod, axis=2).
        # This works.
        # Memory access for x: (M, N, K).
        # We need to load x with shape (M, N, K).
        # This requires loading (M, N) positions for each K.
        # This is complex for memory coalescing.
        # Better to load x as (M, K) and iterate N?
        # Or load x as (N, K) and iterate M?
        # Given the grid structure, we have fixed M block and N block.
        # So we have M and N ranges.
        # We can load x for the current M block and N block.
        # x_tile shape (M, N, K).
        # We can load this by iterating K blocks.
        # For each K block, we load x part for (M, N, K_block).
        # x_ptr offset depends on m, n, k.
        # This is doable.
        # Let's proceed with this approach.
        # x_tile shape (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K).
        # w_tile shape (BLOCK_SIZE_M, BLOCK_SIZE_K).
        # Broadcast w to (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K).
        # Multiply and sum over K.
        
        # Compute x offsets
        # x offset = batch_idx * stride_x[0] + c_in * stride_x[1] + l_in * stride_x[2]
        # c_in shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # l_in shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # We need x offset shape (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        # c_in_tile = c_in[:, :, None] ? No.
        # c_in is (M, K). l_in is (N, K).
        # We need to combine M, N, K.
        # c_in for (m, n, k) is same for all n.
        # So c_in_tile = c_in[:, None, :] -> (M, 1, K) -> broadcast to (M, N, K).
        # l_in_tile = l_in[None, :, :] -> (1, N, K) -> broadcast to (M, N, K).
        # x_offset = batch_idx * stride_x[0] + c_in_tile * stride_x[1] + l_in_tile * stride_x[2]
        
        # Load x
        # x_ptr + x_offset
        # mask = mask_l & mask_k & mask_m (for c_in bounds? c_in is valid)
        # mask_x = mask_l[None, :, :] & mask_k[None, None, :] & mask_m[:, None, None]
        # Actually mask_m is for m offsets.
        # mask_x shape (M, N, K).
        # x_val = tl.load(x_ptr + x_offset, mask=mask_x, other=0.0)
        
        # Load w
        # w offset = c_out * stride_w[0] + k_offset * stride_w[1]
        # c_out shape (M, 1)
        # k_offset shape (1, K)
        # w_offset shape (M, K)
        # w_val = tl.load(w_ptr + w_offset, mask=mask_w, other=0.0)
        # mask_w = mask_m[:, None] & mask_k[None, :]
        # Broadcast w_val to (M, N, K)
        # w_val_tile = w_val[:, None, :]
        
        # Multiply and reduce
        # prod = x_val * w_val_tile
        # acc += tl.sum(prod, axis=2)
        
        # This seems correct and uses memory efficiently?
        # Loading x with mask is important.
        # l_in can be out of bounds.
        # mask_l handles l_in bounds.
        # mask_k handles k bounds.
        # mask_m handles m bounds.
        # This should work.
        
        # Construct offsets
        # c_in_tile shape (M, 1, K)
        c_in_tile = c_in[:, None, :]
        # l_in_tile shape (1, N, K)
        l_in_tile = l_in[None, :, :]
        
        # x offset
        x_offset = batch_idx * stride_x + c_in_tile * stride_x_ch + l_in_tile * stride_x_len
        # Note: stride_x is tuple. Need to extract.
        # In kernel args, stride_x is likely a tuple or we pass individual strides.
        # Let's assume we pass stride_x_ch and stride_x_len separately or as tuple.
        # In the wrapper, we can pass strides as tuple.
        # Here we assume stride_x, stride_w, etc. are tuples.
        # x_offset = batch_idx * stride_x[0] + c_in_tile * stride_x[1] + l_in_tile * stride_x[2]
        # But c_in_tile shape (M, 1, K), stride_x[1] is scalar.
        # So c_in_tile * stride_x[1] works.
        # Similarly for l_in_tile.
        
        # Load x
        # mask_x
        mask_x = mask_l[None, :, :] & mask_k[None, None, :] & mask_m[:, None, None]
        x_val = tl.load(x_ptr + x_offset, mask=mask_x, other=0.0)
        
        # Load w
        # w offset
        # c_out shape (M, 1)
        # k_offset shape (1, K)
        w_offset = m_offsets[:, None] * stride_w_ch + k_offsets[None, :] * stride_w_k
        mask_w = mask_m[:, None] & mask_k[None, :]
        w_val = tl.load(w_ptr + w_offset, mask=mask_w, other=0.0)
        
        # Broadcast w
        w_val_tile = w_val[:, None, :]
        
        # Accumulate
        acc += tl.sum(x_val * w_val_tile, axis=2)
    
    # Add bias
    # bias shape (M, 1)
    bias_val = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)[:, None]
    acc += bias_val
    
    # Store output
    # out offset
    # out_ptr + batch_idx * stride_out[0] + m_offsets * stride_out[1] + n_offsets * stride_out[2]
    out_offset = batch_idx * stride_out[0] + m_offsets[:, None] * stride_out[1] + n_offsets[None, :] * stride_out[2]
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask_out)

def triton_conv1d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = w.shape
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, out_length), dtype=x.dtype, device=x.device)
    
    # Strides
    stride_x = x.stride()
    stride_w = w.stride()
    stride_b = bias.stride() if bias is not None else (0,)
    stride_out = out.stride()
    
    # Block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    NUM_BLOCKS_M = triton.cdiv(out_channels, BLOCK_SIZE_M)
    NUM_BLOCKS_N = triton.cdiv(out_length, BLOCK_SIZE_N)
    
    grid = lambda meta: (batch_size * NUM_BLOCKS_M * NUM_BLOCKS_N,)
    
    conv1d_kernel[grid](
        x, w, bias, out,
        stride_x[0], stride_w[0], stride_b[0] if bias is not None else 0, stride_out[0],
        batch_size, in_channels, out_channels, kernel_size, length, out_length, groups, padding, dilation, stride,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_BLOCKS_M, NUM_BLOCKS_N
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)