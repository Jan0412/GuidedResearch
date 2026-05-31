import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, output_ptr, weight_ptr,
    batch_size, out_channels,
    H_in, W_in, H_out, W_out,
    in_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
):
    b = tl.program_id(0)
    o = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W_out
    w = hw % W_out

    input_ptr += b * H_in * W_in * in_channels
    output_ptr += b * H_out * W_out * out_channels + o * H_out * W_out + h * W_out + w
    weight_ptr += o * kernel_size * kernel_size * in_channels

    acc = 0.0
    for i in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h * stride + kh - padding
                w_in = w * stride + kw - padding
                if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                    inp = tl.load(input_ptr + i * H_in * W_in + h_in * W_in + w_in)
                    wgt = tl.load(weight_ptr + i * kernel_size * kernel_size + kh * kernel_size + kw)
                    acc += inp * wgt
    tl.store(output_ptr, acc)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor):
    assert input.is_cuda and weight.is_cuda
    input = input.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, H_in, W_in = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    stride = 4
    padding = 2
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1

    output = torch.empty((batch_size, out_channels, H_out, W_out), dtype=torch.float32, device=input.device)

    grid = (batch_size, out_channels, H_out * W_out)
    conv2d_kernel[grid](
        input, output, weight,
        batch_size, out_channels,
        H_in, W_in, H_out, W_out,
        in_channels, kernel_size, stride, padding
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False)

    def forward(self, x):
        return triton_conv2d(x, self.conv1.weight)