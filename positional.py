import torch
import math
import matplotlib.pyplot as plt


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Adds sinusoidal position encoding to a tensor in NCHW format.

    Args:
        x: tensor of shape [batch, channels, d1, d2, ..., dn]
        min_timescale: minimum timescale for the frequencies
        max_timescale: maximum timescale for the frequencies

    Returns:
        A tensor with the same shape as x, with position encodings added
    """
    static_shape = x.shape
    num_dims = len(static_shape) - 2    # d1...dn
    channels = static_shape[1]          # channels in NCHW
    num_timescales = channels // (num_dims * 2)

    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, device=x.device, dtype=torch.float32) * -log_timescale_increment
    )

    # loop over positional dimensions
    for dim in range(num_dims):
        length = static_shape[dim + 2]  # skip batch and channel
        position = torch.arange(length, device=x.device, dtype=torch.float32)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

        # pad to full channel dimension
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        if prepad + postpad > 0:
            """
                创建一个4D张量 (batch_size, channels, height, width)
                x = torch.ones(1, 1, 3, 3)
                在宽度维度上每侧填充2个单位，在高度维度上每侧填充1个单位，使用默认的0填充
                result = torch.nn.functional.pad(x, pad=(2, 2,1,1), mode='constant', value=0)
            """
            signal = torch.nn.functional.pad(signal, (prepad, postpad))

        # expand to match the shape of x
        """
        矩阵 * 常数
        [1] * 4 -> [1, 1, 1, 1]
        """
        shape = [1, channels] + [1] * num_dims
        shape[dim + 2] = length
        signal = signal.view(shape)

        x = x + signal

    return x

if __name__ == '__main__':
    import torch

    # 创建一个3x4的张量
    x = torch.randn(3, 4)
    print("Original tensor:\n", x)
    # 创建一个3x4的张量
    y= x.view(4, 3)
    print("Original tensor:\n", y)
    # 将其变更为1x12的张量
    y = x.view(1, 12)
    print("\nReshaped to 1x12:\n", y)
    y = x.view(12, 1)
    print("\nReshaped to 1x12:\n", y)

    # 使用-1让PyTorch自动计算合适的维度大小
    z = x.view(-1, 6)  # 自动计算第一个维度的大小为2
    print("\nReshaped to 2x6 using -1:\n", z)

    # 如果尝试变更成元素数量不符的形状将会报错
    try:
        x.view(12, 1)  # 这将抛出一个运行时错误
    except RuntimeError as e:
        print("\nError:", e)
