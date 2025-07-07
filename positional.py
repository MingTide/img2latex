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
            signal = torch.nn.functional.pad(signal, (prepad, postpad))

        # expand to match the shape of x
        shape = [1, channels] + [1] * num_dims
        shape[dim + 2] = length
        signal = signal.view(shape)

        x = x + signal

    return x