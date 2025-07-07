import torch
import math
import matplotlib.pyplot as plt


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be expressed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.
    """
    static_shape = x.shape
    num_dims = len(static_shape) - 2
    channels = static_shape[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, device=x.device, dtype=torch.float32) * -log_timescale_increment
    )

    for dim in range(num_dims):
        length = static_shape[dim + 1]
        position = torch.arange(length, device=x.device, dtype=torch.float32)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

        # Pad to correct dimension
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        if prepad + postpad > 0:
            signal = torch.nn.functional.pad(signal, (prepad, postpad))

        # Expand dimensions to match x's shape
        for _ in range(1 + dim):
            signal = signal.unsqueeze(0)
        for _ in range(num_dims - 1 - dim):
            signal = signal.unsqueeze(-2)

        x = x + signal

    return x
