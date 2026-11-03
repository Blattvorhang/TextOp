"""Shared time budget for goal encoding, deployment and statistics."""

import math

import torch


def clamp_goal_time(time, fps, time_min=0.0, time_max=None):
    """Clamp durations; zero/expired valid goals get the minimum duration.

    Masked time remains the caller's responsibility. time_min=0 retains the
    one-frame numerical floor used by existing training encodings.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("goal time fps must be finite and positive")
    if not math.isfinite(time_min) or time_min < 0:
        raise ValueError("goal time_min must be finite and nonnegative")
    minimum = max(float(time_min), 1.0 / float(fps))
    if time_max is not None and (not math.isfinite(time_max) or time_max < minimum):
        raise ValueError("goal time_max must be finite and >= max(time_min, 1/fps)")
    time = torch.as_tensor(time)
    if not time.is_floating_point():
        time = time.float()
    if not torch.isfinite(time).all():
        raise ValueError("goal arrival time must be finite")
    return time.clamp(min=minimum, max=time_max)
