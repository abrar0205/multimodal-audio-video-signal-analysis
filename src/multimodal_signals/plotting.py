"""Small plotting helpers for demos."""

from __future__ import annotations

import numpy as np


def plot_waveform(signal: np.ndarray, sample_rate: int, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.5))
    t = np.arange(len(signal)) / sample_rate
    ax.plot(t, signal, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    return ax
