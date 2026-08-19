"""Generate synthetic demo audio for local examples and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multimodal_signals.audio_io import write_wav


def make_synthetic_voice_like_signal(sample_rate: int = 16_000, duration_s: float = 6.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = np.zeros_like(t)
    segments = [(0.4, 1.5, 140), (2.1, 3.1, 185), (3.8, 5.4, 165)]
    for start, end, f0 in segments:
        mask = (t >= start) & (t < end)
        tt = t[mask] - start
        fm = f0 + 8 * np.sin(2 * np.pi * 0.7 * tt)
        phase = 2 * np.pi * np.cumsum(fm) / sample_rate
        envelope = 0.25 * np.sin(np.pi * np.linspace(0, 1, mask.sum())) ** 0.5
        signal[mask] = envelope * (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase))
    signal += 0.01 * rng.standard_normal(signal.size)
    return np.clip(signal, -1, 1).astype(np.float32)


def main() -> None:
    output = Path("data/demo_synthetic_voice.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    write_wav(output, make_synthetic_voice_like_signal(sample_rate), sample_rate)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
