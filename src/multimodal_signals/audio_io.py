"""Audio/media inspection and decoding utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import wave

import numpy as np


def inspect_audio_stream(path: str | Path) -> dict[str, Any]:
    """Inspect an audio stream without assuming a specific container or layout."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for media inspection. Install requirements.txt.") from exc

    info: dict[str, Any] = {"path": str(path), "has_audio": False}
    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if not streams:
            return info
        stream = streams[0]
        ctx = stream.codec_context
        info.update(
            has_audio=True,
            sample_rate=ctx.sample_rate,
            channels=ctx.channels,
            format=str(ctx.format.name) if ctx.format else None,
            codec=ctx.codec.name if ctx.codec else None,
            duration_s=float(stream.duration * stream.time_base) if stream.duration else None,
        )
    return info


def decode_audio(path: str | Path, channel: int | str = "mono") -> tuple[np.ndarray, int]:
    """Decode the first audio stream as float32 in [-1, 1].

    ``channel`` may be ``"mono"`` for an average, ``"all"`` to preserve channels,
    or a zero-based channel index. Stereo data are never collapsed unless requested.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for audio decoding. Install requirements.txt.") from exc

    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if not streams:
            raise ValueError(f"No audio stream found in {path}")
        stream = streams[0]
        sample_rate = stream.codec_context.sample_rate
        for frame in container.decode(stream):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            if np.nanmax(np.abs(arr)) > 1.5:
                arr = arr / 32768.0
            chunks.append(arr)
    if not chunks or sample_rate is None:
        raise ValueError(f"Audio stream in {path} could not be decoded")
    data = np.concatenate(chunks, axis=1)
    if channel == "mono":
        return data.mean(axis=0), sample_rate
    if channel == "all":
        return data, sample_rate
    idx = int(channel)
    if idx < 0 or idx >= data.shape[0]:
        raise ValueError(f"Requested channel {idx}, but decoded audio has {data.shape[0]} channel(s)")
    return data[idx], sample_rate


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    """Write a mono float signal as 16-bit PCM WAV."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("write_wav expects a mono 1D signal")
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
