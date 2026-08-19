"""Session-level synchronization and cross-channel diagnostics."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.signal import correlate, correlation_lags


def envelope(signal: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> tuple[np.ndarray, float]:
    x = np.asarray(signal, dtype=float)
    frame_n = max(1, int(round(frame_s * sample_rate)))
    n_frames = int(np.ceil(x.size / frame_n))
    env = np.zeros(n_frames, dtype=float)
    for idx in range(n_frames):
        frame = x[idx * frame_n : (idx + 1) * frame_n]
        env[idx] = np.sqrt(np.mean(frame**2)) if frame.size else 0.0
    return env - np.mean(env), 1.0 / frame_s


def xcorr_lag(a: np.ndarray, b: np.ndarray, sample_rate: float, max_lag_s: float = 0.5) -> dict[str, float]:
    n = min(len(a), len(b))
    if n < 2:
        return {"lag_s": np.nan, "peak_corr": np.nan}
    aa = np.asarray(a[:n], dtype=float) - np.mean(a[:n])
    bb = np.asarray(b[:n], dtype=float) - np.mean(b[:n])
    corr = correlate(aa, bb, mode="full", method="fft")
    lags = correlation_lags(len(aa), len(bb), mode="full")
    keep = np.abs(lags) <= int(round(max_lag_s * sample_rate))
    corr = corr[keep]
    lags = lags[keep]
    idx = int(np.argmax(np.abs(corr)))
    norm = np.sqrt(np.sum(aa**2) * np.sum(bb**2)) + 1e-12
    return {"lag_s": float(-lags[idx] / sample_rate), "peak_corr": float(corr[idx] / norm)}


def session_sync_qc(recordings: dict[str, tuple[np.ndarray, int]], max_lag_s: float = 0.5) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for (id_a, (sig_a, sr_a)), (id_b, (sig_b, sr_b)) in itertools.combinations(recordings.items(), 2):
        if sr_a != sr_b:
            rows.append({"recording_a": id_a, "recording_b": id_b, "lag_s": np.nan, "peak_corr": np.nan, "warning": "sample_rates_differ"})
            continue
        env_a, env_sr = envelope(sig_a, sr_a)
        env_b, _ = envelope(sig_b, sr_b)
        rows.append({"recording_a": id_a, "recording_b": id_b, **xcorr_lag(env_a, env_b, env_sr, max_lag_s), "warning": ""})
    return rows
