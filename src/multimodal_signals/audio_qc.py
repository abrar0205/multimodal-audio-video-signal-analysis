"""Audio quality-control metrics."""

from __future__ import annotations

import numpy as np


def level_stats(signal: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(signal, dtype=float)
    if x.ndim > 1:
        x = np.nanmean(x, axis=0)
    if x.size == 0:
        return {"dc_offset": np.nan, "peak": np.nan, "rms_dbfs": np.nan, "clipped_samples": 0}
    rms = float(np.sqrt(np.nanmean(x**2)))
    return {
        "dc_offset": float(np.nanmean(x)),
        "peak": float(np.nanmax(np.abs(x))),
        "rms_dbfs": float(20 * np.log10(rms + 1e-12)),
        "clipped_samples": int(np.sum(np.abs(x) >= 0.999)),
    }


def spectrum_report(signal: np.ndarray, sample_rate: int, nfft: int = 8192) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    if x.ndim > 1:
        x = np.nanmean(x, axis=0)
    if x.size < max(16, nfft // 4):
        return {
            "lt80_hz_pct": np.nan,
            "b80_300_hz_pct": np.nan,
            "b300_3400_hz_pct": np.nan,
            "gt8k_hz_pct": np.nan,
            "hum50_db": np.nan,
            "hum100_db": np.nan,
        }
    nfft = min(nfft, 2 ** int(np.floor(np.log2(x.size))))
    nfft = max(nfft, 16)
    window = np.hanning(nfft)
    psd = np.zeros(nfft // 2 + 1)
    count = 0
    step = nfft
    for start in range(0, max(1, x.size - nfft + 1), step):
        chunk = x[start : start + nfft]
        if chunk.size != nfft:
            continue
        psd += np.abs(np.fft.rfft(chunk * window)) ** 2
        count += 1
    if count == 0:
        chunk = np.pad(x, (0, nfft - x.size))
        psd = np.abs(np.fft.rfft(chunk * window)) ** 2
    else:
        psd /= count
    freqs = np.fft.rfftfreq(nfft, 1 / sample_rate)
    total = float(psd.sum() + 1e-12)

    def band_pct(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(100 * psd[mask].sum() / total)

    def hum_db(freq: float) -> float:
        idx = int(np.argmin(np.abs(freqs - freq)))
        return float(10 * np.log10(psd[idx] / (psd.mean() + 1e-12) + 1e-12))

    return {
        "lt80_hz_pct": band_pct(0, 80),
        "b80_300_hz_pct": band_pct(80, 300),
        "b300_3400_hz_pct": band_pct(300, min(3400, sample_rate / 2)),
        "gt8k_hz_pct": band_pct(8000, sample_rate / 2) if sample_rate > 16000 else 0.0,
        "hum50_db": hum_db(50),
        "hum100_db": hum_db(100),
    }


def audio_qc(signal: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    return {**level_stats(signal), **spectrum_report(signal, sample_rate)}
