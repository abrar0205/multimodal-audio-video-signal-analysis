"""Voice/activity detection helpers."""

from __future__ import annotations

import numpy as np

from .temporal import segment_statistics


def energy_activity_mask(
    signal: np.ndarray,
    sample_rate: int,
    frame_s: float = 0.05,
    threshold_db_above_noise: float = 6.0,
) -> np.ndarray:
    """Return a lightweight frame-energy activity mask.

    This is not speech ground truth or validated diarisation. It is a local,
    dependency-light activity detector suitable for demos and exploratory QC.
    """
    x = np.asarray(signal, dtype=float)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_n = max(1, int(round(frame_s * sample_rate)))
    n_frames = int(np.ceil(x.size / frame_n))
    if n_frames == 0:
        return np.array([], dtype=bool)
    rms = np.zeros(n_frames, dtype=float)
    for idx in range(n_frames):
        frame = x[idx * frame_n : (idx + 1) * frame_n]
        rms[idx] = np.sqrt(np.mean(frame * frame)) if frame.size else 0.0
    db = 20 * np.log10(rms + 1e-12)
    noise = np.percentile(db, 20) if db.size else -240.0
    return db > (noise + threshold_db_above_noise)


def activity_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_s: float = 0.05,
    min_gap_s: float = 0.15,
    min_segment_s: float = 0.20,
    threshold_db_above_noise: float = 6.0,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Summarize heuristic frame-energy activity and return its binary mask."""
    mask = energy_activity_mask(
        signal,
        sample_rate,
        frame_s=frame_s,
        threshold_db_above_noise=threshold_db_above_noise,
    )
    total_s = len(signal) / sample_rate if sample_rate else 0.0
    features = segment_statistics(
        mask,
        frame_s,
        total_s,
        min_gap_s=min_gap_s,
        min_segment_s=min_segment_s,
    )
    features["activity_detector"] = "frame_energy"
    features["activity_threshold_db_above_noise"] = float(threshold_db_above_noise)
    features["activity_warning"] = "Activity mask is heuristic and not validated speech ground truth."
    return features, mask
