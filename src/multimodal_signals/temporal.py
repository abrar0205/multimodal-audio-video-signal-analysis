"""Temporal segment utilities that preserve missing-data gaps."""

from __future__ import annotations

import numpy as np


def mask_to_segments(mask: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(float(s * frame_s), float((e + 1) * frame_s)) for s, e in zip(starts, ends)]


def smooth_segments(
    segments: list[tuple[float, float]], min_gap_s: float, min_segment_s: float
) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged: list[list[float]] = [[segments[0][0], segments[0][1]]]
    for start, end in segments[1:]:
        if start - merged[-1][1] < min_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_segment_s]


def segment_statistics(
    mask: np.ndarray, frame_s: float, total_s: float, min_gap_s: float = 0.15, min_segment_s: float = 0.20
) -> dict[str, float | int]:
    segments = smooth_segments(mask_to_segments(mask, frame_s), min_gap_s, min_segment_s)
    durations = np.array([end - start for start, end in segments], dtype=float)
    gaps = np.array([segments[i + 1][0] - segments[i][1] for i in range(len(segments) - 1)], dtype=float)
    active_s = float(np.sum(mask) * frame_s)
    return {
        "active_duration_s": active_s,
        "activity_ratio": active_s / total_s if total_s > 0 else np.nan,
        "n_segments": int(len(segments)),
        "median_segment_s": float(np.median(durations)) if durations.size else np.nan,
        "mean_segment_s": float(np.mean(durations)) if durations.size else np.nan,
        "longest_segment_s": float(np.max(durations)) if durations.size else np.nan,
        "median_inter_segment_s": float(np.median(gaps)) if gaps.size else np.nan,
        "long_inter_segment_ge1s": int(np.sum(gaps >= 1.0)) if gaps.size else 0,
    }


def missing_data_statistics(valid: np.ndarray, frame_s: float) -> dict[str, float | int]:
    valid = np.asarray(valid, dtype=bool)
    missing_segments = mask_to_segments(~valid, frame_s)
    longest = max((end - start for start, end in missing_segments), default=0.0)
    return {
        "valid_pct": float(100 * np.mean(valid)) if valid.size else np.nan,
        "n_missing_segments": int(len(missing_segments)),
        "longest_missing_interval_s": float(longest),
    }
