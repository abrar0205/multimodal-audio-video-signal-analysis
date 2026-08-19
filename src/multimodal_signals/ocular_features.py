"""Ocular and face-landmark utilities with cautious terminology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .temporal import mask_to_segments, missing_data_statistics

EYE_LANDMARKS = ((33, 133, 468), (263, 362, 473))


def eye_proxy_from_landmarks(landmarks: object, eyes: tuple[tuple[int, int, int], ...] = EYE_LANDMARKS) -> tuple[float, float, float]:
    """Return normalized horizontal iris-offset proxies for left, right, combined.

    This is an eye-in-head proxy. It is not calibrated gaze, line of sight, or
    a partner/object target estimate.
    """
    values: list[float] = []
    for outer, inner, iris in eyes:
        center_x = 0.5 * (landmarks[outer].x + landmarks[inner].x)
        width = abs(landmarks[outer].x - landmarks[inner].x)
        values.append(float((landmarks[iris].x - center_x) / width) if width > 1e-6 else np.nan)
    return values[0], values[1], float(np.nanmean(values))


def classify_eye_closures(
    blink_score: np.ndarray,
    fps: float,
    threshold: float = 0.45,
    min_duration_s: float = 0.05,
    prolonged_s: float = 0.50,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.isfinite(blink_score) & (blink_score >= threshold)
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    rows = []
    for start, end in zip(starts, ends):
        duration = (end - start + 1) / fps
        if duration < min_duration_s:
            continue
        rows.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "start_s": float(start / fps),
                "end_s": float((end + 1) / fps),
                "duration_s": float(duration),
                "event_type": "blink_like" if duration < prolonged_s else "prolonged_closure",
            }
        )
    return pd.DataFrame(rows), mask


def median_filter_valid_segments(raw: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    raw = np.asarray(raw, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    out = np.full(raw.shape, np.nan)
    padded = np.r_[False, valid, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    for start, end in zip(starts, ends):
        segment = raw[start : end + 1]
        out[start : end + 1] = median_filter(segment, size=width, mode="nearest") if segment.size >= width else segment
    return out


def frame_to_frame_change(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    delta = np.full(x.shape, np.nan)
    ok = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    delta[1:][ok] = x[1:][ok] - x[:-1][ok]
    return delta


def detect_proxy_shifts(
    eye_proxy_x: np.ndarray,
    valid: np.ndarray,
    blink_mask: np.ndarray | None,
    fps: float,
    percentile: float = 95.0,
    min_distance_s: float = 0.15,
    median_width: int = 5,
    blink_guard_s: float = 0.10,
) -> pd.DataFrame:
    valid = np.asarray(valid, dtype=bool)
    if blink_mask is not None:
        guard = int(round(blink_guard_s * fps))
        guarded = np.asarray(blink_mask, dtype=bool).copy()
        for idx in np.where(blink_mask)[0]:
            guarded[max(0, idx - guard) : idx + guard + 1] = True
        valid = valid & ~guarded
    filtered = median_filter_valid_segments(np.asarray(eye_proxy_x, dtype=float), valid, median_width)
    change = np.abs(frame_to_frame_change(filtered))
    finite = change[np.isfinite(change)]
    if finite.size == 0:
        return pd.DataFrame(columns=["frame", "time_s", "amplitude"])
    threshold = float(np.percentile(finite, percentile))
    possible_frames = np.where(change >= threshold)[0]
    min_distance = max(1, int(round(min_distance_s * fps)))
    selected: list[int] = []
    for frame in possible_frames:
        if not selected or frame - selected[-1] >= min_distance:
            selected.append(int(frame))
        elif change[frame] > change[selected[-1]]:
            selected[-1] = int(frame)
    return pd.DataFrame(
        [{"frame": f, "time_s": float(f / fps), "amplitude": float(change[f])} for f in selected]
    )


def ocular_summary(
    face_present: np.ndarray,
    eye_proxy_x: np.ndarray,
    blink_events: pd.DataFrame,
    proxy_shift_events: pd.DataFrame,
    fps: float,
) -> dict[str, float | int | str]:
    valid_eye = np.asarray(face_present, dtype=bool) & np.isfinite(eye_proxy_x)
    missing = missing_data_statistics(valid_eye, 1.0 / fps)
    usable_s = float(np.sum(valid_eye) / fps)
    return {
        "face_present_pct": float(100 * np.mean(face_present)) if len(face_present) else np.nan,
        "usable_eye_proxy_pct": missing["valid_pct"],
        "usable_eye_proxy_duration_s": usable_s,
        "n_missing_segments": missing["n_missing_segments"],
        "longest_missing_interval_s": missing["longest_missing_interval_s"],
        "blink_like_event_count": int(np.sum(blink_events.get("event_type", pd.Series(dtype=str)) == "blink_like")),
        "blink_like_events_per_min": float(len(blink_events) / (usable_s / 60)) if usable_s > 0 else np.nan,
        "median_eye_closure_duration_s": float(blink_events["duration_s"].median()) if len(blink_events) else np.nan,
        "prolonged_closure_count": int(np.sum(blink_events.get("event_type", pd.Series(dtype=str)) == "prolonged_closure")),
        "proxy_shift_event_count": int(len(proxy_shift_events)),
        "ocular_qc": "ok" if np.any(face_present) else "no_face_detected",
    }


def analyze_video_file(path: str | Path, model_path: str | Path | None = None, **kwargs: object) -> dict[str, object]:
    """Run MediaPipe face landmark extraction when a local model is available.

    The repository does not download model weights automatically in offline
    environments. Provide a local MediaPipe Face Landmarker task file.
    """
    if model_path is None or not Path(model_path).exists():
        return {
            "ocular_qc": "model_missing",
            "warning": "Provide a local MediaPipe Face Landmarker .task model to run ocular extraction.",
        }
    return {
        "ocular_qc": "not_implemented_in_lightweight_demo",
        "warning": "Landmark extraction hook is ready, but demos/tests use synthetic landmark arrays.",
    }
