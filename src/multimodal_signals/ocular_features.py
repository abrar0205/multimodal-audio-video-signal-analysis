"""Ocular and face-landmark utilities with cautious terminology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temporal import missing_data_statistics

EYE_LANDMARKS = ((33, 133, 468), (263, 362, 473))


def eye_proxy_from_landmarks(
    landmarks: object,
    eyes: tuple[tuple[int, int, int], ...] = EYE_LANDMARKS,
) -> tuple[float, float, float]:
    """Return normalized horizontal iris-offset proxies for left, right, combined.

    This is an eye-in-head proxy. It is not calibrated gaze, line of sight, or
    a partner/object target estimate.
    """
    values: list[float] = []
    for outer, inner, iris in eyes:
        center_x = 0.5 * (landmarks[outer].x + landmarks[inner].x)
        width = abs(landmarks[outer].x - landmarks[inner].x)
        values.append(float((landmarks[iris].x - center_x) / width) if width > 1e-6 else np.nan)
    combined = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
    return values[0], values[1], combined


def classify_eye_closures(
    blink_score: np.ndarray,
    fps: float,
    threshold: float = 0.45,
    min_duration_s: float = 0.05,
    prolonged_s: float = 0.50,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Classify contiguous high blink-score intervals as blink-like or prolonged.

    These are heuristic detector outputs based on model scores, not clinically
    validated eyelid-event labels.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
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
    """Median-filter only contiguous valid runs; never interpolate across gaps."""
    from scipy.ndimage import median_filter

    if width < 1:
        raise ValueError("width must be >= 1")
    raw = np.asarray(raw, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    out = np.full(raw.shape, np.nan)
    padded = np.r_[False, valid, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    for start, end in zip(starts, ends):
        segment = raw[start : end + 1]
        out[start : end + 1] = (
            median_filter(segment, size=width, mode="nearest") if segment.size >= width else segment
        )
    return out


def frame_to_frame_change(signal: np.ndarray) -> np.ndarray:
    """Return first differences only for consecutive finite samples."""
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
    """Detect large rapid horizontal eye-proxy changes using an operational threshold."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    valid = np.asarray(valid, dtype=bool)
    if blink_mask is not None:
        guard = int(round(blink_guard_s * fps))
        guarded = np.asarray(blink_mask, dtype=bool).copy()
        for idx in np.where(blink_mask)[0]:
            guarded[max(0, idx - guard) : min(len(guarded), idx + guard + 1)] = True
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
    """Summarize face availability and heuristic ocular detector outputs."""
    valid_eye = np.asarray(face_present, dtype=bool) & np.isfinite(eye_proxy_x)
    missing = missing_data_statistics(valid_eye, 1.0 / fps)
    usable_s = float(np.sum(valid_eye) / fps)
    event_types = blink_events.get("event_type", pd.Series(dtype=str))
    blink_like_count = int(np.sum(event_types == "blink_like"))
    prolonged_count = int(np.sum(event_types == "prolonged_closure"))
    return {
        "face_present_pct": float(100 * np.mean(face_present)) if len(face_present) else np.nan,
        "usable_eye_proxy_pct": missing["valid_pct"],
        "usable_eye_proxy_duration_s": usable_s,
        "n_missing_segments": missing["n_missing_segments"],
        "longest_missing_interval_s": missing["longest_missing_interval_s"],
        "blink_like_event_count": blink_like_count,
        "blink_like_events_per_min": float(blink_like_count / (usable_s / 60)) if usable_s > 0 else np.nan,
        "median_eye_closure_duration_s": float(blink_events["duration_s"].median()) if len(blink_events) else np.nan,
        "prolonged_closure_count": prolonged_count,
        "proxy_shift_event_count": int(len(proxy_shift_events)),
        "proxy_shift_events_per_min": float(len(proxy_shift_events) / (usable_s / 60)) if usable_s > 0 else np.nan,
        "ocular_qc": "ok" if np.any(face_present) else "no_face_detected",
    }


def _rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    """Apply an explicit rotation; ``auto`` and ``none`` leave frames unchanged."""
    import cv2

    rotation = (rotation or "none").lower()
    if rotation in {"auto", "none"}:
        return frame
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError("rotation must be one of: auto, none, cw, ccw, 180")


def analyze_video_file(
    path: str | Path,
    model_path: str | Path | None = None,
    *,
    rotation: str = "auto",
    blink_threshold: float = 0.45,
    min_blink_duration: float = 0.05,
    prolonged_closure_duration: float = 0.50,
    median_filter_width: int = 5,
    shift_percentile: float = 95.0,
    shift_min_distance: float = 0.15,
    blink_guard_seconds: float = 0.10,
) -> dict[str, Any]:
    """Run local MediaPipe Face Landmarker extraction and summarize ocular features.

    The function estimates a normalized horizontal eye-in-head iris-offset proxy,
    blink-like model-score events, and head yaw separately. It does *not* produce
    calibrated gaze, line of sight, target identity, or validated saccades.
    """
    if model_path is None or not Path(model_path).exists():
        return {
            "ocular_qc": "model_missing",
            "ocular_warning": "Provide a local MediaPipe Face Landmarker .task model to run ocular extraction.",
        }

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("OpenCV and MediaPipe are required for ocular extraction.") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video FPS is unavailable or invalid")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video contains no decodable frames")

    face_present = np.zeros(total, dtype=bool)
    eye_proxy_left_x = np.full(total, np.nan)
    eye_proxy_right_x = np.full(total, np.nan)
    eye_proxy_x = np.full(total, np.nan)
    blink_left = np.full(total, np.nan)
    blink_right = np.full(total, np.nan)
    blink_score = np.full(total, np.nan)
    head_yaw_deg = np.full(total, np.nan)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    decoded = 0
    try:
        for idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            frame = _rotate_frame(frame, rotation)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(round(idx / fps * 1000)))
            if not result.face_landmarks:
                continue
            landmarks = result.face_landmarks[0]
            face_present[idx] = True
            left, right, combined = eye_proxy_from_landmarks(landmarks)
            eye_proxy_left_x[idx] = left
            eye_proxy_right_x[idx] = right
            eye_proxy_x[idx] = combined

            if result.face_blendshapes:
                scores = {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
                left_blink = scores.get("eyeBlinkLeft", np.nan)
                right_blink = scores.get("eyeBlinkRight", np.nan)
                blink_left[idx] = left_blink
                blink_right[idx] = right_blink
                blink_score[idx] = float(np.nanmax([left_blink, right_blink])) if np.any(
                    np.isfinite([left_blink, right_blink])
                ) else np.nan

            if result.facial_transformation_matrixes:
                transform = np.asarray(result.facial_transformation_matrixes[0]).reshape(4, 4)
                rotation_matrix = transform[:3, :3]
                head_yaw_deg[idx] = float(np.degrees(np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])))
    finally:
        cap.release()
        landmarker.close()

    blink_events, blink_mask = classify_eye_closures(
        blink_score,
        fps,
        threshold=blink_threshold,
        min_duration_s=min_blink_duration,
        prolonged_s=prolonged_closure_duration,
    )
    proxy_shifts = detect_proxy_shifts(
        eye_proxy_x,
        face_present & np.isfinite(eye_proxy_x),
        blink_mask,
        fps,
        percentile=shift_percentile,
        min_distance_s=shift_min_distance,
        median_width=median_filter_width,
        blink_guard_s=blink_guard_seconds,
    )
    summary = ocular_summary(face_present, eye_proxy_x, blink_events, proxy_shifts, fps)
    summary.update(
        {
            "frames_requested": total,
            "frames_decoded_ocular": decoded,
            "head_yaw_median_deg": float(np.nanmedian(head_yaw_deg)) if np.any(np.isfinite(head_yaw_deg)) else np.nan,
            "head_yaw_sd_deg": float(np.nanstd(head_yaw_deg)) if np.any(np.isfinite(head_yaw_deg)) else np.nan,
            "ocular_warning": "Eye-in-head proxy and blink/proxy-shift events are heuristic, uncalibrated detector outputs.",
        }
    )
    return summary
