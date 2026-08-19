from types import SimpleNamespace

import numpy as np

from multimodal_signals.ocular_features import (
    classify_eye_closures,
    detect_proxy_shifts,
    eye_proxy_from_landmarks,
    frame_to_frame_change,
    median_filter_valid_segments,
)


def _landmarks():
    pts = [SimpleNamespace(x=0.0) for _ in range(474)]
    pts[33].x = 0.1
    pts[133].x = 0.3
    pts[468].x = 0.25
    pts[263].x = 0.7
    pts[362].x = 0.9
    pts[473].x = 0.75
    return pts


def test_eye_proxy_uses_neutral_terminology_and_shared_sign():
    left, right, combined = eye_proxy_from_landmarks(_landmarks())
    assert left > 0
    assert right < 0
    assert np.isfinite(combined)


def test_median_filter_does_not_bridge_missing_gap():
    raw = np.array([1.0, 1.2, np.nan, 4.0, 4.2])
    valid = np.isfinite(raw)
    filtered = median_filter_valid_segments(raw, valid, width=3)
    assert np.isnan(filtered[2])


def test_frame_change_ignores_nan_gap():
    delta = frame_to_frame_change(np.array([1.0, np.nan, 4.0]))
    assert np.isnan(delta[1])
    assert np.isnan(delta[2])


def test_closure_and_proxy_shift_detection():
    events, blink_mask = classify_eye_closures(np.array([0.0, 0.6, 0.7, 0.0]), fps=10, threshold=0.5)
    assert len(events) == 1
    proxy = np.array([0.0, 0.01, 0.02, 0.7, 0.72, 0.73])
    shifts = detect_proxy_shifts(proxy, np.ones_like(proxy, dtype=bool), blink_mask=np.zeros_like(proxy, dtype=bool), fps=10)
    assert len(shifts) >= 1
