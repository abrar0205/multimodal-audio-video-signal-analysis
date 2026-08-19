"""Acoustic feature extraction with explicit QC states."""

from __future__ import annotations

from pathlib import Path
import tempfile
import warnings

import numpy as np

from .audio_io import write_wav


def f0_from_wav(
    wav_path: str | Path,
    first_pass_floor: float = 60.0,
    first_pass_ceiling: float = 500.0,
    adaptive_bounds: bool = True,
    min_voiced_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    """Track F0 with Praat/parselmouth and optional adaptive pitch bounds.

    The adaptive pass follows the De Looze and Hirst quartile idea in spirit:
    estimate broad voiced-frame quartiles, then narrow the pitch range. This is
    a heuristic to reduce octave errors, not proof that all errors are removed.
    """
    try:
        import parselmouth
    except ImportError as exc:
        raise RuntimeError("praat-parselmouth is required for F0 extraction.") from exc

    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch(pitch_floor=first_pass_floor, pitch_ceiling=first_pass_ceiling)
    floor = float(first_pass_floor)
    ceiling = float(first_pass_ceiling)
    state = "ok"
    warning = ""
    first = pitch.selected_array["frequency"]
    voiced = first[first > 0]
    if adaptive_bounds and voiced.size >= min_voiced_frames:
        q25, q75 = np.percentile(voiced, [25, 75])
        floor = max(20.0, float(0.75 * q25))
        ceiling = min(1200.0, float(1.5 * q75))
        if floor < ceiling:
            pitch = snd.to_pitch(pitch_floor=floor, pitch_ceiling=ceiling)
        else:
            state = "warning"
            warning = "Adaptive pitch bounds were invalid; used first-pass bounds."
    elif adaptive_bounds:
        state = "warning"
        warning = "Too few voiced frames for adaptive pitch bounds."
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    return pitch.xs(), f0, {"pitch_floor_hz": floor, "pitch_ceiling_hz": ceiling, "f0_state": state, "f0_warning": warning}


def f0_statistics(f0_hz: np.ndarray, min_voiced_frames: int = 5) -> dict[str, float | int | str]:
    voiced = np.asarray(f0_hz, dtype=float)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
    if voiced.size == 0:
        return {
            "f0_median_hz": np.nan,
            "f0_iqr_st": np.nan,
            "f0_sd_st": np.nan,
            "f0_p5_hz": np.nan,
            "f0_p95_hz": np.nan,
            "f0_range_st_robust": np.nan,
            "n_voiced_frames": 0,
            "f0_qc": "no_usable_f0",
        }
    if voiced.size < min_voiced_frames:
        qc = "few_voiced_frames"
        warnings.warn("Very few voiced frames; F0 summaries may be unstable.", RuntimeWarning, stacklevel=2)
    else:
        qc = "ok"
    median = float(np.median(voiced))
    semitone = 12 * np.log2(voiced / median)
    return {
        "f0_median_hz": median,
        "f0_iqr_st": float(np.percentile(semitone, 75) - np.percentile(semitone, 25)),
        "f0_sd_st": float(np.std(semitone)),
        "f0_p5_hz": float(np.percentile(voiced, 5)),
        "f0_p95_hz": float(np.percentile(voiced, 95)),
        "f0_range_st_robust": float(12 * np.log2(np.percentile(voiced, 95) / np.percentile(voiced, 5))),
        "n_voiced_frames": int(voiced.size),
        "f0_qc": qc,
    }


def f0_from_signal(signal: np.ndarray, sample_rate: int, **kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_wav(tmp_path, signal, sample_rate)
        return f0_from_wav(tmp_path, **kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)


def intensity_statistics(
    signal: np.ndarray, sample_rate: int, gate_db: float = 20.0, frame_s: float = 0.05
) -> tuple[dict[str, float | int | str], np.ndarray]:
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        return {"intensity_qc": "empty_signal"}, np.array([], dtype=bool)
    frame_n = max(1, int(round(frame_s * sample_rate)))
    rms = np.sqrt(np.convolve(x**2, np.ones(frame_n) / frame_n, mode="same"))
    db = 20 * np.log10(rms + 1e-12)
    peak = float(np.percentile(db, 99))
    gate = db > peak - gate_db
    vals = db[gate]
    if vals.size == 0:
        return {"intensity_qc": "no_high_energy_frames"}, gate
    return {
        "relative_level_mean_dbfs": float(np.mean(vals)),
        "relative_level_median_dbfs": float(np.median(vals)),
        "relative_level_peak_dbfs": peak,
        "relative_level_dynamic_range_db": float(np.percentile(vals, 95) - np.percentile(vals, 5)),
        "relative_level_sd_db": float(np.std(vals)),
        "n_high_energy_samples": int(vals.size),
        "intensity_qc": "ok",
    }, gate
