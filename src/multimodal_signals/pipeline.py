"""High-level single-recording and session pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .acoustic_features import f0_from_signal, f0_statistics, intensity_statistics
from .audio_io import decode_audio, inspect_audio_stream
from .audio_qc import audio_qc
from .config import AnalysisConfig
from .ocular_features import analyze_video_file
from .speech_activity import activity_features
from .synchronization import session_sync_qc
from .video_io import frame_timestamp_report, inspect_video_stream


def inspect_media(path: str | Path) -> dict[str, Any]:
    """Inspect available audio/video streams without requiring both modalities."""
    info = {"path": str(path)}
    try:
        info.update(inspect_audio_stream(path))
    except Exception as exc:
        info.update(has_audio=False, audio_error=str(exc))
    try:
        video = inspect_video_stream(path)
        info.update({f"video_{k}" if k in info else k: v for k, v in video.items()})
    except Exception as exc:
        info.update(has_video=False, video_error=str(exc))
    return info


def analyze_audio(
    path: str | Path,
    config: AnalysisConfig | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Analyze one audio stream independently and return features plus activity series."""
    cfg = config or AnalysisConfig()
    x, sr = decode_audio(path, channel=cfg.audio.channel)
    if x.ndim > 1:
        raise ValueError("analyze_audio expects mono or a selected channel. Use channel='mono' or a channel index.")
    duration_s = len(x) / sr
    features: dict[str, Any] = {
        "recording_id": recording_id or Path(path).stem,
        "modality": "audio",
        "duration_s": duration_s,
        "sample_rate": sr,
    }
    features.update(audio_qc(x, sr))
    try:
        _, f0, f0_meta = f0_from_signal(
            x,
            sr,
            first_pass_floor=cfg.pitch.first_pass_floor,
            first_pass_ceiling=cfg.pitch.first_pass_ceiling,
            adaptive_bounds=cfg.pitch.adaptive_bounds,
            min_voiced_frames=cfg.pitch.min_voiced_frames,
        )
        features.update(f0_meta)
        features.update(f0_statistics(f0, min_voiced_frames=cfg.pitch.min_voiced_frames))
    except Exception as exc:
        features.update({"f0_qc": "not_available", "f0_warning": str(exc)})

    intensity, _ = intensity_statistics(
        x,
        sr,
        gate_db=cfg.intensity.gate_db,
        frame_s=cfg.audio.frame_duration,
    )
    features.update(intensity)

    activity, activity_mask = activity_features(
        x,
        sr,
        frame_s=cfg.audio.frame_duration,
        min_gap_s=cfg.audio.min_gap_duration,
        min_segment_s=cfg.audio.min_segment_duration,
        threshold_db_above_noise=cfg.audio.energy_threshold_db_above_noise,
    )
    features.update(activity)
    timeseries = pd.DataFrame(
        {
            "time_s": np.arange(len(activity_mask)) * cfg.audio.frame_duration,
            "activity": activity_mask.astype(int),
        }
    )
    return {
        "features": features,
        "timeseries": timeseries,
        "signal": x,
        "sample_rate": sr,
    }


def analyze_video(
    path: str | Path,
    config: AnalysisConfig | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Analyze video metadata/timestamps and, when configured, ocular features."""
    cfg = config or AnalysisConfig()
    features: dict[str, Any] = {
        "recording_id": recording_id or Path(path).stem,
        "modality": "video",
    }
    features.update(inspect_video_stream(path))
    features.update(frame_timestamp_report(path))
    features.update(
        analyze_video_file(
            path,
            model_path=cfg.video.model_path,
            rotation=cfg.video.rotation,
            blink_threshold=cfg.ocular.blink_threshold,
            min_blink_duration=cfg.ocular.min_blink_duration,
            prolonged_closure_duration=cfg.ocular.prolonged_closure_duration,
            median_filter_width=cfg.ocular.median_filter_width,
            shift_percentile=cfg.ocular.shift_percentile,
            shift_min_distance=cfg.ocular.shift_min_distance,
            blink_guard_seconds=cfg.ocular.blink_guard_seconds,
        )
    )
    return {"features": features, "timeseries": pd.DataFrame()}


def analyze_recording(
    path: str | Path,
    config: AnalysisConfig | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Analyze whichever supported modalities are present in a recording."""
    cfg = config or AnalysisConfig()
    inspection = inspect_media(path)
    outputs: dict[str, Any] = {"inspection": inspection}
    if inspection.get("has_audio"):
        outputs["audio"] = analyze_audio(path, cfg, recording_id=recording_id)
    if inspection.get("has_video"):
        outputs["video"] = analyze_video(path, cfg, recording_id=recording_id)
    if "audio" not in outputs and "video" not in outputs:
        raise ValueError(f"No supported audio or video stream found in {path}")
    return outputs


def analyze_session(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze arbitrary configured recordings and optional synchronized-session QC."""
    session = json.loads(Path(config_path).read_text(encoding="utf-8"))
    analysis_cfg = AnalysisConfig.from_dict(session.get("analysis", {}))
    recordings = session.get("recordings", [])
    if not recordings:
        raise ValueError("Session configuration contains no recordings")

    rows: list[dict[str, Any]] = []
    decoded: dict[str, tuple[np.ndarray, int]] = {}
    for rec in recordings:
        if "path" not in rec:
            raise ValueError("Each recording entry must contain a path")
        rec_id = rec.get("id") or Path(rec["path"]).stem
        result = analyze_recording(rec["path"], analysis_cfg, recording_id=rec_id)
        if "audio" in result:
            rows.append(result["audio"]["features"])
            decoded[rec_id] = (result["audio"]["signal"], result["audio"]["sample_rate"])
        if "video" in result:
            rows.append(result["video"]["features"])

    sync_rows: list[dict[str, Any]] = []
    if session.get("synchronized", False) and len(decoded) >= 2:
        sync_rows = session_sync_qc(decoded)

    out = {
        "recording_summary": pd.DataFrame(rows),
        "sync_qc": pd.DataFrame(sync_rows),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        out["recording_summary"].to_csv(output / "recording_summary.csv", index=False)
        if len(out["sync_qc"]):
            out["sync_qc"].to_csv(output / "sync_qc.csv", index=False)
    return out
