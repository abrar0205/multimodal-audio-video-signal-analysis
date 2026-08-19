"""Video inspection and frame timestamp quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def inspect_video_stream(path: str | Path) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for video inspection. Install requirements.txt.") from exc

    info: dict[str, Any] = {"path": str(path), "has_video": False}
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not streams:
            return info
        stream = streams[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        duration_s = float(stream.duration * stream.time_base) if stream.duration else None
        info.update(
            has_video=True,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            frame_count=stream.frames or None,
            duration_s=duration_s,
            codec=stream.codec_context.codec.name if stream.codec_context.codec else None,
        )
    return info


def frame_timestamp_report(path: str | Path) -> dict[str, float | int | str]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for timestamp QC. Install requirements.txt.") from exc

    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not streams:
            return {"video_qc": "no_video"}
        stream = streams[0]
        nominal_dt = 1.0 / float(stream.average_rate) if stream.average_rate else np.nan
        pts: list[float] = []
        failed = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                failed += 1
                continue
            pts.append(float(frame.pts * stream.time_base))
    if len(pts) < 2:
        return {"frames_decoded": len(pts), "failed_timestamp_frames": failed, "video_qc": "too_few_frames"}
    dt = np.diff(np.asarray(pts))
    return {
        "frames_decoded": int(len(pts)),
        "failed_timestamp_frames": int(failed),
        "nominal_dt_ms": float(1000 * nominal_dt),
        "dt_mean_ms": float(1000 * np.mean(dt)),
        "dt_max_ms": float(1000 * np.max(dt)),
        "gaps_gt_1p5x_nominal": int(np.sum(dt > 1.5 * nominal_dt)) if np.isfinite(nominal_dt) else 0,
        "gaps_lt_0p5x_nominal": int(np.sum(dt < 0.5 * nominal_dt)) if np.isfinite(nominal_dt) else 0,
        "video_qc": "ok",
    }
