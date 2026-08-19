"""Configuration objects for the analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class AudioConfig:
    frame_duration: float = 0.05
    min_segment_duration: float = 0.20
    min_gap_duration: float = 0.15
    channel: int | str = "mono"
    energy_threshold_db_above_noise: float = 6.0


@dataclass(slots=True)
class PitchConfig:
    first_pass_floor: float = 60.0
    first_pass_ceiling: float = 500.0
    min_voiced_frames: int = 5
    adaptive_bounds: bool = True


@dataclass(slots=True)
class IntensityConfig:
    gate_db: float = 20.0


@dataclass(slots=True)
class VideoConfig:
    rotation: str = "auto"
    expected_fps: float | None = None
    model_path: str | None = None


@dataclass(slots=True)
class OcularConfig:
    blink_threshold: float = 0.45
    min_blink_duration: float = 0.05
    prolonged_closure_duration: float = 0.50
    median_filter_width: int = 5
    shift_percentile: float = 95.0
    shift_min_distance: float = 0.15
    blink_guard_seconds: float = 0.10


@dataclass(slots=True)
class AnalysisConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ocular: OcularConfig = field(default_factory=OcularConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            pitch=PitchConfig(**data.get("pitch", {})),
            intensity=IntensityConfig(**data.get("intensity", {})),
            video=VideoConfig(**data.get("video", {})),
            ocular=OcularConfig(**data.get("ocular", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested configuration dictionary."""
        return asdict(self)
