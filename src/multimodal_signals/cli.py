"""Command-line interface for multimodal_signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import AnalysisConfig
from .pipeline import analyze_audio, analyze_recording, analyze_session, inspect_media


def _write_feature_row(row: dict, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output / name, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multimodal-signals")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect audio/video stream metadata and QC-relevant properties.")
    inspect_p.add_argument("path")

    analyze_p = sub.add_parser("analyze", help="Analyze one media file independently.")
    analyze_p.add_argument("path")
    analyze_p.add_argument("--output", default="outputs")
    analyze_p.add_argument("--config")

    audio_p = sub.add_parser("audio", help="Analyze audio from one media file.")
    audio_p.add_argument("path")
    audio_p.add_argument("--output", default="outputs")
    audio_p.add_argument("--config")

    session_p = sub.add_parser("session", help="Analyze a configured multi-recording session.")
    session_p.add_argument("config")
    session_p.add_argument("--output", default="outputs")

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(pd.Series(inspect_media(args.path)).to_string())
            return 0
        if args.command == "audio":
            cfg = AnalysisConfig.from_json(args.config) if args.config else AnalysisConfig()
            result = analyze_audio(args.path, cfg)
            _write_feature_row(result["features"], Path(args.output), "acoustic_features.csv")
            result["timeseries"].to_csv(Path(args.output) / f"{Path(args.path).stem}_activity.csv", index=False)
            return 0
        if args.command == "analyze":
            cfg = AnalysisConfig.from_json(args.config) if args.config else AnalysisConfig()
            result = analyze_recording(args.path, cfg)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            rows = []
            if "audio" in result:
                rows.append(result["audio"]["features"])
                result["audio"]["timeseries"].to_csv(output / f"{Path(args.path).stem}_activity.csv", index=False)
            if "video" in result:
                rows.append(result["video"]["features"])
            pd.DataFrame(rows).to_csv(output / "recording_summary.csv", index=False)
            return 0
        if args.command == "session":
            analyze_session(args.config, args.output)
            return 0
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
