from multimodal_signals.config import AnalysisConfig


def test_config_loads_example_file():
    cfg = AnalysisConfig.from_json("configs/example_single.json")
    assert cfg.audio.frame_duration == 0.05
    assert cfg.pitch.min_voiced_frames == 5
