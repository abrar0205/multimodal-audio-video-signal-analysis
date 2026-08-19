from multimodal_signals.config import AnalysisConfig


def test_config_loads_example_file():
    cfg = AnalysisConfig.from_json("configs/example_single.json")
    assert cfg.audio.frame_duration == 0.05
    assert cfg.audio.energy_threshold_db_above_noise == 6.0
    assert cfg.pitch.min_voiced_frames == 5


def test_config_to_dict_works_with_slotted_dataclasses():
    cfg = AnalysisConfig()
    data = cfg.to_dict()
    assert data["audio"]["channel"] == "mono"
    assert data["ocular"]["blink_threshold"] == 0.45
