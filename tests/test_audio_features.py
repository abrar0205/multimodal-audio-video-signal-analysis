import numpy as np

from multimodal_signals.acoustic_features import f0_statistics
from multimodal_signals.audio_qc import level_stats
from multimodal_signals.speech_activity import activity_features


def test_f0_statistics_no_voiced_frames_returns_qc_state():
    stats = f0_statistics(np.array([np.nan, np.nan]))
    assert stats["f0_qc"] == "no_usable_f0"
    assert stats["n_voiced_frames"] == 0


def test_f0_statistics_reports_semitone_variability():
    stats = f0_statistics(np.array([100.0, 110.0, 120.0, np.nan, 130.0, 140.0]))
    assert stats["f0_qc"] == "ok"
    assert stats["f0_sd_st"] > 0


def test_level_stats_reports_clipping_and_dbfs():
    stats = level_stats(np.array([0.0, 1.0, -1.0, 0.5]))
    assert stats["clipped_samples"] == 2
    assert stats["rms_dbfs"] <= 0


def test_activity_features_handles_silence():
    features, mask = activity_features(np.zeros(1600), sample_rate=16000)
    assert len(mask) > 0
    assert features["n_segments"] == 0
