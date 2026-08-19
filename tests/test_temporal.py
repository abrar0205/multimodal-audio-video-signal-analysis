import numpy as np
import pytest

from multimodal_signals.temporal import mask_to_segments, missing_data_statistics, segment_statistics


def test_mask_to_segments_handles_empty_mask():
    assert mask_to_segments(np.zeros(5, dtype=bool), 0.1) == []


def test_segment_statistics_counts_activity_and_gaps():
    mask = np.array([False, True, True, False, False, True, True])
    stats = segment_statistics(mask, frame_s=0.5, total_s=3.5, min_gap_s=0.1, min_segment_s=0.1)
    assert stats["n_segments"] == 2
    assert stats["active_duration_s"] == 2.0
    assert stats["median_inter_segment_s"] == 1.0


def test_missing_data_statistics_keeps_longest_gap():
    valid = np.array([True, False, False, True, False])
    stats = missing_data_statistics(valid, frame_s=0.2)
    assert stats["n_missing_segments"] == 2
    assert stats["longest_missing_interval_s"] == pytest.approx(0.4)
