import numpy as np
import pytest

from terra.evaluation.fidelity import (
    contact_timing_statistics,
    kinematic_contact_mask,
    pelvis_aligned_mpjpe_mm,
)
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS


def test_pelvis_aligned_mpjpe_is_duration_weighted_mean_distance():
    source = np.zeros((2, 3, 3), dtype=float)
    retargeted = np.zeros_like(source)
    retargeted[:, 0, 0] = 10.0  # arbitrary world translation removed by pelvis alignment
    retargeted[:, 1, 0] = 11.0
    retargeted[0, 2, 0] = 12.0
    retargeted[1, 2, 0] = 14.0
    weights = np.array([1.0, 3.0])

    assert pelvis_aligned_mpjpe_mm(retargeted, source, weights) == pytest.approx(1500.0)
    assert pelvis_aligned_mpjpe_mm(
        retargeted,
        source,
        weights,
        include_pelvis=False,
    ) == pytest.approx(2250.0)


def test_contact_timing_f1_weights_each_frame_probe_sample():
    reference = np.array([[True, False], [True, True], [False, False]])
    candidate = np.array([[True, True], [False, True], [False, False]])
    result = contact_timing_statistics(reference, candidate, np.array([1.0, 2.0, 100.0]))

    assert result.true_positive_s == pytest.approx(3.0)
    assert result.false_positive_s == pytest.approx(1.0)
    assert result.false_negative_s == pytest.approx(2.0)
    assert result.precision == pytest.approx(0.75)
    assert result.recall == pytest.approx(0.60)
    assert result.f1 == pytest.approx(2.0 / 3.0)


def test_contact_timing_empty_masks_are_a_perfect_match():
    empty = np.zeros((3, 4), dtype=bool)
    result = contact_timing_statistics(empty, empty, np.ones(3))

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_kinematic_contact_mask_applies_minimum_duration_in_seconds():
    names = list(DEFAULT_CONTACT_JOINTS)
    long_stationary = np.zeros((15, len(names), 3), dtype=float)
    short_stationary = np.zeros((14, len(names), 3), dtype=float)

    assert np.all(kinematic_contact_mask(long_stationary, names, 100.0))
    assert not np.any(kinematic_contact_mask(short_stationary, names, 100.0))
