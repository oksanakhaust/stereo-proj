"""Tests for CubicSystem: pole generation and angle calculations."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from stereo_proj.crystal.cubic import CubicSystem
from stereo_proj.crystal.base import MAX_SUM_SQ


# ── generate_hkl ──────────────────────────────────────────────────────────────

def test_basic_poles_present():
    system = CubicSystem()
    poles = system.generate_hkl(max_sum_sq=3)
    for expected in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]:
        assert expected in poles, f"{expected} missing from max_sum_sq=3 set"


def test_no_zero_pole():
    poles = CubicSystem().generate_hkl(max_sum_sq=3)
    assert (0, 0, 0) not in poles


def test_no_antiparallel_duplicates():
    poles = CubicSystem().generate_hkl(max_sum_sq=9)
    for h, k, l in poles:
        assert (-h, -k, -l) not in poles or (h, k, l) == (-h, -k, -l), \
            f"Antiparallel pair present: {(h,k,l)} and {(-h,-k,-l)}"


def test_canonical_form():
    """First non-zero index must be positive."""
    poles = CubicSystem().generate_hkl(max_sum_sq=9)
    for h, k, l in poles:
        for x in (h, k, l):
            if x != 0:
                assert x > 0, f"Non-canonical pole {(h,k,l)}: first non-zero is negative"
                break


def test_no_non_primitive():
    """gcd(|h|,|k|,|l|) must equal 1 for every pole."""
    import math
    poles = CubicSystem().generate_hkl(max_sum_sq=25)
    for h, k, l in poles:
        g = math.gcd(math.gcd(abs(h), abs(k)), abs(l))
        assert g == 1, f"Non-primitive pole: {(h,k,l)} has gcd={g}"
        # Also ensures (2,0,0), (2,2,0) etc. are absent
    assert (2, 0, 0) not in poles
    assert (2, 2, 0) not in poles


def test_count_n1():
    """max_sum_sq=1 gives exactly 3 poles: {100} family."""
    poles = CubicSystem().generate_hkl(max_sum_sq=1)
    assert len(poles) == 3
    assert set(poles) == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_count_n3():
    """max_sum_sq=3: {100}+{110}+{111} = 3+6+4 = 13."""
    poles = CubicSystem().generate_hkl(max_sum_sq=3)
    assert len(poles) == 13


def test_max_sum_sq_too_large():
    with pytest.raises(ValueError, match="max_sum_sq"):
        CubicSystem().generate_hkl(max_sum_sq=MAX_SUM_SQ + 1)


# ── angle_between ─────────────────────────────────────────────────────────────

def test_angle_100_010():
    a = CubicSystem().angle_between((1, 0, 0), (0, 1, 0))
    assert abs(a - math.pi / 2) < 1e-10, "100 ∧ 010 should be 90°"


def test_angle_100_100():
    a = CubicSystem().angle_between((1, 0, 0), (1, 0, 0))
    assert abs(a) < 1e-10, "100 ∧ 100 should be 0°"


def test_angle_111_100():
    expected = math.acos(1.0 / math.sqrt(3))  # ≈ 54.74°
    a = CubicSystem().angle_between((1, 1, 1), (1, 0, 0))
    assert abs(a - expected) < 1e-9


def test_angle_110_001():
    expected = math.pi / 2  # 90°
    a = CubicSystem().angle_between((1, 1, 0), (0, 0, 1))
    assert abs(a - expected) < 1e-10


def test_angle_zero_vector():
    with pytest.raises(ValueError):
        CubicSystem().angle_between((0, 0, 0), (1, 0, 0))
