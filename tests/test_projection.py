"""Tests for StereographicProjection: coordinates and hemisphere filtering."""
import math
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from stereo_proj.projection import StereographicProjection


R = 100.0


# ── Basic geometry ────────────────────────────────────────────────────────────

def test_center_projects_to_origin():
    proj = StereographicProjection((0, 0, 1), radius=R)
    pole = proj.project((0, 0, 1))
    assert pole is not None
    assert abs(pole.x) < 1e-9
    assert abs(pole.y) < 1e-9
    assert pole.marker == "filled"


def test_equatorial_pole_at_radius():
    """[100] is 90° from [001] → projects exactly onto the equatorial circle."""
    proj = StereographicProjection((0, 0, 1), radius=R)
    pole = proj.project((1, 0, 0))
    assert pole is not None
    r = math.hypot(pole.x, pole.y)
    assert abs(r - R) < 1e-6, f"Expected r={R}, got r={r}"


def test_back_pole_at_origin():
    """[00-1] is 180° from [001] → its antipodal [001] projects to centre."""
    proj = StereographicProjection((0, 0, 1), radius=R)
    # [00-1] has cos_rho = -1 → rho_prime = 0 → r = 0
    pole = proj.project((0, 0, -1), hemisphere="both")
    assert pole is not None
    assert abs(pole.x) < 1e-9
    assert abs(pole.y) < 1e-9
    assert pole.marker == "open"


def test_45deg_pole_radius():
    """[011] is 45° from [001]: r = R·tan(22.5°)."""
    proj = StereographicProjection((0, 0, 1), radius=R)
    pole = proj.project((0, 1, 1))  # angle from [001]: cos = 1/√2, rho = 45°
    assert pole is not None
    expected_r = R * math.tan(math.radians(22.5))
    r = math.hypot(pole.x, pole.y)
    assert abs(r - expected_r) < 1e-6


# ── Azimuthal angles ──────────────────────────────────────────────────────────

def test_100_and_010_are_90deg_apart():
    """[100] and [010] should be 90° apart on the equatorial circle."""
    proj = StereographicProjection((0, 0, 1), radius=R)
    p1 = proj.project((1, 0, 0))
    p2 = proj.project((0, 1, 0))
    assert p1 is not None and p2 is not None
    phi1 = math.atan2(p1.y, p1.x)
    phi2 = math.atan2(p2.y, p2.x)
    diff = abs(phi2 - phi1)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    assert abs(diff - math.pi / 2) < 1e-6


# ── Hemisphere filtering ──────────────────────────────────────────────────────

def test_upper_hides_back():
    proj = StereographicProjection((0, 0, 1), radius=R)
    # [1 1 -1]: dot with [0,0,1] = -1/√3 < 0 → back hemisphere
    assert proj.project((1, 1, -1), hemisphere="upper") is None


def test_lower_hides_front():
    proj = StereographicProjection((0, 0, 1), radius=R)
    # [1 0 0]: dot with [0,0,1] = 0 → front
    assert proj.project((1, 0, 0), hemisphere="lower") is None


def test_both_shows_all():
    proj = StereographicProjection((0, 0, 1), radius=R)
    assert proj.project((1, 0, 0), hemisphere="both") is not None
    assert proj.project((1, 1, -1), hemisphere="both") is not None


def test_back_pole_marker_is_open():
    proj = StereographicProjection((0, 0, 1), radius=R)
    pole = proj.project((0, 0, -1), hemisphere="both")
    assert pole is not None
    assert pole.marker == "open"


# ── Invalid centre ────────────────────────────────────────────────────────────

def test_zero_center_raises():
    with pytest.raises(ValueError, match="0, 0, 0"):
        StereographicProjection((0, 0, 0))


# ── project_all ───────────────────────────────────────────────────────────────

def test_project_all_count():
    from stereo_proj.crystal.cubic import CubicSystem
    proj = StereographicProjection((0, 0, 1), radius=R)
    poles = proj.project_all(CubicSystem().generate_hkl(3), hemisphere="both")
    # 26 primitive poles (both signs) with max_sum_sq=3 should all project
    assert len(poles) == 26
