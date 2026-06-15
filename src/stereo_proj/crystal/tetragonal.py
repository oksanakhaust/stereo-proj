import math
import numpy as np
from .base import CrystalSystem, MAX_SUM_SQ, MAX_POINTS
from .cubic import _gcd3


class TetragonalSystem(CrystalSystem):
    """Tetragonal crystal system: a = b ≠ c.

    The pole of plane (hkl) in Cartesian space is proportional to
    [h/a, k/a, l/c], which reduces to [h, k, l/γ] where γ = c/a.
    """

    def __init__(self, c_over_a: float = 1.0) -> None:
        if c_over_a <= 0:
            raise ValueError("c/a ratio must be positive")
        self.c_over_a = float(c_over_a)

    def generate_hkl(
        self,
        max_sum_sq: int = 9,
        max_points: int = MAX_POINTS,
    ) -> list[tuple[int, int, int]]:
        """Return all primitive canonical (hkl) with h²+k²+l² ≤ max_sum_sq."""
        if max_sum_sq < 1 or max_sum_sq > MAX_SUM_SQ:
            raise ValueError(f"max_sum_sq must be between 1 and {MAX_SUM_SQ}")

        max_idx = math.isqrt(max_sum_sq)
        poles: set[tuple[int, int, int]] = set()

        for h in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for l in range(-max_idx, max_idx + 1):
                    if h == k == l == 0:
                        continue
                    if h * h + k * k + l * l > max_sum_sq:
                        continue
                    if _gcd3(h, k, l) != 1:
                        continue
                    poles.add((h, k, l))

        result = sorted(poles, key=lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2, p))

        if len(result) > max_points:
            raise ValueError(
                f"Too many poles ({len(result)} > {max_points}). "
                f"Reduce max_sum_sq."
            )
        return result

    def pole_vector(self, hkl: tuple[int, int, int]) -> np.ndarray:
        """Unit Cartesian vector for pole of (hkl) in tetragonal system."""
        h, k, l = hkl
        # With a = 1, c = c/a: n ∝ [h, k, l/γ]
        v = np.array([h, k, l / self.c_over_a], dtype=float)
        return v / float(np.linalg.norm(v))

    def angle_between(
        self,
        hkl1: tuple[int, int, int],
        hkl2: tuple[int, int, int],
    ) -> float:
        """Inter-planar angle between poles hkl1 and hkl2 (radians)."""
        n1 = self.pole_vector(hkl1)
        n2 = self.pole_vector(hkl2)
        cos_angle = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return math.acos(cos_angle)

    def multiplicity(self, hkl: tuple[int, int, int]) -> int:
        """Number of equivalent planes {hkl} in tetragonal 4/mmm (up to 16)."""
        h, k, l = hkl
        # 4-fold rotations + mirrors on (h,k), mirror on l
        hk_ops: set[tuple[int, int]] = {
            (h, k), (-k, h), (-h, -k), (k, -h),
            (h, -k), (k, h), (-h, k), (-k, -h),
        }
        equiv: set[tuple[int, int, int]] = set()
        for rh, rk in hk_ops:
            equiv.add((rh, rk, l))
            if l != 0:
                equiv.add((rh, rk, -l))
        return len(equiv)

    def d_relative(self, hkl: tuple[int, int, int]) -> float:
        """Relative d-spacing for tetragonal (a=1, c=c/a)."""
        h, k, l = hkl
        g = self.c_over_a
        return 1.0 / math.sqrt(h * h + k * k + (l / g) ** 2)
