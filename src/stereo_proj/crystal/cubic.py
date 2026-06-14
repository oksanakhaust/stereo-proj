import math
import numpy as np
from .base import CrystalSystem, MAX_SUM_SQ, MAX_POINTS, MAX_INDEX


def _gcd3(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))


class CubicSystem(CrystalSystem):
    """Cubic crystal system: generates primitive (hkl) poles and computes angles."""

    def generate_hkl(
        self,
        max_sum_sq: int = 9,
        max_points: int = MAX_POINTS,
    ) -> list[tuple[int, int, int]]:
        """Return all primitive canonical (hkl) with h²+k²+l² ≤ max_sum_sq.

        Canonical form: antiparallel pairs are merged by requiring the first
        non-zero index to be positive.  Non-primitive directions (gcd > 1) are
        excluded because they represent the same crystallographic direction.

        Raises ValueError if constraints are violated.
        """
        if max_sum_sq < 1 or max_sum_sq > MAX_SUM_SQ:
            raise ValueError(
                f"max_sum_sq must be between 1 and {MAX_SUM_SQ}, got {max_sum_sq}"
            )

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
                        continue  # non-primitive: same direction as a smaller (hkl)

                    poles.add((h, k, l))

        result = sorted(poles, key=lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2, p))

        if len(result) > max_points:
            raise ValueError(
                f"Too many poles ({len(result)} > {max_points}). "
                f"Reduce max_sum_sq or increase max_points."
            )

        return result

    def pole_vector(self, hkl: tuple[int, int, int]) -> np.ndarray:
        h, k, l = hkl
        v = np.array([h, k, l], dtype=float)
        return v / float(np.linalg.norm(v))

    def angle_between(
        self,
        hkl1: tuple[int, int, int],
        hkl2: tuple[int, int, int],
    ) -> float:
        """Inter-planar angle between poles hkl1 and hkl2 (radians)."""
        h1, k1, l1 = hkl1
        h2, k2, l2 = hkl2
        dot = h1 * h2 + k1 * k2 + l1 * l2
        mag1 = math.sqrt(h1 ** 2 + k1 ** 2 + l1 ** 2)
        mag2 = math.sqrt(h2 ** 2 + k2 ** 2 + l2 ** 2)
        if mag1 < 1e-12 or mag2 < 1e-12:
            raise ValueError("Zero-length Miller index vector")
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_angle)
