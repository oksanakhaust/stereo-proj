from abc import ABC, abstractmethod

import numpy as np

MAX_SUM_SQ = 100
MAX_POINTS = 1000
MAX_INDEX = 10


class CrystalSystem(ABC):
    @abstractmethod
    def generate_hkl(
        self, max_sum_sq: int, max_points: int = MAX_POINTS
    ) -> list[tuple[int, int, int]]:
        """Return list of primitive canonical (h,k,l) poles."""
        ...

    @abstractmethod
    def pole_vector(self, hkl: tuple[int, int, int]) -> np.ndarray:
        """Return unit Cartesian vector for the pole of plane (hkl)."""
        ...

    @abstractmethod
    def angle_between(
        self, hkl1: tuple[int, int, int], hkl2: tuple[int, int, int]
    ) -> float:
        """Inter-planar angle in radians."""
        ...
