from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedPole:
    x: float
    y: float
    marker: str          # 'filled' = front hemisphere, 'open' = back hemisphere
    hkl: tuple[int, int, int]
    custom: bool = False  # True = manually added by user


class StereographicProjection:
    """Polar stereographic projection for a given center pole [HKL].

    Coordinate frame:
      P  = normalize([H, K, L])   — projection axis (north pole)
      e1 = ref projected onto plane perp to P, then normalized
      e2 = P × e1                 — completes the right-handed frame {e1, e2, P}

    For [001] center this gives e1 = [1,0,0] (east) and e2 = [0,1,0] (north),
    matching the conventional orientation of standard pole figures.

    Projection formula (upper hemisphere, from south pole):
      r = R · tan(ρ/2),  x = r·cos(φ),  y = r·sin(φ)

    Back-hemisphere poles (ρ > π/2) are plotted as their antipodal projection
    (open circles) at r = R·tan((π−ρ)/2) in the opposite azimuthal direction.
    """

    def __init__(
        self,
        center_hkl: tuple[int, int, int],
        radius: float = 100.0,
    ) -> None:
        H, K, L = center_hkl
        P = np.array([H, K, L], dtype=float)
        norm = float(np.linalg.norm(P))
        if norm < 1e-10:
            raise ValueError("Center pole [HKL] cannot be (0, 0, 0)")

        self.center_hkl = center_hkl
        self.radius = float(radius)
        self.P = P / norm

        # Convention: [001] projected onto the equatorial plane defines North (top).
        # This matches standard crystallographic atlas orientation.
        # Fall back to [010] when P is nearly parallel to [001].
        ref = np.array([0.0, 0.0, 1.0]) if abs(self.P[2]) < 0.9 else np.array([0.0, 1.0, 0.0])

        # e2: North direction (top of projection) — projection of ref onto equatorial plane
        ref_perp = ref - float(np.dot(ref, self.P)) * self.P
        self.e2: np.ndarray = ref_perp / float(np.linalg.norm(ref_perp))

        # e1: East direction (right of projection); {e1, e2, P} is right-handed
        self.e1: np.ndarray = np.cross(self.e2, self.P)

    # ------------------------------------------------------------------
    def project(
        self,
        hkl: tuple[int, int, int],
        hemisphere: str = "both",
    ) -> ProjectedPole | None:
        """Project a single pole onto the stereonet.

        Parameters
        ----------
        hkl : Miller indices of the pole to project.
        hemisphere : 'upper' — show only front-hemisphere poles (filled circles);
                     'lower'  — show only back-hemisphere poles (open circles);
                     'both'   — show all poles.

        Returns None if the pole is filtered by the hemisphere setting.
        """
        h, k, l = hkl
        Q = np.array([h, k, l], dtype=float)
        q_norm = float(np.linalg.norm(Q))
        if q_norm < 1e-10:
            return None
        Q = Q / q_norm

        cos_rho = float(np.clip(np.dot(self.P, Q), -1.0, 1.0))
        rho = float(np.arccos(cos_rho))

        # Azimuthal angle φ in the projection plane
        Q_perp = Q - cos_rho * self.P
        perp_norm = float(np.linalg.norm(Q_perp))
        if perp_norm > 1e-10:
            Q_perp_unit = Q_perp / perp_norm
            phi = float(np.arctan2(
                float(np.dot(Q_perp_unit, self.e2)),
                float(np.dot(Q_perp_unit, self.e1)),
            ))
        else:
            phi = 0.0  # pole exactly at center or antipode

        if cos_rho >= 0.0:  # front (upper) hemisphere
            if hemisphere == "lower":
                return None
            r = self.radius * float(np.tan(rho / 2.0))
            return ProjectedPole(
                x=r * float(np.cos(phi)),
                y=r * float(np.sin(phi)),
                marker="filled",
                hkl=hkl,
            )
        else:  # back (lower) hemisphere
            if hemisphere == "upper":
                return None
            # Project the antipodal pole −Q (which is in the front hemisphere)
            # and mark as open circle.
            rho_prime = float(np.pi) - rho          # angle of −Q from P
            r = self.radius * float(np.tan(rho_prime / 2.0))
            # Azimuthal direction of −Q is phi + π
            return ProjectedPole(
                x=-r * float(np.cos(phi)),
                y=-r * float(np.sin(phi)),
                marker="open",
                hkl=hkl,
            )

    def project_all(
        self,
        hkl_list: list[tuple[int, int, int]],
        hemisphere: str = "both",
    ) -> list[ProjectedPole]:
        """Project a list of poles, omitting filtered/degenerate ones."""
        result: list[ProjectedPole] = []
        for hkl in hkl_list:
            pole = self.project(hkl, hemisphere=hemisphere)
            if pole is not None:
                result.append(pole)
        return result

    def project_custom(
        self,
        hkl_list: list[tuple[int, int, int]],
        hemisphere: str = "both",
    ) -> list[ProjectedPole]:
        """Project user-supplied poles; marks each with custom=True."""
        result: list[ProjectedPole] = []
        for hkl in hkl_list:
            pole = self.project(hkl, hemisphere=hemisphere)
            if pole is not None:
                pole.custom = True
                result.append(pole)
        return result
