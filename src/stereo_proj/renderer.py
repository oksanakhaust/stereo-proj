from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .projection import ProjectedPole, StereographicProjection


_SIGNATURE = (
    "Кафедра физического материаловедения\n"
    "Хаустович Оксана Алексеевна  БМТМ-24-4-1"
)


def _find_logo() -> str | None:
    """Find misis_logo.png using glob (works with Cyrillic paths on Windows)."""
    import glob
    for pattern in ["misis_logo.png", "*/misis_logo.png"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def _fmt_index(n: int) -> str:
    """Miller index as mathtext fragment: negative gets \\bar{}."""
    if n < 0:
        return r"\bar{" + str(-n) + "}"
    return str(n)


def _fmt_hkl(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(l)
    return f"$({inner})$"


def _fmt_hkl_bracket(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(l)
    return f"$[{inner}]$"


def _fmt_hkil(hkl: tuple[int, int, int]) -> str:
    """4-index Miller-Bravais notation for hexagonal plane (hkil), i = -(h+k)."""
    h, k, l = hkl
    i = -(h + k)
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(i) + _fmt_index(l)
    return f"$({inner})$"


def _fmt_uvtw(uvw: tuple[int, int, int]) -> str:
    """4-index Miller-Bravais direction [uvtw] from 3-index [UVW] for hexagonal."""
    import math
    U, V, W = uvw
    # [UVW] → [u,v,t,w]: u=(2U-V)/3, v=(2V-U)/3, t=-(U+V)/3, w=W (×3 for integers)
    u3, v3, t3, w3 = 2 * U - V, 2 * V - U, -(U + V), 3 * W
    g = math.gcd(math.gcd(math.gcd(abs(u3), abs(v3)), abs(t3)), abs(w3))
    if g:
        u3, v3, t3, w3 = u3 // g, v3 // g, t3 // g, w3 // g
    inner = _fmt_index(u3) + _fmt_index(v3) + _fmt_index(t3) + _fmt_index(w3)
    return f"$[{inner}]$"


class StereogramRenderer:
    """Renders a stereographic projection using matplotlib and exports to file/bytes."""

    def __init__(self, projection: StereographicProjection) -> None:
        self.projection = projection
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    def draw(
        self,
        poles: list[ProjectedPole],
        *,
        show_grid: bool = True,
        show_labels: bool = True,
        grid_step: int = 10,
        title: str | None = None,
        custom_poles: list[ProjectedPole] | None = None,
        crystal_system: str = "кубической",
        use_miller_bravais: bool = False,
    ) -> None:
        if self.fig is not None:
            plt.close(self.fig)

        _lbl_plane = _fmt_hkil if use_miller_bravais else _fmt_hkl
        _lbl_dir   = _fmt_uvtw if use_miller_bravais else _fmt_hkl_bracket

        R = self.projection.radius

        plt.rcParams["font.family"] = "Arial"

        fig_size = float(np.clip(8.0 * np.sqrt(R / 100.0), 8.0, 16.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # ── Wulff net ──────────────────────────────────────────────────
        if show_grid:
            _gc = "#cccccc"
            _lw = 0.5
            for rho_deg in range(grid_step, 90, grid_step):
                rho_rad = np.radians(rho_deg)
                r = R * np.tan(rho_rad / 2.0)
                circle = plt.Circle((0, 0), r, fill=False, color=_gc, linewidth=_lw, zorder=1)
                ax.add_patch(circle)
            for phi_deg in range(0, 180, grid_step):
                phi_rad = np.radians(phi_deg)
                dx, dy = np.cos(phi_rad), np.sin(phi_rad)
                ax.plot([-R * dx, R * dx], [-R * dy, R * dy],
                        color=_gc, linewidth=_lw, zorder=1)

        # ── Outer boundary circle ──────────────────────────────────────
        outer = plt.Circle((0, 0), R, fill=False, color="black", linewidth=1.5, zorder=2)
        ax.add_patch(outer)

        # Crosshair at centre
        ax.plot(0, 0, "+", color="black", markersize=6, markeredgewidth=0.8, zorder=3)

        # Centre label (always visible, bold)
        ax.annotate(
            _lbl_dir(self.projection.center_hkl),
            (0, 0),
            xytext=(0, -8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=10,
        )

        # ── Scale marker/font sizes with radius ────────────────────────
        _s = float(np.sqrt(100.0 / R))
        _s = max(0.35, min(2.0, _s))
        ms_auto   = round(3.5 * _s, 2)
        ms_cross  = round(4.5 * _s, 2)
        ms_custom = round(5.0 * _s, 2)
        ms_cx     = round(6.0 * _s, 2)
        fs_auto   = max(5, round(7 * _s))
        fs_custom = max(6, round(8 * _s))

        # ── Group coincident poles (tolerance = 0.8 % of R) ───────────
        tol = R * 0.008
        groups: list[list] = []
        for pole in poles:
            placed = False
            for g in groups:
                rx, ry = g[0].x, g[0].y
                if (pole.x - rx) ** 2 + (pole.y - ry) ** 2 < tol ** 2:
                    g.append(pole)
                    placed = True
                    break
            if not placed:
                groups.append([pole])

        # ── Standard poles ─────────────────────────────────────────────
        auto_labels = show_labels
        label_texts = []

        for group in groups:
            has_filled = any(p.marker == "filled" for p in group)
            has_open   = any(p.marker == "open"   for p in group)
            # Position and label from upper-hemisphere pole when present
            rep = next((p for p in group if p.marker == "filled"), group[0])
            x, y = rep.x, rep.y

            # Marker size scales with importance: lower h²+k²+l² → bigger dot
            h, k, l = rep.hkl
            sum_sq = h * h + k * k + l * l
            ms_g = ms_auto * max(0.4, 1.0 / (max(sum_sq, 1) ** 0.28))

            if has_filled and has_open:
                # Both hemispheres: open circle + dot inside (⊙)
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.9 * _s, zorder=4)
                ax.plot(x, y, "o", color="black",
                        markersize=max(1.0, ms_g * 0.35),
                        markeredgewidth=0, zorder=5)
            elif has_filled:
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markeredgewidth=0.6 * _s, zorder=4)
            else:  # open only
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.8 * _s, zorder=4)

            if auto_labels:
                # One label per group (upper pole only — avoids doubled labels)
                t = ax.text(x, y, _lbl_plane(rep.hkl),
                            fontsize=fs_auto, ha="center", va="bottom", zorder=6)
                label_texts.append(t)

        # ── Custom poles (red circles / red crosses) ───────────────────
        if custom_poles:
            for pole in custom_poles:
                if pole.marker == "filled":
                    ax.plot(pole.x, pole.y, "o",
                            color="#d62728", markersize=ms_custom,
                            markeredgewidth=0.7 * _s, zorder=6)
                else:
                    ax.plot(pole.x, pole.y, "x",
                            color="#d62728", markersize=ms_cx,
                            markeredgewidth=1.2 * _s, zorder=6)
                t = ax.text(pole.x, pole.y, _lbl_plane(pole.hkl),
                            fontsize=fs_custom, fontweight="bold", color="#d62728",
                            ha="center", va="bottom", zorder=7)
                label_texts.append(t)

        # ── Repel overlapping labels ───────────────────────────────────
        if label_texts:
            try:
                from adjustText import adjust_text
                adjust_text(
                    label_texts,
                    ax=ax,
                    expand=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                )
            except Exception:
                pass

        # ── Legend ─────────────────────────────────────────────────────
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markersize=5, label="Верхняя полусфера"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="none",
                   markersize=5, markeredgewidth=0.8, label="Нижняя полусфера"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="none",
                   markersize=5, markeredgewidth=0.8, label="Обе полусферы (⊙)"),
        ]
        if custom_poles:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                       markersize=5, label="Заданные вручную")
            )
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=8, framealpha=0.8, handlelength=1)

        # ── Title ──────────────────────────────────────────────────────
        if title is None:
            title = (
                f"Стереографическая проекция "
                f"{_lbl_dir(self.projection.center_hkl)} "
                f"для {crystal_system} сингонии"
            )
        ax.set_title(title, fontsize=12, pad=10)

        # ── Axis limits ────────────────────────────────────────────────
        margin = R * 1.18
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin * 1.52, margin)

        # ── Watermark: logo + signature (bottom-left, always) ──────────
        sig_x = -margin * 0.98
        logo_path = _find_logo()

        if logo_path:
            try:
                logo_img = plt.imread(logo_path)
                h_px, w_px = logo_img.shape[:2]
                logo_h = margin * 0.11
                logo_w = logo_h * (w_px / h_px)
                logo_ax = ax.inset_axes(
                    [-margin * 0.98, -margin * 1.46, logo_w, logo_h],
                    transform=ax.transData,
                )
                logo_ax.imshow(logo_img)
                logo_ax.axis("off")
                sig_x = -margin * 0.98 + logo_w + margin * 0.04
            except Exception:
                pass

        lines = _SIGNATURE.split("\n")
        line_h = margin * 0.048
        top_y = -margin * 1.355  # align text top with logo top (-margin*1.46 + logo_h≈0.11)
        for i, line in enumerate(lines):
            ax.text(
                sig_x, top_y - i * line_h,
                line,
                fontsize=7, va="top", ha="left", color="#555555",
                fontfamily="Arial",
            )

        self.fig = fig
        self.ax = ax

    # ------------------------------------------------------------------
    def get_figure(self) -> plt.Figure:
        if self.fig is None:
            raise RuntimeError("Call draw() before get_figure()")
        return self.fig

    def get_bytes(self, fmt: str = "png") -> io.BytesIO:
        if self.fig is None:
            raise RuntimeError("Call draw() before get_bytes()")
        buf = io.BytesIO()
        self.fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight",
                         facecolor="white")
        buf.seek(0)
        return buf

    def export(self, path: str, fmt: str = "png") -> None:
        if self.fig is None:
            raise RuntimeError("Call draw() before export()")
        self.fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight",
                         facecolor="white")
