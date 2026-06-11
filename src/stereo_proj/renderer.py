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
    ) -> None:
        if self.fig is not None:
            plt.close(self.fig)

        R = self.projection.radius

        plt.rcParams["font.family"] = "Arial"

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
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
            _fmt_hkl(self.projection.center_hkl),
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

        # ── Standard poles ─────────────────────────────────────────────
        auto_labels = show_labels
        label_texts = []

        for pole in poles:
            if pole.marker == "filled":
                ax.plot(pole.x, pole.y, "o",
                        color="black", markersize=ms_auto,
                        markeredgewidth=0.6 * _s, zorder=4)
            else:
                ax.plot(pole.x, pole.y, "x",
                        color="black", markersize=ms_cross,
                        markeredgewidth=0.9 * _s, zorder=4)

            if auto_labels:
                t = ax.text(pole.x, pole.y, _fmt_hkl(pole.hkl),
                            fontsize=fs_auto, ha="center", va="bottom", zorder=5)
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
                t = ax.text(pole.x, pole.y, _fmt_hkl(pole.hkl),
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
            Line2D([0], [0], marker="x", color="black", markersize=5,
                   markeredgewidth=0.9, label="Нижняя полусфера"),
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
                f"{_fmt_hkl_bracket(self.projection.center_hkl)} "
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
        line_h = margin * 0.075
        top_y = -margin * 1.30
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
