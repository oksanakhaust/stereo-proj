from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .projection import ProjectedPole, StereographicProjection

_LABEL_LIMIT = 80  # auto-disable labels when more poles than this


def _fmt_index(n: int) -> str:
    """Format a single Miller index with crystallographic overbar for negatives."""
    if n < 0:
        return f"{-n}̅"   # combining overline: 1̄, 2̄, …
    return str(n)


def _fmt_hkl(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    return f"({_fmt_index(h)}{_fmt_index(k)}{_fmt_index(l)})"


def _fmt_hkl_bracket(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    return f"[{_fmt_index(h)}{_fmt_index(k)}{_fmt_index(l)}]"


class StereogramRenderer:
    """Renders a stereographic projection using matplotlib and exports to file/bytes.

    Usage::

        proj = StereographicProjection((0, 0, 1), radius=100)
        poles = proj.project_all(hkl_list)
        renderer = StereogramRenderer(proj)
        renderer.draw(poles)
        renderer.export("cubic_001.png", "png")
        fig = renderer.get_figure()       # embed in Jupyter / Flask / Streamlit
        buf = renderer.get_bytes("jpeg")  # bytes for download buttons
    """

    def __init__(self, projection: StereographicProjection) -> None:
        self.projection = projection
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    # ------------------------------------------------------------------
    def draw(
        self,
        poles: list[ProjectedPole],
        *,
        show_grid: bool = True,
        show_labels: bool = True,
        grid_step: int = 10,
        title: str | None = None,
        custom_poles: list[ProjectedPole] | None = None,
    ) -> None:
        """Render the stereonet into an internal matplotlib Figure.

        Parameters
        ----------
        poles       : list of projected poles from StereographicProjection.
        show_grid   : draw concentric circles and radial lines (Wulff net).
        show_labels : annotate each pole with its (hkl) label.
        grid_step   : angular step in degrees for the grid (5, 10, 15, or 30).
        title       : override the default figure title.
        """
        # Release previous figure to free memory
        if self.fig is not None:
            plt.close(self.fig)

        R = self.projection.radius
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # ── Wulff net ──────────────────────────────────────────────────
        if show_grid:
            _gc = "#cccccc"
            _lw = 0.5
            rho_range = range(grid_step, 90, grid_step)
            for rho_deg in rho_range:
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

        # Cross-hair at centre
        ax.plot(0, 0, "+", color="black", markersize=6, markeredgewidth=0.8, zorder=3)

        # ── Standard poles ─────────────────────────────────────────────
        auto_labels = show_labels and len(poles) <= _LABEL_LIMIT

        for pole in poles:
            if pole.marker == "filled":
                ax.plot(pole.x, pole.y, "o",
                        color="black", markersize=5, markeredgewidth=0.8, zorder=4)
            else:
                ax.plot(pole.x, pole.y, "x",
                        color="black", markersize=6, markeredgewidth=1.2, zorder=4)

            if auto_labels:
                ax.annotate(
                    _fmt_hkl(pole.hkl),
                    (pole.x, pole.y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    zorder=5,
                )

        # ── Custom poles (user-defined, shown in red) ──────────────────
        if custom_poles:
            for pole in custom_poles:
                if pole.marker == "filled":
                    ax.plot(pole.x, pole.y, "o",
                            color="#d62728", markersize=7,
                            markeredgewidth=0.8, zorder=6)
                else:
                    ax.plot(pole.x, pole.y, "x",
                            color="#d62728", markersize=8,
                            markeredgewidth=1.5, zorder=6)
                ax.annotate(
                    _fmt_hkl(pole.hkl),
                    (pole.x, pole.y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#d62728",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

        # ── Legend ─────────────────────────────────────────────────────
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markersize=7, label="Верхняя полусфера"),
            Line2D([0], [0], marker="x", color="black", markersize=7,
                   markeredgewidth=1.2, label="Нижняя полусфера"),
        ]
        if custom_poles:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                       markersize=7, label="Заданные вручную")
            )
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=8, framealpha=0.8, handlelength=1)

        # ── Title ──────────────────────────────────────────────────────
        if title is None:
            title = f"Стереографическая проекция {_fmt_hkl_bracket(self.projection.center_hkl)}"
        ax.set_title(title, fontsize=13, pad=10)

        # ── Axis limits (add 15 % margin around the circle) ───────────
        margin = R * 1.18
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

        self.fig = fig
        self.ax = ax

    # ------------------------------------------------------------------
    def get_figure(self) -> plt.Figure:
        """Return the underlying matplotlib Figure (for embedding)."""
        if self.fig is None:
            raise RuntimeError("Call draw() before get_figure()")
        return self.fig

    def get_bytes(self, fmt: str = "png") -> io.BytesIO:
        """Return the rendered figure as a BytesIO buffer.

        Parameters
        ----------
        fmt : 'png', 'jpeg', or 'pdf'
        """
        if self.fig is None:
            raise RuntimeError("Call draw() before get_bytes()")
        buf = io.BytesIO()
        self.fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight",
                         facecolor="white")
        buf.seek(0)
        return buf

    def export(self, path: str, fmt: str = "png") -> None:
        """Save the figure to *path* in the given format."""
        if self.fig is None:
            raise RuntimeError("Call draw() before export()")
        self.fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight",
                         facecolor="white")
