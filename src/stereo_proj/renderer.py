from __future__ import annotations

import io
import math
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

_RIM_THRESH = 0.85  # poles beyond this fraction of R get labels outside the circle


def _find_logo() -> str | None:
    import glob
    for pattern in ["misis_logo.png", "*/misis_logo.png"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def _fmt_index(n: int) -> str:
    if n < 0:
        return r"\overline{" + str(-n) + "}"
    return str(n)


# ── formatters with brackets (used in title) ──────────────────────────────────

def _fmt_hkl(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(l)
    return f"$({inner})$"


def _fmt_hkl_bracket(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(l)
    return f"$[{inner}]$"


def _fmt_hkil(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    i = -(h + k)
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(i) + _fmt_index(l)
    return f"$({inner})$"


def _fmt_uvtw(uvw: tuple[int, int, int]) -> str:
    U, V, W = uvw
    u3, v3, t3, w3 = 2 * U - V, 2 * V - U, -(U + V), 3 * W
    g = math.gcd(math.gcd(math.gcd(abs(u3), abs(v3)), abs(t3)), abs(w3))
    if g:
        u3, v3, t3, w3 = u3 // g, v3 // g, t3 // g, w3 // g
    inner = _fmt_index(u3) + _fmt_index(v3) + _fmt_index(t3) + _fmt_index(w3)
    return f"$[{inner}]$"


# ── bare formatters (no brackets, used for all pole labels) ───────────────────

def _fmt_hkl_bare(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(l)
    return f"${inner}$"


def _fmt_hkil_bare(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    i = -(h + k)
    inner = _fmt_index(h) + _fmt_index(k) + _fmt_index(i) + _fmt_index(l)
    return f"${inner}$"


def _fmt_uvtw_bare(uvw: tuple[int, int, int]) -> str:
    U, V, W = uvw
    u3, v3, t3, w3 = 2 * U - V, 2 * V - U, -(U + V), 3 * W
    g = math.gcd(math.gcd(math.gcd(abs(u3), abs(v3)), abs(t3)), abs(w3))
    if g:
        u3, v3, t3, w3 = u3 // g, v3 // g, t3 // g, w3 // g
    inner = _fmt_index(u3) + _fmt_index(v3) + _fmt_index(t3) + _fmt_index(w3)
    return f"${inner}$"


def _rim_text_align(phi: float) -> tuple[str, str]:
    """Return (ha, va) so that text placed outside the circle at angle phi
    reads outward from the circle boundary."""
    a = abs(phi)
    if a < math.pi / 3:
        return "left", "center"
    elif a > 2 * math.pi / 3:
        return "right", "center"
    elif phi > 0:
        return "center", "bottom"
    else:
        return "center", "top"


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
        marker_mode: str = "auto",
        cs_obj: object = None,
    ) -> None:
        if self.fig is not None:
            plt.close(self.fig)

        # Pole labels — always bare (no brackets)
        if use_miller_bravais:
            _lbl_plane = _fmt_hkil_bare
            _lbl_dir   = _fmt_uvtw_bare
            _lbl_plane_t = _fmt_hkil       # with brackets for title
            _lbl_dir_t   = _fmt_uvtw
        else:
            _lbl_plane = _fmt_hkl_bare
            _lbl_dir   = _fmt_hkl_bare
            _lbl_plane_t = _fmt_hkl
            _lbl_dir_t   = _fmt_hkl_bracket

        R = self.projection.radius

        plt.rcParams["font.family"] = "DejaVu Sans"

        fig_size = float(np.clip(8.0 * np.sqrt(R / 100.0), 8.0, 16.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # ── Wulff net (equatorial, 2° fine + 10° bold) ─────────────────
        if show_grid:

            def _arc_wulff(cx, cy, rc):
                """Arc of circle (cx,cy,rc) clipped to disk of radius R.
                Uses analytic intersection so there are no sampling gaps."""
                d2 = cx * cx + cy * cy
                d = math.sqrt(d2)
                eps = 1e-9
                if d + rc < R - eps:
                    # Entirely inside — full circle
                    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
                    return cx + rc * np.cos(t), cy + rc * np.sin(t)
                if d > rc + R + eps or rc > d + R + eps or d + R < rc - eps:
                    return None
                if d < eps:
                    return None
                phi = math.atan2(cy, cx)
                cosv = (R * R - d2 - rc * rc) / (2 * rc * d)
                cosv = max(-1.0, min(1.0, cosv))
                dt = math.acos(cosv)
                t1, t2 = phi - dt, phi + dt
                xm = cx + rc * math.cos((t1 + t2) / 2)
                ym = cy + rc * math.sin((t1 + t2) / 2)
                if xm * xm + ym * ym <= R * R:
                    t_arr = np.linspace(t1, t2, 400)
                else:
                    t_arr = np.linspace(t2, t1 + 2 * math.pi, 400)
                return cx + rc * np.cos(t_arr), cy + rc * np.sin(t_arr)

            def _draw_net(step, color, lw):
                ax.plot([0, 0], [-R, R], color=color, linewidth=lw, zorder=1)
                ax.plot([-R, R], [0, 0], color=color, linewidth=lw, zorder=1)
                for a_deg in range(step, 90, step):
                    a = math.radians(a_deg)
                    m_r, m_c = R / math.sin(a), R / math.tan(a)
                    p_c, p_r = R / math.cos(a), R * math.tan(a)
                    for s in (-1, 1):
                        for (cx, cy, rc) in [(s * m_c, 0, m_r), (0, s * p_c, p_r)]:
                            res = _arc_wulff(cx, cy, rc)
                            if res is not None:
                                ax.plot(res[0], res[1], color=color, linewidth=lw, zorder=1)

            _draw_net(2,  "#cccccc", 0.4)   # 2°  fine grid — very light
            _draw_net(10, "#666666", 1.2)   # 10° bold grid — clearly darker/thicker

        # ── Outer boundary circle (thin dark-gray so rim poles show) ────
        outer = plt.Circle((0, 0), R, fill=False, color="#555555", linewidth=0.7, zorder=2)
        ax.add_patch(outer)

        # Centre dot
        ax.plot(0, 0, ".", color="black", markersize=3, markeredgewidth=0, zorder=3)

        # Centre label
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

        # ── Pre-compute size factors for d_hkl / 1/P modes ───────────
        _size_raw: dict[tuple, float] = {}
        _max_f = 1.0
        if marker_mode in ("d_hkl", "1/P") and cs_obj is not None:
            for group in groups:
                rep = next((p for p in group if p.marker == "filled"), group[0])
                try:
                    f = (cs_obj.d_relative(rep.hkl) if marker_mode == "d_hkl"
                         else 1.0 / cs_obj.multiplicity(rep.hkl))
                except Exception:
                    f = 1.0
                _size_raw[rep.hkl] = f
            if _size_raw:
                _max_f = max(_size_raw.values())

        # ── Standard poles ─────────────────────────────────────────────
        label_texts = []

        for group in groups:
            has_filled = any(p.marker == "filled" for p in group)
            has_open   = any(p.marker == "open"   for p in group)
            rep = next((p for p in group if p.marker == "filled"), group[0])
            x, y = rep.x, rep.y

            # Marker size
            h, k, l = rep.hkl
            sum_sq = h * h + k * k + l * l
            if marker_mode == "fixed":
                ms_g = ms_auto
            elif marker_mode in ("d_hkl", "1/P") and rep.hkl in _size_raw:
                ms_g = ms_auto * max(0.3, _size_raw[rep.hkl] / _max_f)
            else:
                ms_g = ms_auto

            if has_filled and has_open:
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.9 * _s, zorder=4)
                ax.plot(x, y, "o", color="black",
                        markersize=max(1.0, ms_g * 0.35),
                        markeredgewidth=0, zorder=5)
            elif has_filled:
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markeredgewidth=0.6 * _s, zorder=4)
            else:
                ax.plot(x, y, "o", color="black", markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.8 * _s, zorder=4)

            if show_labels:
                r_pole = math.sqrt(x * x + y * y)
                if r_pole > R * _RIM_THRESH and r_pole > 1e-6:
                    phi = math.atan2(y, x)
                    lx = R * 1.04 * math.cos(phi)
                    ly = R * 1.04 * math.sin(phi)
                    ha, va = _rim_text_align(phi)
                    t = ax.text(lx, ly, _lbl_plane(rep.hkl),
                                fontsize=fs_auto, ha=ha, va=va, zorder=6)
                else:
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
                r_pole = math.sqrt(pole.x * pole.x + pole.y * pole.y)
                if r_pole > R * _RIM_THRESH and r_pole > 1e-6:
                    phi = math.atan2(pole.y, pole.x)
                    lx = R * 1.04 * math.cos(phi)
                    ly = R * 1.04 * math.sin(phi)
                    ha, va = _rim_text_align(phi)
                    t = ax.text(lx, ly, _lbl_plane(pole.hkl),
                                fontsize=fs_custom, fontweight="bold", color="#d62728",
                                ha=ha, va=va, zorder=7)
                else:
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
                f"{_lbl_dir_t(self.projection.center_hkl)} "
                f"для {crystal_system} сингонии"
            )
        ax.set_title(title, fontsize=12, pad=10)

        # ── Axis limits (wider to accommodate outside-rim labels) ───────
        margin = R * 1.18
        ax.set_xlim(-R * 1.22, R * 1.22)
        ax.set_ylim(-margin * 1.52, R * 1.22)

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
        top_y = -margin * 1.355
        for i, line in enumerate(lines):
            ax.text(
                sig_x, top_y - i * line_h,
                line,
                fontsize=7, va="top", ha="left", color="#555555",
                fontfamily="DejaVu Sans",
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
