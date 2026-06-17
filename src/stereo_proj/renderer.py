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

_RIM_THRESH = 0.97  # poles beyond this fraction of R get labels outside the circle


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


def _rim_text_align(phi: float) -> tuple[str, str, float, float]:
    """Return (ha, va, dx, dy) for a rim label at angle phi.

    dx, dy are extra offsets in units of R applied to the anchor
    (in addition to the base R*1.01 radial position).

    Left/right extremes (|phi| < 60° or > 120°): label centered
    below the dot so it never overflows the sides of the figure.
    Top/bottom arc: label above or below the dot radially.
    """
    a = abs(phi)
    if a < math.pi / 3 or a > 2 * math.pi / 3:
        # Left / right extreme — place text centered directly below the dot
        return "center", "top", 0.0, -0.04
    elif phi > 0:
        return "center", "bottom", 0.0, 0.0   # upper arc: text above
    else:
        return "center", "top",    0.0, 0.0   # lower arc: text below


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
        phi_rotation: float = 0.0,
        _paper: str = "screen",
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

        _cos_r = math.cos(math.radians(phi_rotation))
        _sin_r = math.sin(math.radians(phi_rotation))

        def _rot(px, py):
            return px * _cos_r - py * _sin_r, px * _sin_r + py * _cos_r

        custom_hkl_set = {p.hkl for p in (custom_poles or [])}

        plt.rcParams["font.family"] = "DejaVu Sans"

        if _paper == "a4":
            # A4 portrait: 210×297 mm = 8.268×11.693 inches, 300 dpi for print
            fig, ax = plt.subplots(figsize=(8.268, 11.693), dpi=300)
            # Remove side margins so circle fills full page width → exactly 200 mm diameter
            # Keep 7% top margin for the title text
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
        else:
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
            _draw_net(10, "#666666", 0.4)   # 10° grid — same lw, darker color

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
        # Auto-disable labels when too many poles — unreadable anyway and slow
        _too_many = len(groups) > 300
        if _too_many:
            show_labels = False

        inner_texts = []  # fed to adjust_text (interior poles only)

        for group in groups:
            has_filled = any(p.marker == "filled" for p in group)
            has_open   = any(p.marker == "open"   for p in group)
            rep = next((p for p in group if p.marker == "filled"), group[0])
            x, y = _rot(rep.x, rep.y)

            _highlighted = any(p.hkl in custom_hkl_set for p in group)
            _pc = "#d62728" if _highlighted else "black"

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
                ax.plot(x, y, "o", color=_pc, markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.9 * _s, zorder=4)
                ax.plot(x, y, "o", color=_pc,
                        markersize=max(1.0, ms_g * 0.35),
                        markeredgewidth=0, zorder=5)
            elif has_filled:
                ax.plot(x, y, "o", color=_pc, markersize=ms_g,
                        markeredgewidth=0.6 * _s, zorder=4)
            else:
                ax.plot(x, y, "o", color=_pc, markersize=ms_g,
                        markerfacecolor="none", markeredgewidth=0.8 * _s, zorder=4)

            if show_labels:
                r_pole = math.sqrt(x * x + y * y)
                if r_pole > R * _RIM_THRESH and r_pole > 1e-6:
                    phi = math.atan2(y, x)
                    ha, va, dx, dy = _rim_text_align(phi)
                    lx = R * 1.01 * math.cos(phi) + dx * R
                    ly = R * 1.01 * math.sin(phi) + dy * R
                    # Rim labels: fixed position, not passed to adjust_text
                    ax.text(lx, ly, _lbl_plane(rep.hkl),
                            fontsize=fs_auto, ha=ha, va=va, zorder=6,
                            color=_pc)
                else:
                    t = ax.text(x, y, _lbl_plane(rep.hkl),
                                fontsize=fs_auto, ha="center", va="bottom", zorder=6,
                                color=_pc)
                    inner_texts.append(t)

        # ── Custom poles (red circles / red crosses) ───────────────────
        # Poles that duplicate an existing standard pole are already highlighted
        # red in the standard group — skip re-drawing them here.
        if custom_poles:
            _existing_hkl = {p.hkl for g in groups for p in g}
            for pole in custom_poles:
                if pole.hkl in _existing_hkl:
                    continue
                px, py = _rot(pole.x, pole.y)
                if pole.marker == "filled":
                    ax.plot(px, py, "o",
                            color="#d62728", markersize=ms_custom,
                            markeredgewidth=0.7 * _s, zorder=6)
                else:
                    ax.plot(px, py, "x",
                            color="#d62728", markersize=ms_cx,
                            markeredgewidth=1.2 * _s, zorder=6)
                r_pole = math.sqrt(px * px + py * py)
                if r_pole > R * _RIM_THRESH and r_pole > 1e-6:
                    phi = math.atan2(py, px)
                    ha, va, dx, dy = _rim_text_align(phi)
                    lx = R * 1.01 * math.cos(phi) + dx * R
                    ly = R * 1.01 * math.sin(phi) + dy * R
                    ax.text(lx, ly, _lbl_plane(pole.hkl),
                            fontsize=fs_custom, fontweight="bold", color="#d62728",
                            ha=ha, va=va, zorder=7)
                else:
                    t = ax.text(px, py, _lbl_plane(pole.hkl),
                                fontsize=fs_custom, fontweight="bold", color="#d62728",
                                ha="center", va="bottom", zorder=7)
                    inner_texts.append(t)

        # ── Repel overlapping interior labels only (screen mode, rim labels stay fixed) ──
        if inner_texts and len(inner_texts) <= 150 and _paper == "screen":
            try:
                from adjustText import adjust_text
                adjust_text(inner_texts, ax=ax, expand=(1.2, 1.4))
            except Exception:
                pass


        # ── Title ──────────────────────────────────────────────────────
        if title is None:
            title = (
                f"Стереографическая проекция "
                f"{_lbl_dir_t(self.projection.center_hkl)} "
                f"для {crystal_system} сингонии"
            )
        ax.set_title(title, fontsize=12, pad=10)

        # ── Axis limits ────────────────────────────────────────────────
        # A4: xlim=±1.05R → with no side margins, circle = exactly 200mm diameter
        # Screen: wider margins for crowded labels
        margin = R * (0.95 if _paper == "a4" else 1.18)
        xlim_f = 1.05 if _paper == "a4" else 1.30
        ax.set_xlim(-R * xlim_f, R * xlim_f)
        ax.set_ylim(-margin * 1.52, R * 1.22)

        # ── Watermark: logo + signature ────────────────────────────────
        logo_path = _find_logo()

        if _paper == "a4":
            # Use figure-fraction coordinates so logo/sig reliably sit
            # in the bottom margin, independent of data coordinate scaling.
            _fl = 0.03   # left edge fraction
            _fb = 0.015  # bottom edge fraction (~4 mm from page bottom)
            _fh = 0.045  # logo height fraction (~13 mm)
            _sig_x_frac = _fl

            if logo_path:
                try:
                    logo_img = plt.imread(logo_path)
                    h_px, w_px = logo_img.shape[:2]
                    _fw = _fh * (w_px / h_px) * (
                        fig.get_figheight() / fig.get_figwidth()
                    )
                    logo_ax2 = fig.add_axes([_fl, _fb, _fw, _fh])
                    logo_ax2.imshow(logo_img)
                    logo_ax2.axis("off")
                    _sig_x_frac = _fl + _fw + 0.01
                except Exception:
                    pass

            for i, line in enumerate(_SIGNATURE.split("\n")):
                fig.text(
                    _sig_x_frac,
                    _fb + _fh - i * (_fh * 0.45),
                    line,
                    fontsize=7, va="top", ha="left", color="#555555",
                    fontfamily="DejaVu Sans",
                )
        else:
            # Screen mode: data-coordinate based placement
            sig_x = -margin * 0.98

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

        # Store args so get_bytes_a4_pdf() can re-render at A4 size
        if _paper == "screen":
            self._draw_state = dict(
                poles=poles, custom_poles=custom_poles,
                show_grid=show_grid, show_labels=show_labels,
                crystal_system=crystal_system,
                use_miller_bravais=use_miller_bravais,
                marker_mode=marker_mode, cs_obj=cs_obj,
                phi_rotation=phi_rotation,
            )

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

    def get_bytes_a4_pdf(self) -> io.BytesIO:
        """Re-render at true A4 portrait (210×297 mm), circle diameter ≈ 194 mm."""
        if not hasattr(self, "_draw_state") or self._draw_state is None:
            raise RuntimeError("Call draw() before get_bytes_a4_pdf()")

        s = self._draw_state
        saved_fig, saved_ax = self.fig, self.ax
        self.fig = None  # prevent draw() from closing the screen figure

        self.draw(
            s["poles"],
            show_grid=s["show_grid"],
            show_labels=s["show_labels"],
            custom_poles=s["custom_poles"],
            crystal_system=s["crystal_system"],
            use_miller_bravais=s["use_miller_bravais"],
            marker_mode=s["marker_mode"],
            cs_obj=s["cs_obj"],
            phi_rotation=s["phi_rotation"],
            _paper="a4",
        )

        buf = io.BytesIO()
        # Save full A4 page — no bbox_inches="tight" so dimensions stay exact
        self.fig.savefig(buf, format="pdf", facecolor="white")
        buf.seek(0)

        plt.close(self.fig)
        self.fig = saved_fig
        self.ax = saved_ax
        return buf

    def export(self, path: str, fmt: str = "png") -> None:
        if self.fig is None:
            raise RuntimeError("Call draw() before export()")
        self.fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight",
                         facecolor="white")
