"""Streamlit web interface for the stereographic projection generator."""

import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import streamlit.components.v1 as components

from stereo_proj.crystal.cubic import CubicSystem
from stereo_proj.crystal.tetragonal import TetragonalSystem
from stereo_proj.crystal.hexagonal import HexagonalSystem
from stereo_proj.projection import StereographicProjection
from stereo_proj.renderer import StereogramRenderer

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Стереографические проекции",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>"
    "#MainMenu{visibility:hidden;}"
    "[data-testid='stToolbar']{display:none;}"
    "[data-testid='stDecoration']{display:none;}"
    ".stDeployButton{display:none;}"
    "[data-testid='collapsedControl']{"
    "  position:fixed !important;"
    "  left:0 !important;"
    "  top:16px !important;"
    "  z-index:99999 !important;"
    "  display:flex !important;"
    "  visibility:visible !important;"
    "  opacity:1 !important;"
    "  background-color:#1565C0 !important;"
    "  border-radius:0 10px 10px 0 !important;"
    "  width:28px !important;"
    "  height:72px !important;"
    "  align-items:center !important;"
    "  justify-content:center !important;"
    "  cursor:pointer !important;"
    "}"
    "[data-testid='collapsedControl'] svg{fill:white !important;color:white !important;}"
    "</style>",
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (function() {
        var d = window.parent.document;
        if (d.getElementById('_sb_toggle')) return;
        var b = d.createElement('button');
        b.id = '_sb_toggle';
        b.title = 'Открыть / закрыть панель';
        b.innerHTML = '&#9776;';
        b.style.cssText = 'position:fixed;left:0;top:16px;'
            + 'z-index:2147483647;background:#1565C0;color:#fff;border:none;'
            + 'border-radius:0 10px 10px 0;width:32px;height:72px;'
            + 'font-size:20px;cursor:pointer;padding:0;line-height:1;'
            + 'box-shadow:2px 0 6px rgba(0,0,0,.3);';
        b.addEventListener('click', function() {
            var sel = [
                '[data-testid="collapsedControl"]',
                '[data-testid="stSidebarCollapsedControl"]',
                'button[aria-label*="sidebar"]',
                'button[aria-label*="Sidebar"]',
                '[data-testid="stSidebar"] button',
                'section[data-testid="stSidebar"] button'
            ];
            for (var i = 0; i < sel.length; i++) {
                var el = d.querySelector(sel[i]);
                if (el) { el.click(); return; }
            }
            d.dispatchEvent(new KeyboardEvent('keydown', {key:'.', bubbles:true}));
        });
        d.body.appendChild(b);
    })();
    </script>
    """,
    height=0,
    scrolling=False,
)

st.title("Стереографические проекции")
st.caption(
    "Генератор полюсных фигур для кубической, тетрагональной и гексагональной "
    "кристаллографических систем. "
    "Выберите параметры слева и нажмите **«Построить проекцию»**."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — параметры
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Параметры")

    # --- Сингония ---
    st.subheader("Сингония")
    crystal_choice = st.selectbox(
        "Кристаллическая система",
        options=["Кубическая", "Тетрагональная", "Гексагональная"],
        index=0,
    )
    c_over_a = 1.0
    if crystal_choice == "Тетрагональная":
        c_over_a = st.number_input(
            "Параметр c/a",
            min_value=0.1,
            max_value=10.0,
            value=1.5,
            step=0.05,
            format="%.3f",
            help="Отношение параметров решётки c/a.",
            key="ca_tet",
        )
    elif crystal_choice == "Гексагональная":
        c_over_a = 1.6330

    st.divider()

    # --- Центр проекции ---
    st.subheader("Центр проекции [HKL]")
    c1, c2, c3 = st.columns(3)
    H = int(c1.number_input("H", value=0, step=1, min_value=-10, max_value=10, key="H"))
    K = int(c2.number_input("K", value=0, step=1, min_value=-10, max_value=10, key="K"))
    L = int(c3.number_input("L", value=1, step=1, min_value=-10, max_value=10, key="L"))

    st.divider()

    # --- Плоскости ---
    st.subheader("Плоскости")
    only_custom = st.checkbox(
        "Только заданные вручную",
        value=False,
        help="Скрыть все автоматически сгенерированные полюсы.",
    )
    if not only_custom:
        max_sum_sq = st.number_input(
            "Макс. h²+k²+l²",
            min_value=1,
            max_value=100,
            value=9,
            step=1,
            help=(
                "Ограничивает набор плоскостей: все (hkl) с h²+k²+l² ≤ N. "
                "N=1 → {100}, N=2 → +{110}, N=3 → +{111}, N=5 → +{210}, …"
            ),
        )
    else:
        max_sum_sq = 9

    st.divider()

    # --- Вид ---
    st.subheader("Вид")
    radius = st.slider("Радиус сетки", min_value=50, max_value=600, value=100, step=10)

    show_grid = st.checkbox("Сетка Вульфа", value=True)
    show_labels = st.checkbox("Подписи (hkl)", value=True)

    _MARKER_OPTS = ["fixed", "d_hkl", "1/P"]
    _MARKER_LABELS = {
        "fixed": "Постоянный",
        "d_hkl": "∝ d(hkl)",
        "1/P":   "∝ 1/P(hkl)",
    }
    marker_mode = st.radio(
        "Диаметр полюса",
        options=_MARKER_OPTS,
        format_func=lambda x: _MARKER_LABELS[x],
        horizontal=True,
    )

    st.divider()

    # --- Произвольные точки ---
    st.subheader("Дополнительные полюсы")
    st.caption("По одному (hkl) на строку, через пробел. Например: `1 2 3`")
    custom_text = st.text_area(
        "Список (hkl)",
        value="",
        height=120,
        placeholder="1 2 3\n0 1 1\n2 1 0",
        key="custom_hkl",
    )

    st.divider()
    build_btn = st.button(
        "Построить проекцию",
        use_container_width=True,
        type="primary",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Build logic
# ──────────────────────────────────────────────────────────────────────────────
def _parse_custom_hkl(text: str) -> tuple[list[tuple[int, int, int]], str]:
    poles = []
    for lineno, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            return [], f"Строка {lineno}: ожидается 3 индекса, получено {len(parts)} ({line!r})"
        try:
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return [], f"Строка {lineno}: индексы должны быть целыми числами ({line!r})"
        if h == 0 and k == 0 and l == 0:
            return [], f"Строка {lineno}: (0 0 0) не является допустимым полюсом"
        poles.append((h, k, l))
    return poles, ""


if build_btn:
    center = (H, K, L)

    if center == (0, 0, 0):
        st.session_state["error"] = "Центр проекции не может быть (0, 0, 0)."
        st.session_state.pop("renderer", None)
    else:
        custom_hkl_list, parse_error = _parse_custom_hkl(custom_text)
        if parse_error:
            st.session_state["error"] = f"Ошибка в ручном вводе: {parse_error}"
            st.session_state.pop("renderer", None)
        else:
            with st.spinner("Вычисляю проекцию…"):
                try:
                    proj = StereographicProjection(center, radius=float(radius))
                    hemi = "upper"

                    if crystal_choice == "Тетрагональная":
                        system = TetragonalSystem(c_over_a=float(c_over_a))
                        system_label = f"тетрагональной (c/a = {c_over_a:.3f})"
                    elif crystal_choice == "Гексагональная":
                        system = HexagonalSystem(c_over_a=float(c_over_a))
                        system_label = "гексагональной (ГПУ)"
                    else:
                        system = CubicSystem()
                        system_label = "кубической"

                    if only_custom:
                        poles = []
                    else:
                        hkl_list = system.generate_hkl(max_sum_sq=int(max_sum_sq))
                        poles = proj.project_all(hkl_list, hemisphere=hemi,
                                                 crystal_system=system)

                    custom_poles = (
                        proj.project_custom(custom_hkl_list, hemisphere=hemi,
                                            crystal_system=system)
                        if custom_hkl_list else []
                    )

                    # For hexagonal: rotate so (10-10) = (1,0,0) ends up at the bottom
                    phi_rotation = 0.0
                    if crystal_choice == "Гексагональная":
                        import math as _m
                        _p1010 = proj.project((1, 0, 0), hemisphere="both",
                                              crystal_system=system)
                        if _p1010 is not None:
                            _cur = _m.atan2(_p1010.y, _p1010.x)
                            _tgt = -_m.pi / 2  # bottom center
                            phi_rotation = _m.degrees(_tgt - _cur)

                    renderer = StereogramRenderer(proj)
                    renderer.draw(
                        poles,
                        show_grid=show_grid,
                        show_labels=show_labels,
                        custom_poles=custom_poles or None,
                        crystal_system=system_label,
                        use_miller_bravais=(crystal_choice == "Гексагональная"),
                        marker_mode=marker_mode,
                        cs_obj=system,
                        phi_rotation=phi_rotation,
                    )
                    st.session_state["renderer"] = renderer
                    st.session_state["n_poles"] = len(poles)
                    st.session_state["n_custom"] = len(custom_poles)
                    st.session_state["center"] = center
                    # Pre-compute export bytes at build time to avoid
                    # stale-object AttributeErrors on page re-renders
                    st.session_state["export_a4"]  = renderer.get_bytes_a4_pdf()
                    st.session_state["export_jpeg"] = renderer.get_bytes("jpeg")
                    st.session_state["export_png"]  = renderer.get_bytes("png")
                    st.session_state["export_pdf"]  = renderer.get_bytes("pdf")
                    st.session_state["export_svg"]  = renderer.get_bytes("svg")
                    st.session_state["export_eps"]  = renderer.get_bytes("eps")
                    st.session_state.pop("error", None)
                except ValueError as exc:
                    st.session_state["error"] = str(exc)
                    st.session_state.pop("renderer", None)

# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(st.session_state["error"])

elif "renderer" in st.session_state:
    # If export keys are missing the projection was built with an older version —
    # ask the user to rebuild so all bytes get pre-computed fresh.
    if "export_a4" not in st.session_state:
        st.warning("Нажмите **«Построить проекцию»** ещё раз — идёт обновление приложения.")
        st.stop()

    renderer: StereogramRenderer = st.session_state["renderer"]
    n_poles: int = st.session_state["n_poles"]
    center: tuple = st.session_state["center"]

    n_custom: int = st.session_state.get("n_custom", 0)
    info_msg = f"Отображено полюсов: **{n_poles}**"
    if n_custom:
        info_msg += f"  +  **{n_custom}** заданных вручную (красные кружки)"
    st.info(info_msg)

    col_fig, col_dl = st.columns([3, 1])
    with col_fig:
        st.pyplot(renderer.get_figure(), use_container_width=True)

    with col_dl:
        st.subheader("Экспорт")
        center_str = "".join(
            (str(x) if x >= 0 else f"m{-x}") for x in center
        )

        st.download_button(
            "Печать А4 (PDF)",
            data=st.session_state["export_a4"],
            file_name=f"stereo_{center_str}_A4.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        st.divider()

        st.download_button(
            "JPEG",
            data=st.session_state["export_jpeg"],
            file_name=f"stereo_{center_str}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
        st.download_button(
            "PNG",
            data=st.session_state["export_png"],
            file_name=f"stereo_{center_str}.png",
            mime="image/png",
            use_container_width=True,
        )
        st.download_button(
            "PDF",
            data=st.session_state["export_pdf"],
            file_name=f"stereo_{center_str}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "SVG",
            data=st.session_state["export_svg"],
            file_name=f"stereo_{center_str}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
        st.download_button(
            "EPS",
            data=st.session_state["export_eps"],
            file_name=f"stereo_{center_str}.eps",
            mime="application/postscript",
            use_container_width=True,
        )

        st.caption(
            "**Печать А4** — векторный PDF, диаметр ≈ 200 мм.  \n"
            "JPEG / PNG — растровые 300 dpi.  \n"
            "PDF / SVG / EPS — векторные, текущий масштаб."
        )

else:
    st.info("Настройте параметры в боковой панели и нажмите «Построить проекцию».")
