"""Streamlit web interface for the stereographic projection generator."""

import os
import sys

import matplotlib
matplotlib.use("Agg")

# Make the src/ layout importable when running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from stereo_proj.crystal.cubic import CubicSystem
from stereo_proj.projection import StereographicProjection
from stereo_proj.renderer import StereogramRenderer

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Стереографические проекции",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💎 Стереографические проекции")
st.caption(
    "Генератор полюсных фигур кубической кристаллографической системы. "
    "Выберите параметры слева и нажмите **«Построить проекцию»**."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — параметры
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Параметры")

    # --- Центр проекции ---
    st.subheader("Центр проекции [HKL]")
    c1, c2, c3 = st.columns(3)
    H = int(c1.number_input("H", value=0, step=1, min_value=-10, max_value=10, key="H"))
    K = int(c2.number_input("K", value=0, step=1, min_value=-10, max_value=10, key="K"))
    L = int(c3.number_input("L", value=1, step=1, min_value=-10, max_value=10, key="L"))

    st.divider()

    # --- Плоскости ---
    st.subheader("Плоскости")
    max_sum_sq = st.slider(
        "Макс. h²+k²+l²",
        min_value=1, max_value=25, value=9,
        help=(
            "Ограничивает набор плоскостей: все (hkl) с h²+k²+l² ≤ N. "
            "N=1 → {100}, N=2 → +{110}, N=3 → +{111}, N=5 → +{210}, …"
        ),
    )

    st.divider()

    # --- Вид ---
    st.subheader("Вид")
    radius = st.slider("Радиус сетки", min_value=50, max_value=300, value=100, step=10)

    hemisphere_label = st.selectbox(
        "Полусфера",
        options=["Обе", "Верхняя", "Нижняя"],
        index=0,
        help="Верхняя — закрашенные кружки; нижняя — контурные.",
    )
    _HEMI_MAP = {"Обе": "both", "Верхняя": "upper", "Нижняя": "lower"}

    show_grid = st.checkbox("Сетка Вульфа", value=True)
    grid_step = 10
    if show_grid:
        grid_step = st.select_slider(
            "Шаг сетки (°)", options=[5, 10, 15, 30], value=10
        )

    show_labels = st.checkbox("Подписи (hkl)", value=True)

    st.divider()
    build_btn = st.button(
        "▶ Построить проекцию",
        use_container_width=True,
        type="primary",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Build logic
# ──────────────────────────────────────────────────────────────────────────────
if build_btn:
    center = (H, K, L)

    if center == (0, 0, 0):
        st.session_state["error"] = "Центр проекции не может быть (0, 0, 0)."
        st.session_state.pop("renderer", None)
    else:
        with st.spinner("Вычисляю проекцию…"):
            try:
                system = CubicSystem()
                hkl_list = system.generate_hkl(max_sum_sq=max_sum_sq)
                proj = StereographicProjection(center, radius=float(radius))
                poles = proj.project_all(hkl_list, hemisphere=_HEMI_MAP[hemisphere_label])
                renderer = StereogramRenderer(proj)
                renderer.draw(
                    poles,
                    show_grid=show_grid,
                    show_labels=show_labels,
                    grid_step=grid_step,
                )
                st.session_state["renderer"] = renderer
                st.session_state["n_poles"] = len(poles)
                st.session_state["center"] = center
                st.session_state.pop("error", None)
            except ValueError as exc:
                st.session_state["error"] = str(exc)
                st.session_state.pop("renderer", None)

# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(f"❌ {st.session_state['error']}")

elif "renderer" in st.session_state:
    renderer: StereogramRenderer = st.session_state["renderer"]
    n_poles: int = st.session_state["n_poles"]
    center: tuple = st.session_state["center"]

    st.info(f"Отображено полюсов: **{n_poles}**")

    # ── Figure ──────────────────────────────────────────────────────
    col_fig, col_dl = st.columns([3, 1])
    with col_fig:
        st.pyplot(renderer.get_figure(), use_container_width=True)

    # ── Download buttons ────────────────────────────────────────────
    with col_dl:
        st.subheader("Экспорт")
        center_str = "".join(
            (str(x) if x >= 0 else f"m{-x}") for x in center
        )

        st.download_button(
            "⬇ JPEG",
            data=renderer.get_bytes("jpeg"),
            file_name=f"stereo_{center_str}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
        st.download_button(
            "⬇ PNG",
            data=renderer.get_bytes("png"),
            file_name=f"stereo_{center_str}.png",
            mime="image/png",
            use_container_width=True,
        )
        st.download_button(
            "⬇ PDF",
            data=renderer.get_bytes("pdf"),
            file_name=f"stereo_{center_str}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.caption(
            "JPEG / PNG — растровые изображения 300 dpi.  \n"
            "PDF — векторный формат, масштабируется без потерь."
        )

else:
    st.info("👈 Настройте параметры в боковой панели и нажмите **«Построить проекцию»**.")
    st.markdown(
        """
        ### Быстрый старт
        Стандартные проекции кубической системы:

        | Центр | Описание |
        |-------|---------|
        | [0 0 1] | Стандартная проекция {001} |
        | [0 1 1] | Стандартная проекция {011} |
        | [1 1 1] | Стандартная проекция {111} |

        **h²+k²+l²**: чем больше значение, тем больше плоскостей на проекции.
        Рекомендуется 9 для обзорного вида и 25 для детального.
        """
    )
