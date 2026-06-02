"""stereo_proj — stereographic projection generator for crystallographic applications.

Quick usage::

    from stereo_proj import generate

    renderer = generate(
        center=(0, 0, 1),
        max_sum_sq=9,
        radius=100.0,
        hemisphere="both",
        show_grid=True,
        show_labels=True,
    )
    renderer.export("cubic_001.png", "png")
    fig = renderer.get_figure()   # matplotlib Figure for embedding
    buf = renderer.get_bytes("jpeg")  # BytesIO for download
"""

from .crystal.cubic import CubicSystem
from .projection import StereographicProjection, ProjectedPole
from .renderer import StereogramRenderer

__all__ = [
    "CubicSystem",
    "StereographicProjection",
    "ProjectedPole",
    "StereogramRenderer",
    "generate",
]

__version__ = "0.1.0"


def generate(
    center: tuple[int, int, int],
    max_sum_sq: int = 9,
    radius: float = 100.0,
    hemisphere: str = "both",
    show_grid: bool = True,
    show_labels: bool = True,
    grid_step: int = 10,
    max_points: int = 1000,
) -> StereogramRenderer:
    """One-call convenience wrapper: build and render a stereographic projection.

    Parameters
    ----------
    center      : Miller indices [H K L] of the projection centre pole.
    max_sum_sq  : Upper bound for h²+k²+l² (controls how many planes are shown).
    radius      : Radius of the outer circle in plot units (default 100).
    hemisphere  : 'upper', 'lower', or 'both'.
    show_grid   : Include Wulff net (concentric circles + radial lines).
    show_labels : Annotate each pole with its (hkl) label.
    grid_step   : Angular step in degrees for the Wulff net grid.
    max_points  : Safety cap on the number of poles (default 1000).

    Returns
    -------
    StereogramRenderer
        A renderer with the figure already drawn; call .get_figure(),
        .get_bytes(fmt), or .export(path, fmt) to extract output.
    """
    system = CubicSystem()
    hkl_list = system.generate_hkl(max_sum_sq=max_sum_sq, max_points=max_points)
    proj = StereographicProjection(center, radius=radius)
    poles = proj.project_all(hkl_list, hemisphere=hemisphere)
    renderer = StereogramRenderer(proj)
    renderer.draw(
        poles,
        show_grid=show_grid,
        show_labels=show_labels,
        grid_step=grid_step,
    )
    return renderer
