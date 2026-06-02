"""Example: [111] projection, dense pole set, no labels."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stereo_proj import generate

renderer = generate(
    center=(1, 1, 1),
    max_sum_sq=14,   # includes up to {321}, {322} families
    radius=120,
    hemisphere="both",
    show_grid=True,
    show_labels=False,
)
renderer.export("cubic_111.pdf", "pdf")
print("Saved: cubic_111.pdf")
