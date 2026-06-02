"""Example: standard [001] stereographic projection of the cubic system."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stereo_proj import generate

renderer = generate(
    center=(0, 0, 1),
    max_sum_sq=9,
    radius=100,
    hemisphere="both",
    show_grid=True,
    show_labels=True,
)
renderer.export("cubic_001.png", "png")
print("Saved: cubic_001.png")
