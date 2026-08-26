"""Preview the project colour system and verify luminance monotonicity.

A sequential map must rise monotonically in luminance — that is what makes
magnitude legible as brightness, and it is what breaks if anchors route
through an unintended hue. Run this after editing style.py.

    python tests\\preview_palette.py
"""

import numpy as np
import matplotlib.pyplot as plt
from lbm.src.plot import style


def relative_luminance(rgb):
    """WCAG relative luminance: linearise sRGB, then weight by cone response."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def main():
    maps = [
        ("SEQUENTIAL_COOL - forward", style.SEQUENTIAL_COOL, True),
        ("SEQUENTIAL_WARM - adjoint", style.SEQUENTIAL_WARM, True),
        ("DIVERGING - sensitivity", style.DIVERGING, False),
        ("DESIGN - pipe", style.DESIGN, True),
    ]

    fig, axes = plt.subplots(len(maps), 2, figsize=(13, 2.1 * len(maps)),
                             gridspec_kw={"width_ratios": [3, 1]})

    samples = np.linspace(0, 1, 256)
    for (name, cmap, expect_monotonic), (ax_swatch, ax_lum) in zip(maps, axes):
        rgb = cmap(samples)[:, :3]
        lum = relative_luminance(rgb)

        ax_swatch.imshow(samples[None, :], aspect="auto", cmap=cmap)
        ax_swatch.set_title(name, fontsize=10, loc="left", color=style.TEXT)
        ax_swatch.set_xticks([]); ax_swatch.set_yticks([])

        ax_lum.plot(samples, lum, color=style.TEXT, lw=1.6)
        ax_lum.set_ylim(0, 1)
        ax_lum.set_xticks([]); ax_lum.tick_params(labelsize=7)

        if expect_monotonic:
            worst_dip = float(np.min(np.diff(lum)))
            ok = worst_dip > -1e-3
            ax_lum.set_title(
                f"luminance {'monotonic' if ok else f'DIP {worst_dip:.4f}'}",
                fontsize=8, color=style.TEXT if ok else style.CORAL)
        else:
            ax_lum.set_title("diverging: dip at centre expected",
                             fontsize=8, color=style.MUTED)

    style.apply_figure_style(fig, axes.ravel())
    fig.tight_layout()
    out = "results/tests/palette_preview.png"
    fig.savefig(out, dpi=150, facecolor=style.GROUND)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()