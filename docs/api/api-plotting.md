# `precision_physkit.plotting` API

```python
from precision_physkit import plotting
```

The plotting module is a style and convenience layer. It does not transform
scientific data.

## Style generators

```python
plotting.get_adaptive_subplot_style(
    rows=1, cols=1, width=4.7, ratio=0.75,
    font_family="sans-serif", font_base_size=14.0, verbose=True,
)

plotting.get_academic_style(
    rows=1, cols=1, base_width_mm=150,
    use_surplus=1.0, ratio=0.75, verbose=True,
)
```

Both return Matplotlib style text suitable for `temp_style(extra_style=...)`.
The adaptive helper scales figures and text for a subplot grid. The academic
helper uses millimetre-based publication sizing and serif-oriented defaults.

## Temporary style

```python
plotting.temp_style(style_keys=None, extra_style="")
```

Acts as a context manager or decorator. It combines named preset fragments and
extra style text, applies a temporary `.mplstyle`, and deletes it afterward.
Unknown keys raise `ValueError`.

On exit it restores Matplotlib default `rcParams`, not the style state that was
active before entry. Do not rely on nested restoration.

Common layout presets include `ysy_academic`, `sci`, `ieee`, and `nature`;
palette presets include `tab4`, `tab8`, and `4blue_red`. Use
`plotting.print_preset_styles()` for the complete installed list.

## Convenience plotting

```python
plotting.plot(
    x, y, legend_name,
    plot_title="", x_label="X Axis", y_label="Y Axis",
    plot_type="curve", legend_title="", data_point=None,
    *, legend_out=True, legend_up=False, legend_fancy=True,
    svg_save=False, pdf_save=False, fig_show=True,
    x_log=False, y_log=False, return_fig=False,
    x_lim=None, y_lim=None, adjust=False, line_alpha=1.0,
)
```

`plot_type` is `"curve"` or `"scatter"`. Inputs may represent one or multiple
series. Saving and display are controlled by the keyword-only flags;
`return_fig=True` returns `(fig, ax)` only when `fig_show=False`.

```python
plotting.mamplot(...)
plotting.print_preset_styles()
```

`mamplot` is the lower-level compatibility plotting entry point used by `plot`.
Prefer `plot` for new code unless direct compatibility behavior is required.

## Two-stage workflow

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from precision_physkit import plotting

style = plotting.get_adaptive_subplot_style(rows=1, cols=1, verbose=False)
with plotting.temp_style(["tab4"], extra_style=style):
    fig, ax = plt.subplots()
    ax.loglog(frequency, asd, label="ASD")
    ax.legend(frameon=False)
    fig.savefig("artifacts/figures/01_asd.pdf", bbox_inches="tight")
    plt.close(fig)
```

Generate style first, enter the style context, then draw data. Avoid overriding
color, line width, line style, or fonts inside the block; local legend sizing or
`frameon=False` is acceptable when needed to prevent overlap. In headless scripts,
select the `Agg` backend before importing `pyplot`.

Always visually inspect figures for clipped labels, overlaps, illegible legends,
and inconsistent panels. Curated comparison images belong in `reference/`; new
outputs belong in `artifacts/figures/`.

## See also

[Pipeline](api-pipeline.md) · [Spectral analysis](api-spectral.md)
