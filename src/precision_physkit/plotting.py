# -*- coding: utf-8 -*-
"""
precision_physkit.plotting (formerly ysy_mamplot.py)

Copyright (c) 2025 pifuyuini

Author: pifuyuini
Email: You can contact me via Github
Version: 1.11.1
Date: 2026-07-20

Description
-----------
A tiny, pragmatic plotting helper focused on two things:
1) simple, standardized plotting functions (`mamplot`, `plot`) for quick
   curves/scatters, and
2) a lightweight style manager (`temp_style`) that temporarily assembles and applies
   Matplotlib `.mplstyle` snippets (from built-in presets plus optional overrides).

This module is intentionally minimal. It is *not* a full plotting library—just a
thin layer to speed up everyday plotting and keep style usage consistent across
notebooks, scripts, and papers.

Key Features
------------
- `mamplot(...)`: one-call line/scatter plotting with sensible defaults, a clean legend
  placement option (`legend_out=True`), and optional SVG export (`svg_save=True`).
  Wrapped in ``temp_style(["ysy_academic", "tab4"])``.
- `plot(...)`: the same drawing logic without any built-in style decoration.
- `temp_style(...)`: dual-mode style manager: usable as a `with` context manager *or*
  as a function decorator. It composes a temporary `.mplstyle` file from selected
  presets and/or extra rcParams, applies it within scope, then restores Matplotlib
  defaults on exit.
- `print_preset_styles()`: quick guide to available layout and color theme presets.
- `PRESET_STYLES`: a dictionary of small, focused `.mplstyle` fragments. You can mix
  “layout” presets (sizes, ticks, fonts, legends, etc.) with “color” presets.

Usage
-----
Quick start for plotting:
    >>> from precision_physkit.plotting import mamplot
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 200)
    >>> y = np.sin(x)
    >>> mamplot(x, y, legend_name='sin(x)', plot_title='Sine', x_label='x', y_label='y')

Using temporary styles (context manager):
    >>> from precision_physkit.plotting import temp_style, mamplot
    >>> with temp_style(['ysy_academic', 'sky']):
    ...     mamplot(x, y, legend_name='sin(x)', plot_title='Styled Sine')

Using temporary styles (decorator):
    >>> from precision_physkit.plotting import temp_style
    >>> @temp_style(['ysy_academic', 'science_color'])
    ... def draw():
    ...     mamplot(x, y, legend_name='sin(x)')
    ...     return None
    >>> draw()

Composing with overrides:
    >>> extra = "lines.linewidth: 2.5\\naxes.grid: True\\n"
    >>> with temp_style(['science'], extra_style=extra):
    ...     mamplot(x, y, legend_name='sin(x)')

Notes & Limitations
-------------------
- Hex colors in `.mplstyle` usually require a leading '#'. Several presets below
  use bare hex strings (e.g., '89b4fa'). If Matplotlib throws a parsing error on
  your setup, prefix them with '#'.
- The style manager writes a temporary `.mplstyle` file and removes it on exit.
  If your program is interrupted mid-block, the file may linger in temp space.
- `mamplot` is intentionally simple. For complex layouts (subplots, twin axes,
  secondary scales, etc.), call Matplotlib directly and optionally wrap with
  `temp_style(...)`.

License
-------
MIT

Changelog
---------
v1.7.0: Refactor script, remove unnecessary pieces, add IEEE theme, add dual-mode
         style manager, and replace `plot` with `mamplot` (legend_out/svg_save).
v1.8.1: Resumed development of the lightweight module:
    - Removed several visually inconsistent color palettes.
    - Integrated `plot(...)` into this module.
    - Added new color palettes.
    - Refined the primary plotting function.
    - Removed obsolete styles.
v1.9.0:
    - Revised existing plotting styles and added new ones.
    - Added the `get_academic_style` plotting-style generator.
v1.10.0:
    - Added plotting styles.
    - Refined the primary plotting function.
    - v1.10.1 added the `gp` style, using Times New Roman for thesis figures.
v1.11.1: Ported into precision_physkit as ``precision_physkit.plotting``:
    - deduplicated the ~130-line shared drawing body of `mamplot` and `plot`
      into the private `_draw`;
    - fixed: for a single series without ``data_point`` the ``legend_name``
      was silently dropped (no legend was ever drawn);
    - fixed: ``legend_up=True`` with a single series and ``data_point``
      computed ``ncol`` from the data length; it now uses the actual
      number of legend entries;
    - `get_academic_style` and `get_adaptive_subplot_style` gained a
      ``verbose`` flag to silence their stdout reports;
    - `print_preset_styles` guide refreshed (fixed the ``wram_nature``
      typo, listed all current presets, removed stale entries).
"""

from __future__ import annotations

__version__ = "1.11.1"

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
import functools
import os
import re
import tempfile

import matplotlib.pyplot as plt
from matplotlib import style as mplstyle
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "PRESET_STYLES",
    "__version__",
    "get_academic_style",
    "get_adaptive_subplot_style",
    "mamplot",
    "plot",
    "print_preset_styles",
    "temp_style",
]


# =========================
# Style Manager (dual-mode)
# =========================

class _StyleContextOrDecorator:
    """Object usable both as a `with` context manager and a function decorator."""

    def __init__(self, style_keys: Optional[Iterable[str]] = None, extra_style: str = "") -> None:
        self._style_keys = style_keys
        self._extra_style = extra_style
        self._tmp_path: Optional[str] = None

    def __enter__(self):
        combined_style = ""

        if self._style_keys:
            for key in self._style_keys:
                if key in PRESET_STYLES:
                    combined_style += PRESET_STYLES[key] + "\n"
                else:
                    raise ValueError(f"Unknown style key: {key}")

        if self._extra_style:
            if not self._extra_style.endswith("\n"):
                combined_style += self._extra_style + "\n"
            else:
                combined_style += self._extra_style

        # Write the composed style to a temporary .mplstyle file and apply it.
        with tempfile.NamedTemporaryFile("w+", suffix=".mplstyle", delete=False, encoding="utf-8") as tmp:
            tmp.write(combined_style)
            self._tmp_path = tmp.name

        mplstyle.use(self._tmp_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            # Restore Matplotlib defaults (as documented behavior).
            mplstyle.use("default")
        finally:
            if self._tmp_path is not None:
                try:
                    os.remove(self._tmp_path)
                except (FileNotFoundError, PermissionError):
                    pass
                finally:
                    self._tmp_path = None

    def __call__(self, func: Callable):
        """Decorator usage: @_StyleContextOrDecorator(...)."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def temp_style(style_keys: Optional[Iterable[str]] = None, extra_style: str = ""):
    """
    Temporarily apply a composed Matplotlib style (context manager or decorator).

    As a context manager:
        with temp_style(['ysy_academic', 'sky'], extra_style="legend.frameon: False\\n"):
            ...

    As a decorator with parameters:
        @temp_style(['ysy_academic', 'science_color'])
        def draw():
            ...

    As a bare decorator (no parameters, equivalent to empty overrides):
        @temp_style
        def draw():
            ...

    On exit, Matplotlib defaults are restored (not the pre-entry style stack).
    """
    # Bare decorator form: @temp_style
    if callable(style_keys) and extra_style == "":
        func = style_keys  # type: ignore[assignment]
        obj = _StyleContextOrDecorator(None, "")
        return obj(func)  # type: ignore[misc]

    # Normal: return an object that supports both `with` and `@`
    return _StyleContextOrDecorator(style_keys, extra_style)


# =========================
# Shared drawing body
# =========================

def _draw(
    x: Iterable,
    y: Union[Iterable, Sequence[Iterable]],
    legend_name: Union[str, Sequence[str]],
    plot_title: str = "",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    plot_type: str = "curve",  # {'curve', 'scatter'}
    legend_title: str = "",
    data_point: Optional[Tuple[float, float]] = None,
    *,
    legend_out: bool = True,
    legend_up: bool = False,
    legend_fancy: bool = True,
    svg_save: bool = False,
    pdf_save: bool = False,
    fig_show: bool = True,
    x_log: bool = False,
    y_log: bool = False,
    return_fig: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    adjust: bool = False,
    line_alpha: float = 1.0
) -> Optional[Tuple[Figure, Axes]]:
    """Draw a standardized line/scatter figure; see `mamplot` for the full docs."""
    if fig_show:
        return_fig = False

    if return_fig:
        fig, ax = plt.subplots()
    else:
        plt.figure()

    is_multi = isinstance(y, (list, tuple))
    if is_multi:
        if not isinstance(legend_name, (list, tuple)) or len(legend_name) != len(y):  # type: ignore[arg-type]
            raise ValueError(
                "For multiple series, `legend_name` must be a list/tuple of the same length as `y`."
            )
        for yi, name in zip(y, legend_name):  # type: ignore[assignment]
            if plot_type == "curve":
                plt.plot(x, yi, label=name, alpha=line_alpha)
            elif plot_type == "scatter":
                plt.scatter(x, yi, label=name)
            else:
                raise ValueError("plot_type must be 'curve' or 'scatter'.")
    else:
        if plot_type == "curve":
            if data_point is not None:
                plt.plot(x, y, label=legend_name, zorder=3)  # type: ignore[arg-type]
                plt.scatter(
                    data_point[0],
                    data_point[1],
                    label="Data Point",
                    marker="x",
                    zorder=4,
                    color='C1'
                )
            else:
                plt.plot(x, y, label=legend_name, zorder=3)
        elif plot_type == "scatter":
            plt.scatter(x, y, label=legend_name)  # type: ignore[arg-type]
        else:
            raise ValueError("plot_type must be 'curve' or 'scatter'.")

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    # A legend is drawn whenever there is at least one meaningfully labeled
    # artist. In the legacy version the legend (and thus `legend_name`) was
    # silently dropped for a single series without `data_point`.
    single_named = (
        not is_multi and isinstance(legend_name, str) and legend_name != ""
    )
    has_legend = is_multi or (data_point is not None) or single_named

    if legend_up:
        legend_out = False
    if legend_out:
        if has_legend:
            if (is_multi or data_point is not None) and legend_title == '':
                legend_title = 'Legend'
            if legend_fancy:
                plt.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1), shadow=True)
            else:
                plt.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    elif legend_up:
        if has_legend:
            if is_multi:
                n_col = len(y) + 1
            else:
                n_col = 1 + (1 if data_point is not None else 0)
            plt.legend(
                loc='lower center',         # Align by the legend's lower center.
                bbox_to_anchor=(0.5, 1.02), # Center the legend above the axes.
                ncol=n_col,                 # Lay out entries horizontally.
                frameon=False,              # Remove the frame for a minimal style.
            )
            plt.title(plot_title, pad=30)
    else:
        if has_legend:
            if legend_fancy:
                plt.legend(title=legend_title)
            else:
                plt.legend(title=legend_title, frameon=False)

    if adjust:
        plt.tight_layout()

    if svg_save:
        base = plot_title.strip() or "figure"
        base = re.sub(r"[^A-Za-z0-9\-_]+", "_", base)
        fname = f"{base}.svg"
        plt.savefig(fname, format="svg", bbox_inches="tight")

    if pdf_save:
        base = plot_title.strip() or "figure"
        base = re.sub(r"[^A-Za-z0-9\-_]+", "_", base)
        fname = f"{base}.pdf"
        plt.savefig(fname, format="pdf")

    if fig_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        return None


# =========================
# Plot (new: mamplot)
# =========================

@temp_style(['ysy_academic', 'tab4'])
def mamplot(
    x: Iterable,
    y: Union[Iterable, Sequence[Iterable]],
    legend_name: Union[str, Sequence[str]],
    plot_title: str = "",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    plot_type: str = "curve",  # {'curve', 'scatter'}
    legend_title: str = "",
    data_point: Optional[Tuple[float, float]] = None,
    *,
    legend_out: bool = True,
    legend_up: bool = False,
    legend_fancy: bool = True,
    svg_save: bool = False,
    pdf_save: bool = False,
    fig_show: bool = True,
    x_log: bool = False,
    y_log: bool = False,
    return_fig: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    adjust: bool = False,
    line_alpha: float = 1.0
) -> Optional[Tuple[Figure, Axes]]:
    """
    Create a quick standardized plot (line or scatter) with minimal boilerplate.

    The call is wrapped in ``temp_style(['ysy_academic', 'tab4'])``: the figure
    is drawn with the built-in academic layout and the tab4 color cycle. On
    exit Matplotlib's *default* rcParams are restored (not the rcParams that
    were active before the call). Use `plot` for the same drawing logic
    without any style decoration.

    Parameters
    ----------
    x : array-like
        X-axis data.
    y : array-like or list/tuple of array-like
        Y-axis data. If `y` is a list/tuple, each element is plotted as a separate series.
    legend_name : str or list[str]
        Legend label for the series. If `y` is a list/tuple, `legend_name` should be a list
        of equal length providing a label for each series.
    plot_title : str, optional
        Figure title.
    x_label : str, optional
        X-axis label (default 'X Axis').
    y_label : str, optional
        Y-axis label (default 'Y Axis').
    plot_type : {'curve', 'scatter'}, optional
        Plot as continuous lines ('curve') or points ('scatter'). Default 'curve'.
    legend_title : str, optional
        Title for the legend box (empty by default; defaults to ``'Legend'``
        for multi-series or ``data_point`` plots).
    data_point : tuple[float, float] | None, optional
        If provided *and* `y` is a single series, highlight one specific data point as
        an 'x' marker (zorder=4) on top of the line. Example: (x0, y0).

    Keyword-only Parameters
    -----------------------
    legend_out : bool, default True
        If True, place the legend outside at upper-left with
        bbox_to_anchor=(1.02, 1).
    legend_up : bool, default False
        If True, place the legend horizontally above the axes (overrides
        ``legend_out``) and pad the title to make room.
    legend_fancy : bool, default True
        If True, draw the legend with a shadow (outside placement) or a frame
        (inside placement); if False, draw it frameless.
    svg_save : bool, default False
        If True, save the figure as an SVG in the current directory. Filename
        derived from `plot_title` or falls back to 'figure.svg'.
    pdf_save : bool, default False
        If True, save the figure as a PDF in the current directory, with the
        same filename derivation as ``svg_save``.
    fig_show : bool, default True
        If True, display the figure via `plt.show()`. Forces
        ``return_fig=False``.
    x_log, y_log : bool, default False
        Use a logarithmic scale on the respective axis.
    return_fig : bool, default False
        If True (and ``fig_show=False``), return ``(fig, ax)`` instead of
        None so the caller can keep working on the axes.
    x_lim, y_lim : tuple[float, float] | None, optional
        Axis limits.
    adjust : bool, default False
        If True, call `plt.tight_layout()` before saving/showing.
    line_alpha : float, default 1.0
        Line opacity for multi-series 'curve' plots.

    Returns
    -------
    None or tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        None when ``fig_show=True`` or ``return_fig=False``; otherwise the
        ``(fig, ax)`` pair.

    Raises
    ------
    ValueError
        If ``plot_type`` is not 'curve' or 'scatter', or if a multi-series
        ``y`` is not matched by an equally long ``legend_name`` list.
    """
    return _draw(
        x, y, legend_name, plot_title, x_label, y_label, plot_type, legend_title,
        data_point,
        legend_out=legend_out, legend_up=legend_up, legend_fancy=legend_fancy,
        svg_save=svg_save, pdf_save=pdf_save, fig_show=fig_show,
        x_log=x_log, y_log=y_log, return_fig=return_fig,
        x_lim=x_lim, y_lim=y_lim, adjust=adjust, line_alpha=line_alpha,
    )


def plot(
    x: Iterable,
    y: Union[Iterable, Sequence[Iterable]],
    legend_name: Union[str, Sequence[str]],
    plot_title: str = "",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    plot_type: str = "curve",  # {'curve', 'scatter'}
    legend_title: str = "",
    data_point: Optional[Tuple[float, float]] = None,
    *,
    legend_out: bool = True,
    legend_up: bool = False,
    legend_fancy: bool = True,
    svg_save: bool = False,
    pdf_save: bool = False,
    fig_show: bool = True,
    x_log: bool = False,
    y_log: bool = False,
    return_fig: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    adjust: bool = False,
    line_alpha: float = 1.0
) -> Optional[Tuple[Figure, Axes]]:
    """
    Create a quick standardized plot (line or scatter) with no style decoration.

    Identical drawing logic to `mamplot`, but *without* the built-in
    ``temp_style(['ysy_academic', 'tab4'])`` wrapper: the current Matplotlib
    rcParams are used and left untouched.

    Parameters
    ----------
    x : array-like
        X-axis data.
    y : array-like or list/tuple of array-like
        Y-axis data. If `y` is a list/tuple, each element is plotted as a separate series.
    legend_name : str or list[str]
        Legend label for the series. If `y` is a list/tuple, `legend_name` should be a list
        of equal length providing a label for each series.
    plot_title : str, optional
        Figure title.
    x_label : str, optional
        X-axis label (default 'X Axis').
    y_label : str, optional
        Y-axis label (default 'Y Axis').
    plot_type : {'curve', 'scatter'}, optional
        Plot as continuous lines ('curve') or points ('scatter'). Default 'curve'.
    legend_title : str, optional
        Title for the legend box (empty by default; defaults to ``'Legend'``
        for multi-series or ``data_point`` plots).
    data_point : tuple[float, float] | None, optional
        If provided *and* `y` is a single series, highlight one specific data point as
        an 'x' marker (zorder=4) on top of the line. Example: (x0, y0).

    Keyword-only Parameters
    -----------------------
    legend_out : bool, default True
        If True, place the legend outside at upper-left with
        bbox_to_anchor=(1.02, 1).
    legend_up : bool, default False
        If True, place the legend horizontally above the axes (overrides
        ``legend_out``) and pad the title to make room.
    legend_fancy : bool, default True
        If True, draw the legend with a shadow (outside placement) or a frame
        (inside placement); if False, draw it frameless.
    svg_save : bool, default False
        If True, save the figure as an SVG in the current directory. Filename
        derived from `plot_title` or falls back to 'figure.svg'.
    pdf_save : bool, default False
        If True, save the figure as a PDF in the current directory, with the
        same filename derivation as ``svg_save``.
    fig_show : bool, default True
        If True, display the figure via `plt.show()`. Forces
        ``return_fig=False``.
    x_log, y_log : bool, default False
        Use a logarithmic scale on the respective axis.
    return_fig : bool, default False
        If True (and ``fig_show=False``), return ``(fig, ax)`` instead of
        None so the caller can keep working on the axes.
    x_lim, y_lim : tuple[float, float] | None, optional
        Axis limits.
    adjust : bool, default False
        If True, call `plt.tight_layout()` before saving/showing.
    line_alpha : float, default 1.0
        Line opacity for multi-series 'curve' plots.

    Returns
    -------
    None or tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        None when ``fig_show=True`` or ``return_fig=False``; otherwise the
        ``(fig, ax)`` pair.

    Raises
    ------
    ValueError
        If ``plot_type`` is not 'curve' or 'scatter', or if a multi-series
        ``y`` is not matched by an equally long ``legend_name`` list.
    """
    return _draw(
        x, y, legend_name, plot_title, x_label, y_label, plot_type, legend_title,
        data_point,
        legend_out=legend_out, legend_up=legend_up, legend_fancy=legend_fancy,
        svg_save=svg_save, pdf_save=pdf_save, fig_show=fig_show,
        x_log=x_log, y_log=y_log, return_fig=return_fig,
        x_lim=x_lim, y_lim=y_lim, adjust=adjust, line_alpha=line_alpha,
    )


# =========================
# Preset inspector
# =========================

def print_preset_styles() -> None:
    """Print a short guide to recommended style-loading patterns and available presets."""
    print("=== Recommended Loading Format ===")
    print('with temp_style(["ysy_academic", "tab4"]):')
    print()
    print("=== Drawing Layout ===")
    print("ysy_academic, ysy_sans, ieee, sci, science, nature, ysy_tnr (Times New Roman), sci_new (DLS Style), gp (Graduation Project)")
    print()
    print("=== Color Themes ===")
    print(
        """catppuccin_mocha (dark),
catppuccin_latte,
ysy_firefly_1,
science_color (for multicolors),
catppuccin_farppe (dark),
sky,
ysy_firefly_2 (grayscale),
cold_nature,
ieee_color,
warm_nature,
tableau10,
tab10,
matlab,
lancet,
winter_sunny,
morandi,
waiting,
6blue_orange,
6blue_red,
2blue_red,
5blue_red,
4blue_red,
4blue_orange,
tab4,
tab8

All in all, I just recommend some new color themes: (1) tab4 (2) tab8 (3) 5blue_red.
"""
    )


# =========================
# Preset Styles (rcParams fragments)
# =========================
PRESET_STYLES = {
    # -------- Layout presets --------
    "sci": """
figure.figsize           : 3.5, 2.625
figure.dpi               : 600

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : True

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : True

axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 1.0

legend.frameon           : False
savefig.bbox             : tight
savefig.pad_inches       : 0.05

font.family              : serif
mathtext.fontset         : dejavuserif
""",

    "science": """
figure.figsize           : 3.5, 2.625

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : True

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : True

axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 1.0

legend.frameon           : False
savefig.bbox             : tight
savefig.pad_inches       : 0.05

font.family              : serif
mathtext.fontset         : dejavuserif
""",

    "ysy_academic": """
figure.figsize         : 4.7, 2.9
figure.dpi             : 300

axes.labelsize         : 10.5
axes.titlesize         : 11.5
axes.linewidth         : 0.75

xtick.direction        : in
xtick.major.size       : 3
xtick.major.width      : 0.75
xtick.minor.size       : 1.5
xtick.minor.width      : 0.5
xtick.minor.visible    : True
xtick.top              : True

ytick.direction        : in
ytick.major.size       : 3
ytick.major.width      : 0.75
ytick.minor.size       : 1.5
ytick.minor.width      : 0.5
ytick.minor.visible    : True
ytick.right            : True

xtick.labelsize        : 9.5
ytick.labelsize        : 9.5

axes.grid              : False
axes.axisbelow         : True
grid.linestyle         : :
grid.alpha             : 0.75
grid.linewidth         : 0.5

legend.frameon         : True
legend.framealpha      : 1
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : False
legend.fontsize        : 9
legend.title_fontsize  : 9

lines.linewidth        : 1.55
lines.markersize       : 3

# Font settings
font.family            : serif
axes.formatter.use_mathtext : True
mathtext.fontset       : cm
text.usetex            : False
""",

    "ieee": """
# IEEE-like compact layout for small figures.
figure.figsize : 3.3, 2.5
figure.dpi : 600

font.size : 8
font.family : serif
font.serif : Times New Roman

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : True

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : True

axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 1.0

legend.frameon           : False
savefig.bbox             : tight
savefig.pad_inches       : 0.05
""",

    "nature": """
# Matplotlib style for Nature journal figures.
# In general, they advocate for all fonts to be panel labels to be sans serif
# and all font sizes in a figure to be 7 pt and panel labels to be 8 pt bold.

# Figure size
figure.figsize           : 3.3, 2.5

# Font sizes
axes.labelsize           : 7
xtick.labelsize          : 7
ytick.labelsize          : 7
legend.fontsize          : 7
font.size                : 7

# Font Family
font.family: sans-serif
font.sans-serif          : Arial, DejaVu Sans, Helvetica, Lucida Grande, Verdana, Geneva, Lucid, Avant Garde, sans-serif
mathtext.fontset         : dejavusans

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : True

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : True

# Set line widths
axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 1.
lines.markersize         : 3

legend.frameon           : False
""",

    "ysy_sans": """
figure.figsize         : 4.7, 2.9
figure.dpi             : 300

axes.labelsize         : 10.5
axes.titlesize         : 11.5
axes.linewidth         : 0.75

xtick.direction        : in
xtick.major.size       : 3
xtick.major.width      : 0.75
xtick.minor.size       : 1.5
xtick.minor.width      : 0.5
xtick.minor.visible    : True
xtick.top              : True

ytick.direction        : in
ytick.major.size       : 3
ytick.major.width      : 0.75
ytick.minor.size       : 1.5
ytick.minor.width      : 0.5
ytick.minor.visible    : True
ytick.right            : True

xtick.labelsize        : 9.5
ytick.labelsize        : 9.5

axes.grid              : False
axes.axisbelow         : True
grid.linestyle         : :
grid.alpha             : 0.75
grid.linewidth         : 0.5

legend.frameon         : True
legend.framealpha      : 1
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : False
legend.fontsize        : 9
legend.title_fontsize  : 9

lines.linewidth        : 1.55
lines.markersize       : 3

# Font settings
font.family            : sans-serif
axes.formatter.use_mathtext : True
mathtext.fontset       : cm
text.usetex            : False
""",

    "ysy_tnr": """
figure.figsize         : 4.7, 2.9
figure.dpi             : 300

axes.labelsize         : 10.5
axes.titlesize         : 11.5
axes.linewidth         : 0.75

xtick.direction        : in
xtick.major.size       : 3
xtick.major.width      : 0.75
xtick.minor.size       : 1.5
xtick.minor.width      : 0.5
xtick.minor.visible    : True
xtick.top              : True

ytick.direction        : in
ytick.major.size       : 3
ytick.major.width      : 0.75
ytick.minor.size       : 1.5
ytick.minor.width      : 0.5
ytick.minor.visible    : True
ytick.right            : True

xtick.labelsize        : 9.5
ytick.labelsize        : 9.5

axes.grid              : False
axes.axisbelow         : True
grid.linestyle         : :
grid.alpha             : 0.75
grid.linewidth         : 0.5

legend.frameon         : True
legend.framealpha      : 1
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : False
legend.fontsize        : 9
legend.title_fontsize  : 9

lines.linewidth        : 1.55
lines.markersize       : 3

# Font settings
font.family : serif
font.serif : Times New Roman
axes.formatter.use_mathtext : True
mathtext.fontset       : cm
text.usetex            : False
""",

    "sci_new": """
figure.figsize           : 4.7, 3.76
figure.dpi               : 600

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : False
xtick.bottom             : False

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : False
ytick.left               : False

axes.grid                : True
axes.edgecolor           : 0.8
grid.linestyle           : -
grid.color               : 0.8
axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 2.0

legend.frameon          : False
legend.fancybox         : True

font.family              : serif
mathtext.fontset         : dejavuserif

axes.labelsize         : 10.5
axes.titlesize         : 11.5
xtick.labelsize        : 9.5
ytick.labelsize        : 9.5
legend.fontsize        : 9
legend.title_fontsize  : 9
""",

    "gp": """
figure.figsize           : 4.7, 3.525
figure.dpi               : 600

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : False
xtick.bottom             : False

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : False
ytick.left               : False

axes.grid                : True
axes.edgecolor           : 0.75
grid.linestyle           : -
grid.color               : 0.75
axes.linewidth           : 0.85
grid.linewidth           : 0.85
lines.linewidth          : 2.5

legend.frameon          : False
legend.fancybox         : False
legend.framealpha       : 1

font.family              : serif
font.serif               : Times New Roman
mathtext.fontset         : dejavuserif

axes.labelsize         : 12.5
axes.titlesize         : 12.5
xtick.labelsize        : 12.5
ytick.labelsize        : 12.5
legend.fontsize        : 10
legend.title_fontsize  : 10
""",

    # -------- Color/theme presets --------
    "catppuccin_mocha": """
# Catppuccin Mocha color theme (dark UI style)
axes.prop_cycle: cycler('color', ['89b4fa', 'fab387', 'a6e3a1', 'f38ba8', 'cba6f7', 'eba0ac', 'f5c2e7', 'f5e0dc', '94e2d5', 'b4befe'])

# Font color: Text
text.color: cdd6f4
axes.labelcolor: cdd6f4
xtick.labelcolor: cdd6f4
ytick.labelcolor: cdd6f4

# Background color: Base
figure.facecolor: 1e1e2e
axes.facecolor: 1e1e2e
savefig.facecolor: 1e1e2e

# Edge color: Surface 0
axes.edgecolor: 313244
legend.edgecolor: 313244
xtick.color: 313244
ytick.color: 313244
patch.edgecolor: 313244
hatch.color: 313244

# Grid color: Surface 0
grid.color: 313244

# Boxplots
boxplot.flierprops.color: 6c7086
boxplot.flierprops.markerfacecolor: 6c7086
boxplot.flierprops.markeredgecolor: 6c7086
boxplot.boxprops.color: 6c7086
boxplot.whiskerprops.color: 6c7086
boxplot.capprops.color: 6c7086
boxplot.medianprops.color: 6c7086
boxplot.meanprops.color: 6c7086
boxplot.meanprops.markerfacecolor: 6c7086
boxplot.meanprops.markeredgecolor: 6c7086
""",

    "catppuccin_latte": """
# Light variant of Catppuccin.
axes.prop_cycle: cycler('color', ['1e66f5', 'fe640b', '40a02b', 'd20f39', '8839ef', 'e64553', 'ea76cb', 'dc8a78', '179299', '7287fd'])

text.color: 4c4f69
axes.labelcolor: 4c4f69
xtick.labelcolor: 4c4f69
ytick.labelcolor: 4c4f69

figure.facecolor: eff1f5
axes.facecolor: eff1f5
savefig.facecolor: eff1f5

axes.edgecolor: ccd0da
legend.edgecolor: ccd0da
xtick.color: ccd0da
ytick.color: ccd0da
patch.edgecolor: ccd0da
hatch.color: ccd0da

grid.color: ccd0da

# Boxplots
boxplot.flierprops.color: 9ca0b0
boxplot.flierprops.markerfacecolor: 9ca0b0
boxplot.flierprops.markeredgecolor: 9ca0b0
boxplot.boxprops.color: 9ca0b0
boxplot.whiskerprops.color: 9ca0b0
boxplot.capprops.color: 9ca0b0
boxplot.medianprops.color: 9ca0b0
boxplot.meanprops.color: 9ca0b0
boxplot.meanprops.markerfacecolor: 9ca0b0
boxplot.meanprops.markeredgecolor: 9ca0b0
""",

    "ysy_firefly_1": """
axes.prop_cycle : cycler('color', ['475d7b', '97c6c0', 'e26e1b', '4df8e8', '3e324a', '6b8fb4', 'f1b349', 'a081af'])
grid.color: k
""",

    "science_color": """
axes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])
grid.color: k
""",

    "catppuccin_farppe": """
axes.prop_cycle: cycler('color', ['8caaee', 'ef9f76', 'a6d189', 'e78284', 'ca9ee6', 'ea999c', 'f4b8e4', 'f2d5cf', '81c8be', 'babbf1'])

text.color: c6d0f5
axes.labelcolor: c6d0f5
xtick.labelcolor: c6d0f5
ytick.labelcolor: c6d0f5

figure.facecolor: 303446
axes.facecolor: 303446
savefig.facecolor: 303446

axes.edgecolor: 414559
legend.edgecolor: 414559
xtick.color: 414559
ytick.color: 414559
patch.edgecolor: 414559
hatch.color: 414559

grid.color: 414559

# Boxplots
boxplot.flierprops.color: 737994
boxplot.flierprops.markerfacecolor: 737994
boxplot.flierprops.markeredgecolor: 737994
boxplot.boxprops.color: 737994
boxplot.whiskerprops.color: 737994
boxplot.capprops.color: 737994
boxplot.medianprops.color: 737994
boxplot.meanprops.color: 737994
boxplot.meanprops.markerfacecolor: 737994
boxplot.meanprops.markeredgecolor: 737994
""",

    "sky": """
# Sky color theme, based on SkyRelax.
axes.prop_cycle : cycler('color', ['4c55bc', 'ffbe98', 'fc8f9b', 'ad69a2', 'f39477', 'eca4b8', '8c9cc1', 'c8ead4', '63d7fe', '388ef7', 'f7786b', '91dce8', 'd16d7c', '766c9b', '53181f', 'f7b9c2', '555d8b'])
""",

    "ysy_firefly_2": """
# Grayscale-leaning teal sequence for light UIs.
axes.prop_cycle : cycler('color', ['3e5754', '567e79', '6e9f99', '85b6b1', '97c6c0', 'a7d1cb', 'bedfd8', 'd3eae5', 'e8f4f2', 'f5faf9'])
""",

    "cold_nature": """
# Natural dusk palette for light UIs
axes.prop_cycle : cycler('color', ['403990', '80A6E2', 'FBDD85', 'F46F43', 'CF3D3E'])
""",

    "ieee_color": """
# Simple IEEE-like color+linestyle cycler. Pairs with 'ieee' layout.
axes.prop_cycle : (cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.']))
""",

    "warm_nature": """
axes.prop_cycle : cycler('color', ['E64B35', '4DBBD5', '00A087', '3C5488', 'F39B7F', '8491B4'])
""",

    "matlab": """
axes.prop_cycle : cycler('color', ['0072BD', 'D95319', 'EDB120', '7E2F8E', '77AC30'])
""",

    "tableau10": """
axes.prop_cycle : cycler('color', ['4E79A7', 'F28E2B', 'E15759', '76B7B2', '59A14F', 'EDC948', 'B07AA1', 'FF9DA7', '9C755F', 'BAB0AC'])
""",
    "tab10": """
axes.prop_cycle : cycler('color', ['4E79A7', 'F28E2B', 'E15759', '76B7B2', '59A14F', 'EDC948', 'B07AA1', 'FF9DA7', '9C755F', 'BAB0AC'])
""",

    "lancet": """
axes.prop_cycle : cycler('color', ['00468B', 'ED0000', '42B540', '0099B4', '925E9F'])
""",
    "winter_sunny": """
axes.prop_cycle : cycler('color', ['d73221', 'e35235', 'e48070', 'fcb777', 'fde699', 'fef4ae', 'd2edf2', '6491c1', '4573b4'])
""",
    "morandi": """
axes.prop_cycle : cycler('color', ['7da6c6', 'eaaa60', '84c3b7', 'e68b81', 'b7b2d0'])
""",
    "waiting": """
axes.prop_cycle : cycler('color', ['32aeec', 'e9262e', '444a9b', 'e3863c', '712a7d'])
""",
    "6blue_orange": """
axes.prop_cycle : cycler('color', ['006ba4', 'ff800e', '595959', 'a2c8ec', 'ffbc79', 'cfcfcf'])
""",
    "6blue_red": """
axes.prop_cycle : cycler('color', ['2c69b0', 'f02720', '595959', '6ba3d6', 'ea6b73', 'cfcfcf'])
""",
    "2blue_red": """
axes.prop_cycle : cycler('color', ['3951A2', 'DA382A'])
""",
    "5blue_red": """
axes.prop_cycle : cycler('color', ['0000FF', 'FF0000', 'A0A0A0', 'FFCCCC', 'CCCCFF'])
""",
    "4blue_red": """
axes.prop_cycle : cycler('color', ['003fe5', 'fc0000', '70AD47', '6030f0'])
""",
    "4blue_orange": """
axes.prop_cycle : cycler('color', ['1F77B4', 'FF7F0E', 'AEC7E8', 'FFBB78'])
""",
    "tab4": """
axes.prop_cycle : cycler('color', ['4472C4', 'ED7D31', '70AD47', 'A54B98'])
""",
    "tab8": """
axes.prop_cycle : cycler('color', ['377eb8', 'ff7f00', '4daf4a', '984ea3', 'e41a1c', 'ffff33', 'a65628', 'f781bf'])
"""
}


def get_academic_style(
    rows: int = 1,
    cols: int = 1,
    base_width_mm: float = 150,
    use_surplus: float = 1.0,
    ratio: float = 0.75,
    verbose: bool = True,
) -> str:
    """
    Compute strict academic-figure style parameters for a subplot grid.

    Parameters
    ----------
    rows : int, default 1
        Number of subplot rows (m).
    cols : int, default 1
        Number of subplot columns (n).
    base_width_mm : float, default 150
        Full-width baseline of the figure in millimetres (typical journal
        double-column width is 150-160 mm).
    use_surplus : float, default 1.0
        Size multiplier applied to the whole figure, e.g. 1.1 for a 10%
        surplus to accommodate overflowing labels (1.0 = no surplus).
    ratio : float, default 0.75
        Per-panel height/width ratio (strict 4:3 by default).
    verbose : bool, default True
        If True, print the dimension/settings report to stdout. Set to
        False to use the function silently.

    Returns
    -------
    str
        A Matplotlib style string that can be passed to
        `matplotlib.style.use` (via a temporary file), to
        ``temp_style(extra_style=...)``, or written to a `.mplstyle` file.
    """

    # -----------------------------------------------------------
    # 1. Physical dimensions
    # -----------------------------------------------------------
    scale_factor = use_surplus

    # Geometry of one subplot.
    sub_w_mm = base_width_mm / cols
    sub_h_mm = sub_w_mm * ratio # Preserve the requested aspect ratio.

    # Unscaled geometry of the complete figure.
    total_w_mm = base_width_mm
    total_h_mm = sub_h_mm * rows

    # Apply the margin scale.
    final_w_mm = total_w_mm * scale_factor
    final_h_mm = total_h_mm * scale_factor

    # Unit-conversion constants.
    MM_TO_INCH = 1 / 25.4
    INCH_TO_PT = 72
    MM_TO_PT = MM_TO_INCH * INCH_TO_PT

    # -----------------------------------------------------------
    # 2. Adaptive visual elements
    # Denser grids use thinner lines and tighter text, subject to minimum sizes.
    # -----------------------------------------------------------
    if cols == 1:
        font_base = 10.5
        line_base = 1.5
        marker_base = 6
        tick_len_maj = 4.0
    elif cols == 2:
        font_base = 9.5
        line_base = 1.25
        marker_base = 5
        tick_len_maj = 3.5
    else: # cols >= 3
        font_base = 8.5  # Keep a small margin above the 8-point minimum.
        line_base = 1.0
        marker_base = 4
        tick_len_maj = 3.0

    # Tick labels are one point smaller than body text, but never below 8 points.
    tick_font = max(8.0, font_base - 1.0)

    # -----------------------------------------------------------
    # 3. Diagnostic report
    # -----------------------------------------------------------
    if verbose:
        print("[Report]")
        print("="*60)
        print(f"Layout: {rows} Rows x {cols} Cols | Surplus: {use_surplus} | Ratio: {ratio}")
        print("-" * 60)
        print(f"{'Dimension':<15} | {'mm':<10} | {'inch':<10} | {'pt':<10}")
        print("-" * 60)

        # Dimension data.
        dims = [
            ("Subplot W", sub_w_mm),
            ("Subplot H", sub_h_mm),
            ("Total W", final_w_mm),
            ("Total H", final_h_mm),
        ]

        for name, mm_val in dims:
            inch_val = mm_val * MM_TO_INCH
            pt_val = mm_val * MM_TO_PT
            print(f"{name:<15} | {mm_val:<10.2f} | {inch_val:<10.4f} | {pt_val:<10.1f}")

        print("-" * 60)
        print("Adaptive Settings:")
        print(f"  > Font Size (Base): {font_base} pt")
        print(f"  > Font Size (Tick): {tick_font} pt")
        print(f"  > Line Width:       {line_base} pt")
        print("="*60)
        print("\n")

    # -----------------------------------------------------------
    # 4. Style-string generation
    # -----------------------------------------------------------
    # Convert the computed values into Matplotlib-compatible key-value entries.
    # Use a triple-quoted string to keep the generated style readable.

    w_inch = final_w_mm * MM_TO_INCH
    h_inch = final_h_mm * MM_TO_INCH

    style_content = f"""
# Generated by ysy_academic_auto
# Layout: {rows}x{cols}, Width: {final_w_mm:.1f}mm, Height: {final_h_mm:.1f}mm

figure.figsize         : {w_inch:.4f}, {h_inch:.4f}
figure.dpi             : 300

# Font Settings
font.family            : serif
font.size              : {font_base}
text.usetex            : False
mathtext.fontset       : cm

# Axes & Titles
axes.labelsize         : {font_base}
axes.titlesize         : {font_base}
axes.linewidth         : {0.8 if cols < 3 else 0.6}
axes.grid              : False
axes.axisbelow         : True

# Ticks
xtick.labelsize        : {tick_font}
ytick.labelsize        : {tick_font}
xtick.direction        : in
ytick.direction        : in
xtick.top              : True
ytick.right            : True
xtick.major.size       : {tick_len_maj}
xtick.major.width      : {0.8 if cols < 3 else 0.6}
ytick.major.size       : {tick_len_maj}
ytick.major.width      : {0.8 if cols < 3 else 0.6}
xtick.minor.visible    : True
ytick.minor.visible    : True
xtick.minor.size       : {tick_len_maj * 0.5}
xtick.minor.width      : {0.6 if cols < 3 else 0.5}
ytick.minor.size       : {tick_len_maj * 0.5}
ytick.minor.width      : {0.6 if cols < 3 else 0.5}

# Legend
legend.frameon         : False
legend.fontsize        : {tick_font}
legend.title_fontsize  : {tick_font}
legend.numpoints       : 1

# Lines & Markers
lines.linewidth        : {line_base}
lines.markersize       : {marker_base}
lines.markeredgewidth  : 0
grid.linestyle         : :
grid.linewidth         : 0.5
grid.alpha             : 0.75
savefig.bbox           : tight
savefig.pad_inches     : 0.05
"""
    return style_content


def get_adaptive_subplot_style(
    rows: int = 1,
    cols: int = 1,
    width: float = 4.7,
    ratio: float = 0.75,
    font_family: str = "sans-serif",
    font_base_size: float = 14.0,
    verbose: bool = True,
) -> str:
    """
    Generate an adaptive multi-subplot style string for `temp_style(...)`.

    This helper keeps the single-panel style fixed, and scales multi-panel
    typography/lines/spines from the available panel width with lower bounds.

    Parameters
    ----------
    rows : int, default 1
        Number of subplot rows.
    cols : int, default 1
        Number of subplot columns.
    width : float, default 4.7
        Total figure width in inches for the whole canvas. Ignored for 1x1.
    ratio : float, default 0.75
        Per-panel height/width ratio for multi-panel layouts.
    font_family : {'sans-serif', 'serif'}, default 'sans-serif'
        Global font family.
    font_base_size : float, default 14.0
        Base font size in points before adaptive scaling.
    verbose : bool, default True
        If True, print the computed settings report to stdout. Set to
        False to use the function silently.

    Returns
    -------
    str
        A Matplotlib style string that can be passed into
        `temp_style(extra_style=...)`.

    Raises
    ------
    ValueError
        If ``rows``/``cols`` are not positive, ``width`` or ``ratio`` are
        not positive, or ``font_family`` is unknown.
    """
    if rows < 1 or cols < 1:
        raise ValueError("`rows` and `cols` must be positive integers.")
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if ratio <= 0:
        raise ValueError("`ratio` must be positive.")
    if font_family not in {"sans-serif", "serif"}:
        raise ValueError("`font_family` must be either 'sans-serif' or 'serif'.")

    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    single_width = 4.7
    single_height = 3.525
    single_ratio = 0.75

    # Fixed single-panel template.
    if rows == 1 and cols == 1:
        fig_w = single_width
        fig_h = single_height
        panel_w = single_width
        panel_h = single_height

        font_base = font_base_size
        title_size = font_base
        axes_width = 1.75
        tick_width = 1.5
        tick_size = 3.0
        grid_width = 1.25
        line_width = 2.5
        marker_size = 8.0
        scale = 1.0
    else:
        fig_w = width
        panel_w = fig_w / cols
        panel_h = panel_w * ratio
        fig_h = panel_h * rows

        # Main driver: available width per subplot.
        # Secondary driver: row density. Width/ratio only adjust gently.
        panel_scale = panel_w / single_width
        density_scale = (
            panel_scale ** 0.42
            * (rows ** -0.18)
            * ((ratio / single_ratio) ** 0.12)
        )
        scale = _clamp(density_scale, 0.42, 1.15)

        font_base = _clamp(font_base_size * (scale ** 0.92), 9.5, 15.0)
        title_size = font_base

        axes_width = _clamp(1.75 * (scale ** 0.80), 0.95, 1.5)
        grid_width = _clamp(1.25 * (scale ** 0.55), 0.38, 0.5)
        line_width = _clamp(2.5 * (scale ** 0.78), 1.55, 2.5)
        line_width = max(line_width, axes_width + 0.35)
        marker_size = _clamp(8.0 * (scale ** 0.68), 4.9, 8.0)
        marker_size = max(marker_size, line_width + 1.9)

        tick_size = _clamp(3.0 * (scale ** 0.35), 2.35, 3.0)
        tick_width = _clamp(1.5 * (scale ** 0.80), 0.95, 1.5)

    legend_font = font_base
    mathtext_font = "cm" if font_family == "serif" else "dejavusans"
    font_family_block = (
        "font.family            : serif\n"
        "font.serif             : Times New Roman, STIXGeneral, DejaVu Serif"
        if font_family == "serif"
        else "font.family            : sans-serif\n"
             "font.sans-serif        : Arial, Helvetica, DejaVu Sans"
    )

    if verbose:
        print("[Adaptive Subplot Style Report]")
        print("=" * 68)
        print(
            f"Layout: {rows} Rows x {cols} Cols | "
            f"Font: {font_family} | Total Width: {fig_w:.3f} in | Ratio: {ratio:.3f}"
        )
        print("-" * 68)
        print(f"{'Item':<18} | {'Value':<14}")
        print("-" * 68)
        print(f"{'Total Width':<18} | {fig_w:.4f} in")
        print(f"{'Figure Height':<18} | {fig_h:.4f} in")
        print(f"{'Panel Width':<18} | {panel_w:.4f} in")
        print(f"{'Panel Height':<18} | {panel_h:.4f} in")
        print(f"{'Adaptive Scale':<18} | {scale:.4f}")
        print(f"{'Base Font':<18} | {font_base:.2f} pt")
        print(f"{'Title Font':<18} | {title_size:.2f} pt")
        print(f"{'Axes Width':<18} | {axes_width:.2f} pt")
        print(f"{'Tick Size':<18} | {tick_size:.2f} pt")
        print(f"{'Tick Width':<18} | {tick_width:.2f} pt")
        print(f"{'Line Width':<18} | {line_width:.2f} pt")
        print(f"{'Marker Size':<18} | {marker_size:.2f} pt")
        print(f"{'Grid Width':<18} | {grid_width:.2f} pt")
        print("=" * 68)
        print()

    style_content = f"""
# Generated by get_adaptive_subplot_style
# Layout: {rows}x{cols} | Figure: {fig_w:.4f} x {fig_h:.4f} in

figure.figsize         : {fig_w:.4f}, {fig_h:.4f}
figure.dpi             : 600

{font_family_block}
font.size              : {font_base:.2f}
mathtext.fontset       : {mathtext_font}
text.usetex            : False

axes.labelsize         : {font_base:.2f}
axes.titlesize         : {title_size:.2f}
axes.linewidth         : {axes_width:.2f}
axes.grid              : True
axes.axisbelow         : True

xtick.direction        : in
xtick.major.size       : {tick_size:.2f}
xtick.major.width      : {tick_width:.2f}
xtick.minor.visible    : False
xtick.top              : True
xtick.labelsize        : {font_base:.2f}

ytick.direction        : in
ytick.major.size       : {tick_size:.2f}
ytick.major.width      : {tick_width:.2f}
ytick.minor.visible    : False
ytick.right            : True
ytick.labelsize        : {font_base:.2f}

grid.color             : 0.85
grid.linestyle         : -
grid.linewidth         : {grid_width:.2f}

legend.frameon         : False
legend.fontsize        : {legend_font:.2f}
legend.title_fontsize  : {legend_font:.2f}
legend.handlelength    : 1.50
legend.handletextpad   : 0.40

lines.linewidth        : {line_width:.2f}
lines.markersize       : {marker_size:.2f}

savefig.bbox           : tight
savefig.pad_inches     : 0.05
"""
    return style_content
