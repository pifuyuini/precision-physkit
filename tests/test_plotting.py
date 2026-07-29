"""Tests for precision_physkit.plotting (headless; Agg backend forced in conftest)."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from precision_physkit import plotting


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_mamplot_single_series_returns_fig():
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    fig, ax = plotting.mamplot(
        x, np.sin(x), "sin(x)", plot_title="Sine", x_label="x", y_label="y",
        fig_show=False, return_fig=True,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) == 1
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    # A named single series must produce a legend entry.
    assert ax.get_legend() is not None


def test_mamplot_multi_series():
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    fig, ax = plotting.mamplot(
        x, [np.sin(x), np.cos(x)], ["sin", "cos"],
        fig_show=False, return_fig=True,
    )
    assert isinstance(fig, Figure)
    assert len(ax.get_lines()) == 2
    assert ax.get_legend() is not None
    # Mismatched legend list is rejected.
    with pytest.raises(ValueError):
        plotting.mamplot(x, [np.sin(x), np.cos(x)], ["only-one"],
                         fig_show=False, return_fig=True)


def test_mamplot_log_axes():
    x = np.linspace(0.1, 100.0, 200)
    fig, ax = plotting.mamplot(
        x, x**0.5, "sqrt", x_log=True, y_log=True, fig_show=False, return_fig=True
    )
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_mamplot_scatter_and_bad_plot_type():
    x = np.linspace(0.0, 1.0, 50)
    fig, ax = plotting.mamplot(
        x, x**2, "quad", plot_type="scatter", fig_show=False, return_fig=True
    )
    assert isinstance(fig, Figure)
    assert len(ax.collections) == 1
    with pytest.raises(ValueError):
        plotting.mamplot(x, x**2, "quad", plot_type="histogram",
                         fig_show=False, return_fig=True)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_mamplot_fig_show_forces_no_return():
    x = np.linspace(0.0, 1.0, 10)
    # fig_show=True must not return a figure (documented behaviour).
    out = plotting.mamplot(x, x, "line", fig_show=True, return_fig=True)
    assert out is None


def test_plot_unstyled_variant():
    x = np.linspace(0.0, 1.0, 50)
    fig, ax = plotting.plot(x, x**3, "cube", fig_show=False, return_fig=True)
    assert isinstance(fig, Figure)
    assert len(ax.get_lines()) == 1


def test_temp_style_context_manager():
    default_lw = matplotlib.rcParamsDefault["lines.linewidth"]
    with plotting.temp_style(extra_style="lines.linewidth: 7.5\n"):
        assert matplotlib.rcParams["lines.linewidth"] == 7.5
    # On exit the Matplotlib *default* style is restored.
    assert matplotlib.rcParams["lines.linewidth"] == default_lw


def test_temp_style_decorator():
    captured = {}

    @plotting.temp_style(extra_style="lines.linewidth: 6.5\n")
    def draw():
        captured["lw"] = matplotlib.rcParams["lines.linewidth"]
        return 42

    assert draw() == 42
    assert captured["lw"] == 6.5
    assert matplotlib.rcParams["lines.linewidth"] != 6.5


def test_temp_style_preset_and_unknown_key():
    with plotting.temp_style(["ysy_academic", "tab4"]):
        pass  # known preset keys apply without error
    with pytest.raises(ValueError):
        with plotting.temp_style(["no-such-preset"]):
            pass


def test_get_adaptive_subplot_style_silent(capsys):
    style = plotting.get_adaptive_subplot_style(2, 3, verbose=False)
    assert isinstance(style, str)
    assert "figure.figsize" in style
    assert "font.size" in style
    assert capsys.readouterr().out == ""  # verbose=False prints nothing
    # The generated style is actually usable by temp_style.
    with plotting.temp_style(extra_style=style):
        pass


def test_get_adaptive_subplot_style_verbose_prints(capsys):
    plotting.get_adaptive_subplot_style(1, 1, verbose=True)
    out = capsys.readouterr().out
    assert "Adaptive Subplot Style Report" in out


def test_get_adaptive_subplot_style_rejects_bad_input():
    with pytest.raises(ValueError):
        plotting.get_adaptive_subplot_style(0, 2, verbose=False)
    with pytest.raises(ValueError):
        plotting.get_adaptive_subplot_style(1, 1, width=-1.0, verbose=False)
    with pytest.raises(ValueError):
        plotting.get_adaptive_subplot_style(1, 1, ratio=0.0, verbose=False)
    with pytest.raises(ValueError):
        plotting.get_adaptive_subplot_style(1, 1, font_family="monospace", verbose=False)


def test_preset_styles_count_and_keys():
    presets = plotting.PRESET_STYLES
    assert len(presets) == 34
    for key, value in presets.items():
        assert isinstance(key, str) and key
        assert isinstance(value, str) and value.strip()
    for expected in ("ysy_academic", "tab4", "science", "nature", "ieee"):
        assert expected in presets


def test_print_preset_styles(capsys):
    plotting.print_preset_styles()
    out = capsys.readouterr().out
    assert "ysy_academic" in out
