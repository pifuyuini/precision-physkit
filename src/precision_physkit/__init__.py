"""Precision measurement and engineering time-series processing tools.

The high-performance Rust core is exposed as ``precision_physkit._core``.
Submodules import ``_core`` lazily so pure-Python functionality remains
available when the extension has not been built.
"""

__version__ = "0.1.0"

from . import filters, fitting, meta, optimize, peaks, plotting, preprocess, spectral

__all__ = [
    "__version__",
    "filters",
    "fitting",
    "meta",
    "optimize",
    "peaks",
    "preprocess",
    "spectral",
    "plotting",
]
