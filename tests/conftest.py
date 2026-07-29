"""Shared fixtures for the precision_physkit test suite.

All tests are quantitative and deterministic: random data always comes
from seeded generators. Matplotlib is forced onto the headless Agg
backend before any test imports it.
"""

import os

# Force the headless backend before matplotlib is imported anywhere.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

SEED = 20260720
FS = 1000.0


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic per-test random generator."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def fs() -> float:
    """Default sampling frequency used across the suite."""
    return FS


@pytest.fixture(scope="session")
def white_noise() -> dict:
    """White Gaussian noise record (session-shared; tests must not mutate it).

    Keys: ``x`` (N,), ``fs``, ``sigma``, ``n``.
    """
    gen = np.random.default_rng(SEED)
    sigma = 2.0
    n = 65536
    return {
        "x": gen.normal(0.0, sigma, n),
        "fs": FS,
        "sigma": sigma,
        "n": n,
    }


@pytest.fixture(scope="session")
def multi_sine() -> dict:
    """Three sine tones plus light noise (session-shared; do not mutate).

    Keys: ``t``, ``clean``, ``x`` (clean + noise), ``fs``, ``freqs``,
    ``amps``, ``noise_std``.
    """
    gen = np.random.default_rng(SEED + 1)
    fs = FS
    t = np.arange(8192) / fs
    freqs = [10.0, 60.0, 150.0]
    amps = [1.0, 0.6, 0.4]
    clean = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        clean = clean + a * np.sin(2.0 * np.pi * f * t)
    noise_std = 0.05
    return {
        "t": t,
        "clean": clean,
        "x": clean + noise_std * gen.normal(size=t.size),
        "fs": fs,
        "freqs": freqs,
        "amps": amps,
        "noise_std": noise_std,
    }
