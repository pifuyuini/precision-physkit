# `precision_physkit.fitting` API

```python
from precision_physkit import fitting
```

## `FitResult`

```python
@dataclass
class FitResult:
    params: dict[str, float]
    perr: dict[str, float]
    cov: NDArray
    resid: NDArray
    success: bool
    message: str
    stats: dict[str, Any]
```

`perr` contains one-standard-deviation uncertainties from the covariance diagonal.
Statistics include applicable values such as chi-square, reduced chi-square, R²,
rank, singular values, or iteration weights.

## Nonlinear curve fitting

```python
fitting.curve_fit(
    model, x, y, p0=None, sigma=None,
    bounds=(-np.inf, np.inf), param_names=None,
    raise_on_failure=False, **kw,
)
```

Wraps SciPy nonlinear least squares and returns `FitResult`. Parameter names are
inferred from the model when possible or supplied explicitly. With known absolute
measurement uncertainties, pass `sigma` and `absolute_sigma=True`; otherwise
covariance is scaled by residual reduced chi-square. By default failures return
`success=False`; `raise_on_failure=True` propagates them.

## Linear least squares

```python
fitting.linear_lstsq(design, y, rcond=None, param_names=None)
```

Uses truncated SVD in `precision_physkit._core`. Singular values at or below
`rcond*s_max` are discarded. A rank-deficient system can still return
`success=True` because the retained subspace has a valid least-squares solution;
inspect `stats["rank"]`.

Ill-conditioned designs can predict well while parameter uncertainty is enormous.
Center and scale variables before relying on aggressive SVD truncation.

## Iterative multichannel fitting

```python
fitting.iterative_multichannel(
    design, y, group_sizes, max_iter=100, tol=1e-12, param_names=None,
)
```

Fits parameters shared by contiguous channel row groups whose noise variances are
unknown and unequal. It alternates residual variance estimation with inverse-
variance weighted SVD fitting. Group sizes must be positive and sum to the number
of rows. Reported covariance uses normalized weights; rescale it before treating
`perr` as absolute uncertainty when the common noise scale is known.

## Polynomial fitting

```python
fitting.polyfit(x, y, deg)
```

Builds an ascending-power Vandermonde matrix and calls `linear_lstsq`.
Coefficient order is `c0, c1, ..., cdeg`, the reverse of `numpy.polyfit`.

## Example

```python
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)

result = fitting.curve_fit(
    gaussian, x, y, p0=[1.0, 0.0, 1.0],
    sigma=y_sigma, absolute_sigma=True,
)
if not result.success:
    raise RuntimeError(result.message)
print(result.params, result.perr, result.stats["redchi"])
```

## See also

[Optimization](api-optimize.md) · [Peak analysis](api-peaks.md)
