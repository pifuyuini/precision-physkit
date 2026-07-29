# `precision_physkit.optimize` API

```python
from precision_physkit import optimize
```

## `OptimizeResult`

```python
@dataclass
class OptimizeResult:
    x: NDArray
    fun: float
    success: bool
    message: str
    n_eval: int
    history: dict | None
```

Non-convergence is represented by `success` and `message`, not normally raised.
For least squares, `fun` is `0.5*sum(residual**2)`. `history` is currently `None`.

## Local minimization

```python
optimize.minimize(
    func, x0, method="Nelder-Mead", bounds=None, jac=None, **kw,
)
```

Wraps `scipy.optimize.minimize`. Nelder-Mead is derivative-free but local; use a
method compatible with `jac` and bounds when supplying them.

## Global minimization

```python
optimize.global_minimize(
    func, bounds, method="differential_evolution", seed=None, **kw,
)
```

Runs differential evolution on finite box bounds. It is stochastic: pass a fixed
seed in pipelines and record it in metadata. `success=True` means the algorithm
terminated normally, not that global optimality was proven.

## Nonlinear least squares

```python
optimize.least_squares(
    residual, x0, bounds=(-np.inf, np.inf), **kw,
)
```

Wraps `scipy.optimize.least_squares` and minimizes half the squared residual norm.
Use this when residual structure and physical bounds matter. Use
`fitting.curve_fit` instead when named parameters and covariance are required.

## Multimodal pattern

```python
global_result = optimize.global_minimize(
    objective, bounds=[(-5, 5), (-5, 5)], seed=42,
)
if not global_result.success:
    raise RuntimeError(global_result.message)

local_result = optimize.minimize(
    objective, global_result.x, method="Nelder-Mead",
)
```

Check finite objective values, bound ordering, `success`, and sensitivity to seed
or initial conditions before reporting a result.

## See also

[Fitting](api-fitting.md) · [Pipeline](api-pipeline.md)
