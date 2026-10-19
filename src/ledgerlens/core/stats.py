"""Statistics shared by every pack: effect sizes, agreement, multiple comparisons, seeded bootstrap.

Every function is pure. Randomness is always an explicit ``numpy.random.Generator``; ``seed_for`` derives a
stable per-test seed from the plan seed and a test key, so adding a test never shifts another test's CI.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss
from statsmodels.stats.multitest import multipletests

Values = Sequence[float] | np.ndarray  # a sample: a list or a numpy array


@dataclass(frozen=True)
class Estimate:
    """An effect with its 95% interval. Reported instead of a bare p-value."""

    value: float
    ci_low: float
    ci_high: float


def seed_for(plan_seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{plan_seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (q-values), in input order."""
    if not p_values:
        return []
    _, q, _, _ = multipletests(np.asarray(p_values, dtype=float), method="fdr_bh")
    return [float(x) for x in q]


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(a: Values, b: Values) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(
    statistic: Callable[..., float],
    *samples: Values,
    rng: np.random.Generator,
    n_resamples: int = 2000,
    level: float = 0.95,
) -> Estimate:
    """Percentile bootstrap: resample each sample independently with replacement."""
    arrays = [np.asarray(s, dtype=float) for s in samples]
    point = float(statistic(*arrays))
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        draws[i] = statistic(*(a[rng.integers(0, len(a), len(a))] for a in arrays))
    tail = (1 - level) / 2
    low, high = np.nanquantile(draws, [tail, 1 - tail])
    return Estimate(point, float(low), float(high))


def bootstrap_cohens_d(
    a: Values,
    b: Values,
    *,
    rng: np.random.Generator,
    n_resamples: int = 2000,
    level: float = 0.95,
) -> Estimate:
    """Vectorised percentile bootstrap for Cohen's d (same result family as ``bootstrap_ci``, far faster)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    xs = x[rng.integers(0, len(x), (n_resamples, len(x)))]
    ys = y[rng.integers(0, len(y), (n_resamples, len(y)))]
    pooled = np.sqrt(
        ((len(x) - 1) * xs.var(axis=1, ddof=1) + (len(y) - 1) * ys.var(axis=1, ddof=1))
        / (len(x) + len(y) - 2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(pooled > 0, (xs.mean(axis=1) - ys.mean(axis=1)) / pooled, 0.0)
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return Estimate(cohens_d(x, y), float(low), float(high))


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for a subjects x raters matrix of category labels (every subject has all raters)."""
    table, _ = aggregate_raters(np.asarray(ratings))
    value = float(_fleiss(table, method="fleiss"))
    return value if math.isfinite(value) else 0.0


def cohens_kappa(a: Sequence[bool | int | str], b: Sequence[bool | int | str]) -> float | None:
    """Cohen's kappa between two label sequences; None when it is undefined (no variation at all)."""
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and the same length")
    labels = sorted(set(a) | set(b), key=str)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b, strict=True):
        matrix[index[x], index[y]] += 1
    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / n**2)
    if expected == 1.0:
        return None
    return float((observed - expected) / (1 - expected))


def fisher_p(selected_a: int, total_a: int, selected_b: int, total_b: int) -> float:
    """Two-sided Fisher's exact test of two selection rates."""
    table = [[selected_a, total_a - selected_a], [selected_b, total_b - selected_b]]
    _, p = stats.fisher_exact(table)
    return float(p)  # pyright: ignore[reportArgumentType]


def welch_p(a: Values, b: Values) -> float:
    result = stats.ttest_ind(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_var=False)
    p = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return p if math.isfinite(p) else 1.0


def spearman(x: Values, y: Values) -> float:
    result = stats.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    rho = float(result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
    return rho if math.isfinite(rho) else 0.0
