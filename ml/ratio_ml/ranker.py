"""The expected-return ranker, and the exploration that feeds it.

This is the model that should go live first, before anything with a
transformer in it. The baseline it replaces is a median over four bucketed
features (``ratio_manager::policy::estimate``); a model that reads the same
features unbucketed and fits a real regression beats that almost by
construction, is interpretable, and works at the data volume that exists now.

Two pieces:

:class:`Ranker`
    Predicts the return of an add, per ``(release, box)``. Bayesian ridge over
    the feature vector, which gives a **posterior**, not just a point
    estimate. The posterior is the part that matters: it is what the bandit
    samples from, and what tells you the model has never seen a cell before.

:class:`ThompsonBandit`
    Chooses a box by sampling each one's return from the posterior and taking
    the best draw. This buys exploration at a known price and, crucially,
    **logs the probability of the action it took**, which every offline method
    downstream needs and which the current baseline does not record.

Deliberately numpy only. A model that cannot be retrained on the services VM
without a GPU is a model that will not get retrained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Example, NAMES


@dataclass
class Fit:
    """What a fit produced, and how well it did."""

    weights: np.ndarray
    covariance: np.ndarray
    noise_var: float
    n: int

    def feature_table(self) -> list[tuple[str, float]]:
        """Weights by name, largest magnitude first. The reason to prefer a
        linear model while the data is thin: this can be read."""
        pairs = list(zip(NAMES, self.weights[1:]))
        return sorted(pairs, key=lambda p: -abs(p[1]))


class Ranker:
    """Bayesian ridge regression on the shared feature vector.

    The prior precision ``alpha`` is the regularizer; ``beta`` is the
    assumed noise precision. Both are set rather than learned, because with a
    few hundred races an evidence maximization just overfits differently.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.fit_: Fit | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None

    # -- internals ---------------------------------------------------------

    def _design(self, xs: np.ndarray) -> np.ndarray:
        """Standardize and prepend an intercept column."""
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("the ranker has not been fitted")
        z = (xs - self.x_mean_) / self.x_std_
        return np.hstack([np.ones((z.shape[0], 1)), z])

    # -- api ---------------------------------------------------------------

    def fit(self, rows: list[Example]) -> Fit:
        if not rows:
            raise ValueError("no examples to fit — check the horizon and the join")
        xs = np.asarray([r.x for r in rows], dtype=float)
        ys = np.asarray([r.y for r in rows], dtype=float)
        self.x_mean_ = xs.mean(axis=0)
        std = xs.std(axis=0)
        # A feature that never varies carries no information; keep its scale at
        # 1 so it contributes zero rather than a division by zero.
        std[std < 1e-12] = 1.0
        self.x_std_ = std

        phi = self._design(xs)
        d = phi.shape[1]
        precision = self.alpha * np.eye(d) + self.beta * phi.T @ phi
        cov = np.linalg.inv(precision)
        weights = self.beta * cov @ phi.T @ ys
        resid = ys - phi @ weights
        dof = max(len(ys) - d, 1)
        noise = float(resid @ resid / dof)
        self.fit_ = Fit(weights=weights, covariance=cov, noise_var=noise, n=len(ys))
        return self.fit_

    def predict(self, xs: list[list[float]]) -> np.ndarray:
        """Posterior mean return for each row."""
        if self.fit_ is None:
            raise RuntimeError("the ranker has not been fitted")
        phi = self._design(np.asarray(xs, dtype=float))
        return phi @ self.fit_.weights

    def predict_with_sigma(self, xs: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation.

        The standard deviation grows where the data is thin, which is exactly
        where a policy should be either curious or careful.
        """
        if self.fit_ is None:
            raise RuntimeError("the ranker has not been fitted")
        phi = self._design(np.asarray(xs, dtype=float))
        mean = phi @ self.fit_.weights
        var = np.einsum("ij,jk,ik->i", phi, self.fit_.covariance, phi)
        return mean, np.sqrt(np.maximum(var, 0.0) + self.fit_.noise_var)

    def sample(self, xs: list[list[float]], rng: np.random.Generator) -> np.ndarray:
        """One draw of the return per row, from the posterior over weights.

        Sampling the *weights* once per decision rather than each row
        independently is what makes this Thompson sampling rather than noise:
        the boxes are then compared under one coherent hypothesis.
        """
        if self.fit_ is None:
            raise RuntimeError("the ranker has not been fitted")
        w = rng.multivariate_normal(self.fit_.weights, self.fit_.covariance)
        phi = self._design(np.asarray(xs, dtype=float))
        return phi @ w


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. The ranker's job is to order boxes, so rank
    correlation says more about it than squared error does."""
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


class BucketedMedian:
    """The live baseline, reimplemented so it can be scored on the same split.

    A new model's only meaningful claim is that it beats this. Keeping the
    comparison in the same file as the model makes it hard to quietly skip.
    """

    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, str], list[float]] = {}
        self.default = 0.0

    @staticmethod
    def size_bucket(gb: float) -> str:
        for edge, name in ((5, "xs"), (20, "s"), (60, "m"), (150, "l")):
            if gb < edge:
                return name
        return "xl"

    @staticmethod
    def demand_bucket(d: float) -> str:
        for edge, name in ((0.5, "none"), (2.0, "low"), (8.0, "mid")):
            if d < edge:
                return name
        return "high"

    def fit(self, rows: list[Example]) -> "BucketedMedian":
        from .features import NAMES as N

        i_size, i_demand = N.index("log_size_gb"), N.index("demand")
        ys = []
        for r in rows:
            gb = float(np.expm1(r.x[i_size]))
            key = (r.box, self.size_bucket(gb), self.demand_bucket(r.x[i_demand]))
            self.cells.setdefault(key, []).append(r.y)
            ys.append(r.y)
        self.default = float(np.median(ys)) if ys else 0.0
        return self

    def predict(self, rows: list[Example]) -> np.ndarray:
        from .features import NAMES as N

        i_size, i_demand = N.index("log_size_gb"), N.index("demand")
        out = []
        for r in rows:
            gb = float(np.expm1(r.x[i_size]))
            key = (r.box, self.size_bucket(gb), self.demand_bucket(r.x[i_demand]))
            cell = self.cells.get(key)
            # Three races is the service's own threshold for trusting a cell.
            out.append(float(np.median(cell)) if cell and len(cell) >= 3 else self.default)
        return np.asarray(out)


@dataclass
class Choice:
    """A bandit's pick, with the probability that makes it usable later."""

    box: str
    #: Estimated probability this box would be chosen again in this state.
    probability: float
    #: Posterior mean for the chosen box, for the shadow-mode report.
    expected: float


class ThompsonBandit:
    """Chooses a box by sampling the ranker's posterior.

    The probability is estimated by repeating the sample, which costs
    microseconds and is the only honest way to log a propensity for a
    sampling policy. Without it, every importance-weighted estimate
    downstream is quietly wrong.
    """

    def __init__(self, ranker: Ranker, probes: int = 256, seed: int = 0) -> None:
        self.ranker = ranker
        self.probes = probes
        self.rng = np.random.default_rng(seed)

    def choose(self, boxes: list[str], xs: list[list[float]]) -> Choice:
        if len(boxes) != len(xs):
            raise ValueError("one feature row per box")
        if not boxes:
            raise ValueError("no boxes to choose between")
        wins = np.zeros(len(boxes))
        for _ in range(self.probes):
            draw = self.ranker.sample(xs, self.rng)
            wins[int(np.argmax(draw))] += 1
        probs = wins / self.probes
        pick = int(np.argmax(self.rng.multinomial(1, probs)))
        mean, _ = self.ranker.predict_with_sigma(xs)
        return Choice(box=boxes[pick], probability=float(probs[pick]), expected=float(mean[pick]))
