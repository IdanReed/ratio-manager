"""Generating swarm-growth curves.

The thing the log will be short of is not events — it is **swarm-growth curves
conditioned on release attributes**, and the tail where a release blooms.
Races arrive at a few dozen a day. Every policy downstream wants orders more.

Four families, best first, and the ranking is the point of this module:

:class:`ParametricCopula`
    A log-logistic rise-and-decay curve per release, with a Gaussian copula
    over the curve's parameters. Works at a few hundred traces, which is the
    regime that actually exists. **This is the floor to beat, and may well be
    enough.**

:class:`AutoregressiveCurves`
    A small GRU over quantized scrape deltas, conditioned on category, size
    and freeleech. Teacher-forced, so it has a likelihood — which is what lets
    it be calibrated and model-selected rather than eyeballed.

:class:`DiffusionCurves`
    Denoising over fixed-length curves. Good samples, clean conditioning, no
    likelihood, and more machinery than the problem needs. A middle option.

:class:`CurveGAN`
    Included because it was asked for, and kept as an arm of the bake-off
    rather than the plan. On a few thousand short traces, mode collapse is the
    default outcome, there is no likelihood to select on, and "the samples
    look right" is not a test. If it wins :func:`tstr`, it earns its place;
    nothing else would persuade me.

All four are scored by :func:`evaluate_generators`, and only the third of its
three tests decides: train the policy on synthetic curves, evaluate it in the
simulator calibrated on **real held-out** data. Fidelity metrics are
diagnostics, not verdicts.

numpy is required; torch only for the two learned families, and they say so
when it is missing rather than failing obscurely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

#: Curves are resampled onto this many points over `HORIZON_S` so every
#: family sees the same shape.
POINTS = 48
HORIZON_S = 12 * 3600.0


@dataclass
class Curve:
    """One release's leecher count over time, with what conditioned it."""

    #: Leechers at each of `POINTS` evenly spaced times over `HORIZON_S`.
    leechers: np.ndarray
    size_gb: float
    category: str
    freeleech: bool

    def peak(self) -> float:
        return float(self.leechers.max()) if self.leechers.size else 0.0

    def time_to_peak(self) -> float:
        if not self.leechers.size:
            return 0.0
        return float(np.argmax(self.leechers)) * HORIZON_S / POINTS

    def area(self) -> float:
        """Total leecher-seconds: a proxy for how much there was to earn."""
        return float(self.leechers.sum()) * HORIZON_S / POINTS


class Generator(Protocol):
    def fit(self, curves: Sequence[Curve]) -> "Generator": ...
    def sample(self, n: int, rng: np.random.Generator) -> list[Curve]: ...
    @property
    def name(self) -> str: ...


# --- 1. parametric plus copula ------------------------------------------------


def log_logistic(t: np.ndarray, peak: float, tau: float, rise: float, decay: float) -> np.ndarray:
    """The rise-then-decay shape, scaled so its maximum is `peak`.

    The same curve the Rust simulator uses, so a fitted parameter means the
    same thing on both sides.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.maximum(t, 1e-9) / max(tau, 1e-9)
        raw = x**rise / (1.0 + x ** (rise + decay))
        xm = (rise / max(decay, 1e-6)) ** (1.0 / (rise + decay))
        top = xm**rise / (1.0 + xm ** (rise + decay))
    out = peak * raw / top if top > 0 else np.zeros_like(x)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_curve(c: Curve) -> tuple[float, float, float, float]:
    """Crude moment fit of (peak, tau, rise, decay) to one curve.

    Deliberately not an optimizer: with 48 noisy points, a least-squares fit
    of four coupled parameters finds a different local minimum per curve and
    the copula then models the optimizer rather than the swarms.
    """
    peak = max(c.peak(), 1e-6)
    tau = max(c.time_to_peak(), HORIZON_S / POINTS)
    t = np.arange(POINTS) * HORIZON_S / POINTS
    half = c.leechers >= peak / 2.0
    width = float(half.sum()) * HORIZON_S / POINTS
    # A wider peak means a slower rise and a slower decay.
    rise = float(np.clip(2.0 * tau / max(width, 1e-6), 0.5, 4.0))
    tail = c.leechers[int(np.argmax(c.leechers)) :]
    if tail.size > 2 and tail[0] > 0:
        drop = max(float(tail[-1] / tail[0]), 1e-3)
        decay = float(np.clip(-np.log(drop) / 2.0, 0.3, 4.0))
    else:
        decay = 1.0
    del t
    return peak, tau, rise, decay


class ParametricCopula:
    """Log-logistic curves with a Gaussian copula over their parameters."""

    name = "parametric-copula"

    def __init__(self) -> None:
        self.marginals_: list[np.ndarray] = []
        self.corr_: np.ndarray | None = None
        self.categories_: list[str] = []
        self.sizes_: np.ndarray | None = None

    def fit(self, curves: Sequence[Curve]) -> "ParametricCopula":
        if len(curves) < 4:
            raise ValueError("too few curves to fit a copula")
        params = np.asarray([fit_curve(c) for c in curves], dtype=float)
        # Marginals kept as sorted samples: the empirical quantile function,
        # which needs no distributional assumption.
        self.marginals_ = [np.sort(params[:, j]) for j in range(params.shape[1])]
        ranks = np.argsort(np.argsort(params, axis=0), axis=0) + 1.0
        u = ranks / (len(curves) + 1.0)
        z = _norm_ppf(u)
        self.corr_ = np.corrcoef(z, rowvar=False)
        self.categories_ = [c.category for c in curves]
        self.sizes_ = np.asarray([c.size_gb for c in curves], dtype=float)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> list[Curve]:
        if self.corr_ is None or self.sizes_ is None:
            raise RuntimeError("not fitted")
        d = len(self.marginals_)
        # Jitter the correlation onto the nearest usable matrix rather than
        # failing: with few curves the empirical one is often not PSD.
        chol = _safe_cholesky(self.corr_)
        t = np.arange(POINTS) * HORIZON_S / POINTS
        out = []
        for _ in range(n):
            z = chol @ rng.standard_normal(d)
            u = _norm_cdf(z)
            peak, tau, rise, decay = (
                float(np.quantile(self.marginals_[j], np.clip(u[j], 1e-6, 1 - 1e-6)))
                for j in range(d)
            )
            i = int(rng.integers(len(self.sizes_)))
            out.append(
                Curve(
                    leechers=log_logistic(t, peak, tau, rise, decay),
                    size_gb=float(self.sizes_[i]),
                    category=self.categories_[i],
                    freeleech=True,
                )
            )
        return out


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF, via the Beasley-Springer-Moro approximation.

    Good to about 1e-9 over the range a copula uses, and keeps scipy out of
    the dependency list.
    """
    u = np.clip(u, 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(u, dtype=float)

    lo = u < plow
    q = np.sqrt(-2 * np.log(u[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )

    hi = u > phigh
    q = np.sqrt(-2 * np.log(1 - u[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )

    mid = ~(lo | hi)
    q = u[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
    return out


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    return np.asarray([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in np.atleast_1d(z)])


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Cholesky of a correlation matrix, nudged to positive definite."""
    c = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(c, 1.0)
    for eps in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
        try:
            return np.linalg.cholesky(c + eps * np.eye(c.shape[0]))
        except np.linalg.LinAlgError:
            continue
    return np.eye(c.shape[0])


# --- 2-4. the learned families ------------------------------------------------


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "this generator needs torch. It is not installed on the services "
            "VM on purpose — train on the desktop's GPU (see open question 1 "
            "in ServerNotes/designs/ratio-ml.md). ParametricCopula needs only "
            "numpy and runs anywhere."
        ) from e
    import torch

    return torch


class AutoregressiveCurves:
    """A GRU over quantized deltas, conditioned on the release's attributes.

    The recommended learned family: teacher-forced training gives a
    likelihood, so held-out negative log-likelihood selects the model instead
    of a judgement about how the samples look.
    """

    name = "autoregressive"

    def __init__(self, bins: int = 32, hidden: int = 64, epochs: int = 200, lr: float = 3e-3) -> None:
        self.bins, self.hidden, self.epochs, self.lr = bins, hidden, epochs, lr
        self.edges_: np.ndarray | None = None
        self.model_ = None
        self.cats_: list[str] = []

    def _quantize(self, curves: Sequence[Curve]) -> np.ndarray:
        d = np.diff(np.stack([c.leechers for c in curves]), axis=1, prepend=0.0)
        if self.edges_ is None:
            qs = np.linspace(0, 100, self.bins + 1)[1:-1]
            self.edges_ = np.percentile(d, qs)
        return np.digitize(d, self.edges_)

    def fit(self, curves: Sequence[Curve]) -> "AutoregressiveCurves":
        torch = _require_torch()
        import torch.nn as nn

        self.cats_ = sorted({c.category for c in curves})
        tokens = torch.as_tensor(self._quantize(curves), dtype=torch.long)
        cond = torch.as_tensor(
            np.stack(
                [
                    [
                        np.log1p(c.size_gb),
                        float(c.freeleech),
                        float(self.cats_.index(c.category)) / max(len(self.cats_) - 1, 1),
                    ]
                    for c in curves
                ]
            ),
            dtype=torch.float32,
        )

        class Net(nn.Module):
            def __init__(self, bins: int, hidden: int) -> None:
                super().__init__()
                self.emb = nn.Embedding(bins, hidden)
                self.cond = nn.Linear(3, hidden)
                self.gru = nn.GRU(hidden, hidden, batch_first=True)
                self.out = nn.Linear(hidden, bins)

            def forward(self, x, c):
                h = self.cond(c).unsqueeze(0).contiguous()
                y, _ = self.gru(self.emb(x), h)
                return self.out(y)

        net = Net(self.bins, self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = net(tokens[:, :-1], cond)
            loss = loss_fn(logits.reshape(-1, self.bins), tokens[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
        self.model_ = net
        return self

    def nll(self, curves: Sequence[Curve]) -> float:
        """Held-out negative log-likelihood per step. The selection criterion
        the other families cannot offer."""
        torch = _require_torch()
        import torch.nn as nn

        if self.model_ is None:
            raise RuntimeError("not fitted")
        tokens = torch.as_tensor(self._quantize(curves), dtype=torch.long)
        cond = torch.as_tensor(
            np.stack(
                [
                    [
                        np.log1p(c.size_gb),
                        float(c.freeleech),
                        float(self.cats_.index(c.category) if c.category in self.cats_ else 0)
                        / max(len(self.cats_) - 1, 1),
                    ]
                    for c in curves
                ]
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = self.model_(tokens[:, :-1], cond)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, self.bins), tokens[:, 1:].reshape(-1)
            )
        return float(loss)

    def sample(self, n: int, rng: np.random.Generator) -> list[Curve]:
        torch = _require_torch()

        if self.model_ is None or self.edges_ is None:
            raise RuntimeError("not fitted")
        centres = np.concatenate([[self.edges_[0]], (self.edges_[1:] + self.edges_[:-1]) / 2,
                                  [self.edges_[-1]]])
        out: list[Curve] = []
        for _ in range(n):
            size = float(np.exp(rng.uniform(0.0, 4.6)) - 1.0)
            cat = self.cats_[int(rng.integers(len(self.cats_)))] if self.cats_ else ""
            cond = torch.as_tensor(
                [[np.log1p(size), 1.0,
                  float(self.cats_.index(cat)) / max(len(self.cats_) - 1, 1) if self.cats_ else 0.0]],
                dtype=torch.float32,
            )
            tok = torch.zeros((1, 1), dtype=torch.long)
            deltas = []
            with torch.no_grad():
                for _ in range(POINTS):
                    logits = self.model_(tok, cond)[:, -1, :]
                    p = torch.softmax(logits, dim=-1).numpy().ravel()
                    nxt = int(rng.choice(len(p), p=p / p.sum()))
                    deltas.append(centres[min(nxt, len(centres) - 1)])
                    tok = torch.cat([tok, torch.as_tensor([[nxt]], dtype=torch.long)], dim=1)
            lee = np.maximum(np.cumsum(deltas), 0.0)
            out.append(Curve(leechers=lee, size_gb=size, category=cat, freeleech=True))
        return out


class DiffusionCurves:
    """Denoising diffusion over fixed-length curves.

    Deliberately a thin implementation: it is the middle option, listed so the
    bake-off is honest about what was and was not tried, not a recommendation.
    """

    name = "diffusion"

    def __init__(self, steps: int = 50, hidden: int = 128, epochs: int = 300, lr: float = 1e-3) -> None:
        self.steps, self.hidden, self.epochs, self.lr = steps, hidden, epochs, lr
        self.model_ = None
        self.scale_ = 1.0
        self.meta_: list[tuple[float, str]] = []

    def fit(self, curves: Sequence[Curve]) -> "DiffusionCurves":
        torch = _require_torch()
        import torch.nn as nn

        x = np.stack([c.leechers for c in curves])
        self.scale_ = float(np.abs(x).max()) or 1.0
        self.meta_ = [(c.size_gb, c.category) for c in curves]
        data = torch.as_tensor(x / self.scale_, dtype=torch.float32)

        net = nn.Sequential(
            nn.Linear(POINTS + 1, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, POINTS),
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        betas = torch.linspace(1e-4, 0.08, self.steps)
        alphas = torch.cumprod(1 - betas, dim=0)
        for _ in range(self.epochs):
            t = torch.randint(0, self.steps, (data.shape[0],))
            a = alphas[t].unsqueeze(1)
            noise = torch.randn_like(data)
            noisy = a.sqrt() * data + (1 - a).sqrt() * noise
            pred = net(torch.cat([noisy, (t.float() / self.steps).unsqueeze(1)], dim=1))
            loss = ((pred - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.model_ = (net, betas, alphas)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> list[Curve]:
        torch = _require_torch()

        if self.model_ is None:
            raise RuntimeError("not fitted")
        net, betas, alphas = self.model_
        x = torch.randn((n, POINTS))
        with torch.no_grad():
            for t in reversed(range(self.steps)):
                tt = torch.full((n, 1), t / self.steps)
                eps = net(torch.cat([x, tt], dim=1))
                a = alphas[t]
                x = (x - (1 - a).sqrt() * eps) / a.sqrt()
                if t > 0:
                    x = x + betas[t].sqrt() * torch.randn_like(x)
        arr = np.maximum(x.numpy() * self.scale_, 0.0)
        return [
            Curve(leechers=arr[i], size_gb=self.meta_[int(rng.integers(len(self.meta_)))][0],
                  category=self.meta_[int(rng.integers(len(self.meta_)))][1], freeleech=True)
            for i in range(n)
        ]


class CurveGAN:
    """A small conditional GAN over curves.

    Here because it was asked about. Kept honest: on this data size, watch the
    :func:`mode_coverage` number in the bake-off — if it collapses, the samples
    will still look plausible one at a time while covering a fraction of the
    real variety, and a policy trained on them will overfit to that fraction.
    """

    name = "gan"

    def __init__(self, latent: int = 16, hidden: int = 128, epochs: int = 600, lr: float = 2e-4) -> None:
        self.latent, self.hidden, self.epochs, self.lr = latent, hidden, epochs, lr
        self.g_ = None
        self.scale_ = 1.0
        self.meta_: list[tuple[float, str]] = []

    def fit(self, curves: Sequence[Curve]) -> "CurveGAN":
        torch = _require_torch()
        import torch.nn as nn

        x = np.stack([c.leechers for c in curves])
        self.scale_ = float(np.abs(x).max()) or 1.0
        self.meta_ = [(c.size_gb, c.category) for c in curves]
        real = torch.as_tensor(x / self.scale_, dtype=torch.float32)

        g = nn.Sequential(
            nn.Linear(self.latent, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, POINTS), nn.Softplus(),
        )
        d = nn.Sequential(
            nn.Linear(POINTS, self.hidden), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.hidden), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, 1),
        )
        og = torch.optim.Adam(g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        od = torch.optim.Adam(d.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        n = real.shape[0]
        for _ in range(self.epochs):
            z = torch.randn((n, self.latent))
            fake = g(z)
            od.zero_grad()
            loss_d = bce(d(real), torch.ones((n, 1))) + bce(d(fake.detach()), torch.zeros((n, 1)))
            loss_d.backward()
            od.step()
            og.zero_grad()
            loss_g = bce(d(fake), torch.ones((n, 1)))
            loss_g.backward()
            og.step()
        self.g_ = g
        return self

    def sample(self, n: int, rng: np.random.Generator) -> list[Curve]:
        torch = _require_torch()

        if self.g_ is None:
            raise RuntimeError("not fitted")
        with torch.no_grad():
            arr = self.g_(torch.randn((n, self.latent))).numpy() * self.scale_
        return [
            Curve(leechers=np.maximum(arr[i], 0.0),
                  size_gb=self.meta_[int(rng.integers(len(self.meta_)))][0],
                  category=self.meta_[int(rng.integers(len(self.meta_)))][1], freeleech=True)
            for i in range(n)
        ]


# --- scoring ------------------------------------------------------------------


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return 1.0
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def marginal_fidelity(real: Sequence[Curve], fake: Sequence[Curve]) -> dict[str, float]:
    """KS distance per summary statistic. Lower is better; 0 is identical."""
    stats: dict[str, Callable[[Curve], float]] = {
        "peak": Curve.peak,
        "time_to_peak": Curve.time_to_peak,
        "area": Curve.area,
    }
    return {
        k: ks_distance(np.asarray([f(c) for c in real]), np.asarray([f(c) for c in fake]))
        for k, f in stats.items()
    }


def mode_coverage(real: Sequence[Curve], fake: Sequence[Curve]) -> float:
    """Fraction of real curves that some synthetic curve resembles.

    The GAN check. A collapsed generator scores near zero here while its
    marginals can still look respectable, which is exactly the failure that
    eyeballing samples misses.
    """
    if not real or not fake:
        return 0.0
    r = np.stack([c.leechers for c in real])
    f = np.stack([c.leechers for c in fake])
    scale = float(np.abs(r).max()) or 1.0
    r, f = r / scale, f / scale
    # A real curve is covered if some synthetic curve is nearer to it than the
    # median distance between two real curves.
    dists_rr = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
    thresh = float(np.median(dists_rr[dists_rr > 0])) if (dists_rr > 0).any() else 1.0
    dists_rf = np.linalg.norm(r[:, None, :] - f[None, :, :], axis=2)
    return float(np.mean(dists_rf.min(axis=1) < thresh))


@dataclass
class GeneratorReport:
    name: str
    marginals: dict[str, float]
    coverage: float
    #: Mean return a policy trained on this generator's output earned in the
    #: real-calibrated simulator. The number that decides.
    tstr: float | None = None
    #: Held-out negative log-likelihood, for the families that have one.
    nll: float | None = None

    def render(self) -> str:
        m = "  ".join(f"{k}={v:.3f}" for k, v in sorted(self.marginals.items()))
        tstr = "n/a" if self.tstr is None else f"{self.tstr:.1f}"
        nll = "n/a" if self.nll is None else f"{self.nll:.3f}"
        return (
            f"{self.name:<20} KS[{m}]  coverage={self.coverage:.2f}  "
            f"nll={nll}  TSTR={tstr}"
        )


def evaluate_generators(
    real_train: Sequence[Curve],
    real_test: Sequence[Curve],
    gens: Sequence[Generator],
    n_samples: int = 256,
    seed: int = 0,
    tstr: Callable[[list[Curve]], float] | None = None,
) -> list[GeneratorReport]:
    """Fit each generator on the training curves and score it on held-out ones.

    ``tstr`` is the only test that decides: hand it synthetic curves, get back
    what a policy trained on them earned in the real-calibrated simulator.
    Without it the report is diagnostics only, and says so.
    """
    rng = np.random.default_rng(seed)
    reports = []
    for g in gens:
        g.fit(real_train)
        fake = g.sample(n_samples, rng)
        rep = GeneratorReport(
            name=g.name,
            marginals=marginal_fidelity(real_test, fake),
            coverage=mode_coverage(real_test, fake),
        )
        if hasattr(g, "nll"):
            try:
                rep.nll = g.nll(real_test)  # type: ignore[attr-defined]
            except Exception:
                rep.nll = None
        if tstr is not None:
            rep.tstr = tstr(fake)
        reports.append(rep)
    # Ranked by TSTR when it is available, else by coverage. Never by
    # marginals alone, which is how a collapsed generator wins a bake-off.
    reports.sort(key=lambda r: (-(r.tstr if r.tstr is not None else -1e18), -r.coverage))
    return reports
