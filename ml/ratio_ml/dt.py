"""The decision transformer, and the variant that is actually safe here.

The operator's preferred route. It fits the data well: the event log is
already a typed token stream, trajectories are
``(return-to-go, state, action)`` repeated, and the behaviour policy's
candidates and scores are recorded at every step, which is what an offline
sequence model learns from.

**The problem, stated plainly.** A decision transformer is return-conditioned
imitation. At inference you ask for a return and it reproduces the actions
that preceded that return in training. That is sound when the environment is
near-deterministic, and misleading when it is not: whether a swarm blooms is
mostly not a consequence of our action. Conditioning on a *realized* return
therefore teaches the model to imitate luck. The fleet's own history has
examples — races that returned 2.3 at 3% downloaded because the swarm happened
to be seeder-poor, not because the add was clever.

So two heads are provided:

``DecisionTransformer(expected_rtg=False)``
    The textbook version, conditioned on realized return-to-go. Here as the
    comparison point, and because it is what the literature means by "DT".

``DecisionTransformer(expected_rtg=True)``
    Conditioned on the *expected* return of the state, predicted by a learned
    value head and used in place of the realized one. This is the ESPER idea,
    and it is the version to trust in a stochastic environment. It is the
    default.

The honest ordering remains the one in ``ServerNotes/designs/ratio-ml.md``:
train this in simulation, compare it against the ranker and against MPC over
the simulator, and ship it only if it wins. Without the bandit's exploration
data, a DT trained on today's log would learn to imitate a bucketed median.

torch is required. Train on the desktop's GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import NAMES


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the decision transformer needs torch, which is deliberately not "
            "installed on the services VM. Train on the desktop; see open "
            "question 1 in ServerNotes/designs/ratio-ml.md."
        ) from e
    return torch


#: The action space. Adding to a box, or passing. Evictions are a separate
#: head in a later version; mixing them in now would make every trajectory's
#: action distribution dominated by "pass".
ACTIONS = ("pass", "add_home", "add_seedit4me", "add_pulsed")


@dataclass
class Trajectory:
    """One episode: a day of decisions on one fleet, in order."""

    states: np.ndarray  # (T, len(NAMES))
    actions: np.ndarray  # (T,) indices into ACTIONS
    rewards: np.ndarray  # (T,) reward credited at that step
    #: Behaviour-policy probability per step, where it was randomized.
    propensities: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = len(self.actions)
        if self.states.shape[0] != t or self.rewards.shape[0] != t:
            raise ValueError("states, actions and rewards must be the same length")
        if self.states.shape[1] != len(NAMES):
            raise ValueError(
                f"states have {self.states.shape[1]} features, the shared vector has {len(NAMES)}"
            )

    def returns_to_go(self, gamma: float = 1.0) -> np.ndarray:
        """Discounted sum of future rewards at each step.

        Undiscounted by default: the reward is money-like (GB uploaded) and a
        GB earned tomorrow is worth a GB.
        """
        out = np.zeros_like(self.rewards, dtype=float)
        acc = 0.0
        for i in range(len(self.rewards) - 1, -1, -1):
            acc = self.rewards[i] + gamma * acc
            out[i] = acc
        return out


@dataclass
class Batch:
    states: "object"
    actions: "object"
    rtg: "object"
    timesteps: "object"
    mask: "object"


def make_batches(
    trajs: list[Trajectory],
    context: int = 20,
    batch_size: int = 32,
    gamma: float = 1.0,
    seed: int = 0,
):
    """Sample fixed-length windows, left-padded, as a DT expects."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    if not trajs:
        raise ValueError("no trajectories")

    d = len(NAMES)
    while True:
        S = np.zeros((batch_size, context, d), dtype=np.float32)
        A = np.zeros((batch_size, context), dtype=np.int64)
        R = np.zeros((batch_size, context, 1), dtype=np.float32)
        T = np.zeros((batch_size, context), dtype=np.int64)
        M = np.zeros((batch_size, context), dtype=np.float32)
        for b in range(batch_size):
            tr = trajs[int(rng.integers(len(trajs)))]
            n = len(tr.actions)
            start = int(rng.integers(max(n, 1)))
            end = min(start + context, n)
            k = end - start
            if k <= 0:
                continue
            rtg = tr.returns_to_go(gamma)[start:end]
            # Left-pad so the most recent step is always last.
            S[b, -k:] = tr.states[start:end]
            A[b, -k:] = tr.actions[start:end]
            R[b, -k:, 0] = rtg
            T[b, -k:] = np.arange(start, end)
            M[b, -k:] = 1.0
        yield Batch(
            states=torch.as_tensor(S),
            actions=torch.as_tensor(A),
            rtg=torch.as_tensor(R),
            timesteps=torch.as_tensor(T),
            mask=torch.as_tensor(M),
        )


class DecisionTransformer:
    """A small causal transformer over (return, state, action) triples.

    Deliberately small: a few hundred thousand parameters over at most a few
    tens of thousands of real decisions. A larger model on this data would
    memorize it.
    """

    def __init__(
        self,
        context: int = 20,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        expected_rtg: bool = True,
        lr: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.context = context
        self.d_model = d_model
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.expected_rtg = expected_rtg
        self.lr = lr
        self.seed = seed
        self.net_ = None
        self.rtg_scale_ = 1.0
        self.state_mean_: np.ndarray | None = None
        self.state_std_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []

    # -- model -------------------------------------------------------------

    def _build(self):
        torch = _torch()
        import torch.nn as nn

        d, dm = len(NAMES), self.d_model
        n_heads, n_layers, drop = self.heads, self.layers, self.dropout
        n_act = len(ACTIONS)
        expected = self.expected_rtg

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_state = nn.Linear(d, dm)
                self.embed_action = nn.Embedding(n_act, dm)
                self.embed_rtg = nn.Linear(1, dm)
                self.embed_time = nn.Embedding(4096, dm)
                self.ln = nn.LayerNorm(dm)
                block = nn.TransformerEncoderLayer(
                    d_model=dm,
                    nhead=n_heads,
                    dim_feedforward=4 * dm,
                    dropout=drop,
                    batch_first=True,
                    norm_first=True,
                )
                self.body = nn.TransformerEncoder(block, num_layers=n_layers)
                self.head_action = nn.Linear(dm, n_act)
                # The value head: used as the conditioning signal in the
                # expected-return variant, and trained either way so the two
                # variants are otherwise identical.
                self.head_value = nn.Linear(dm, 1)
                self.expected = expected

            def forward(self, states, actions, rtg, timesteps, mask):
                B, L, _ = states.shape
                t = self.embed_time(timesteps.clamp(max=4095))
                s = self.embed_state(states) + t
                a = self.embed_action(actions) + t
                r = self.embed_rtg(rtg) + t
                # Interleave (r, s, a) per step, the DT layout.
                x = torch.stack([r, s, a], dim=2).reshape(B, L * 3, -1)
                x = self.ln(x)
                causal = torch.triu(
                    torch.ones(L * 3, L * 3, dtype=torch.bool, device=x.device), diagonal=1
                )
                pad = mask.repeat_interleave(3, dim=1) < 0.5
                h = self.body(x, mask=causal, src_key_padding_mask=pad)
                # Predict the action from the state token's position.
                hs = h.reshape(B, L, 3, -1)[:, :, 1, :]
                return self.head_action(hs), self.head_value(hs).squeeze(-1)

        torch.manual_seed(self.seed)
        return Net()

    # -- api ---------------------------------------------------------------

    def fit(self, trajs: list[Trajectory], steps: int = 2000, batch_size: int = 32) -> "DecisionTransformer":
        torch = _torch()
        import torch.nn as nn

        states = np.concatenate([t.states for t in trajs], axis=0)
        self.state_mean_ = states.mean(axis=0)
        std = states.std(axis=0)
        std[std < 1e-12] = 1.0
        self.state_std_ = std
        rtgs = np.concatenate([t.returns_to_go() for t in trajs])
        self.rtg_scale_ = float(np.abs(rtgs).max()) or 1.0

        normed = [
            Trajectory(
                states=((t.states - self.state_mean_) / self.state_std_).astype(np.float32),
                actions=t.actions,
                rewards=t.rewards / self.rtg_scale_,
                propensities=t.propensities,
            )
            for t in trajs
        ]

        net = self._build()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss(reduction="none")
        batches = make_batches(normed, context=self.context, batch_size=batch_size, seed=self.seed)
        log: list[dict[str, float]] = []
        net.train()
        for i in range(steps):
            b = next(batches)
            logits, value = net(b.states, b.actions, b.rtg, b.timesteps, b.mask)
            n_act = len(ACTIONS)
            loss_a = (ce(logits.reshape(-1, n_act), b.actions.reshape(-1)) * b.mask.reshape(-1)).sum() / b.mask.sum().clamp(min=1)
            # The value head regresses the realized return-to-go. In the
            # expected-RTG variant this is what conditions inference, so it
            # has to be trained even when it is not the loss of interest.
            loss_v = (((value - b.rtg.squeeze(-1)) ** 2) * b.mask).sum() / b.mask.sum().clamp(min=1)
            loss = loss_a + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
            opt.step()
            if i % 100 == 0:
                log.append(
                    {"step": float(i), "action_loss": float(loss_a), "value_loss": float(loss_v)}
                )
        net.eval()
        self.net_ = net
        self.history_ = log  # type: ignore[assignment]
        return self

    def act(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        target_return: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Choose the next action given the recent window.

        ``states`` is ``(T, len(NAMES))`` and ``actions`` is ``(T,)`` with the
        last entry a placeholder for the action being chosen.

        In the expected-return variant, ``target_return`` is ignored and the
        value head's estimate conditions the model instead. That is the point:
        asking for a return the environment will not hand out is how the
        textbook variant goes wrong here.
        """
        torch = _torch()

        if self.net_ is None or self.state_mean_ is None or self.state_std_ is None:
            raise RuntimeError("not fitted")
        T = min(len(actions), self.context)
        s = ((states[-T:] - self.state_mean_) / self.state_std_).astype(np.float32)
        S = np.zeros((1, self.context, len(NAMES)), dtype=np.float32)
        A = np.zeros((1, self.context), dtype=np.int64)
        R = np.zeros((1, self.context, 1), dtype=np.float32)
        TS = np.zeros((1, self.context), dtype=np.int64)
        M = np.zeros((1, self.context), dtype=np.float32)
        S[0, -T:] = s
        A[0, -T:] = actions[-T:]
        TS[0, -T:] = np.arange(T)
        M[0, -T:] = 1.0
        if not self.expected_rtg:
            if target_return is None:
                raise ValueError(
                    "the realized-return variant needs a target_return; "
                    "the expected-return variant does not take one"
                )
            R[0, -T:, 0] = target_return / self.rtg_scale_

        with torch.no_grad():
            args = (
                torch.as_tensor(S),
                torch.as_tensor(A),
                torch.as_tensor(R),
                torch.as_tensor(TS),
                torch.as_tensor(M),
            )
            if self.expected_rtg:
                # One pass to estimate the state's value, then condition on it.
                _, v = self.net_(*args)
                R[0, -T:, 0] = v.numpy()[0, -T:]
                args = (args[0], args[1], torch.as_tensor(R), args[3], args[4])
            logits, _ = self.net_(*args)
            p = torch.softmax(logits[0, -1], dim=-1).numpy()
        return int(np.argmax(p)), p
