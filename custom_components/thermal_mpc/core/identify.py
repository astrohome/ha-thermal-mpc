"""Parameter estimation for the RC model.

Two stages:

1. **Integral equation error.** Instead of regressing noisy 5-minute
   differences, regress the temperature change over a window of ``m`` steps
   on the *sum* of the regressors over that window. Measurement noise enters
   once per window instead of once per step, so it shrinks by ~``m`` relative
   to the signal, and the problem stays linear (sign-constrained LS).
2. **Output-error refinement.** Starting from stage 1, minimise the
   open-loop multi-step prediction error directly (projected
   Levenberg-Marquardt over many overlapping segments). That is the error
   that matters when the model is used to plan ahead, and it is not biased
   by noisy regressors the way equation-error fits are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lsq import FREE, NEGATIVE, POSITIVE, signed_lstsq


@dataclass
class Layout:
    """Flat parameter vector <-> per-room coefficients."""

    rooms: list[str]
    neighbours: dict[str, list[str]]
    inputs: list[str]
    signs: dict[str, str]  # input -> sign

    def blocks(self) -> list[tuple[str, int, int]]:
        """(room, start, stop) slices of the flat vector."""
        out, pos = [], 0
        for r in self.rooms:
            n = 1 + len(self.neighbours[r]) + len(self.inputs) + 1
            out.append((r, pos, pos + n))
            pos += n
        return out

    @property
    def size(self) -> int:
        """Total number of parameters."""
        return self.blocks()[-1][2] if self.rooms else 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds implied by the physical signs."""
        lo, hi = [], []
        for r in self.rooms:
            for s in (
                [POSITIVE] * (1 + len(self.neighbours[r]))
                + [self.signs[i] for i in self.inputs]
                + [FREE]
            ):
                lo.append(0.0 if s == POSITIVE else -np.inf)
                hi.append(0.0 if s == NEGATIVE else np.inf)
        return np.array(lo), np.array(hi)

    def matrices(self, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return (A, g_out, G, c) with dT/dt = A T + g_out T_out + G u + c."""
        n, idx = len(self.rooms), {r: k for k, r in enumerate(self.rooms)}
        A = np.zeros((n, n))
        g_out = np.zeros(n)
        G = np.zeros((n, len(self.inputs)))
        c = np.zeros(n)
        for k, (r, a, _) in enumerate(self.blocks()):
            g_out[k] = theta[a]
            A[k, k] -= theta[a]
            for j, o in enumerate(self.neighbours[r]):
                g = theta[a + 1 + j]
                A[k, idx[o]] += g
                A[k, k] -= g
            b = a + 1 + len(self.neighbours[r])
            G[k] = theta[b : b + len(self.inputs)]
            c[k] = theta[b + len(self.inputs)]
        return A, g_out, G, c


def integral_fit(
    temps: np.ndarray,
    t_out: np.ndarray,
    u: np.ndarray,
    layout: Layout,
    step_h: float,
    window: int,
) -> tuple[np.ndarray, list[int]]:
    """Stage 1: sign-constrained LS on ``window``-step integrated equations.

    Returns the flat parameter vector and the number of windows used per room.
    """
    n = temps.shape[0]
    theta = np.zeros(layout.size)
    used = []
    idx = {r: k for k, r in enumerate(layout.rooms)}
    for r, a, b in layout.blocks():
        k = idx[r]
        t_i = temps[:, k]
        feats = [t_out - t_i]
        signs = [POSITIVE]
        for o in layout.neighbours[r]:
            feats.append(temps[:, idx[o]] - t_i)
            signs.append(POSITIVE)
        for j, name in enumerate(layout.inputs):
            feats.append(u[:, j])
            signs.append(layout.signs[name])
        feats.append(np.ones(n))
        signs.append(FREE)
        X = np.column_stack(feats)
        bad = np.isnan(X).any(axis=1) | np.isnan(t_i)
        Xz = np.where(np.isnan(X), 0.0, X) * step_h
        # Window sums via cumulative sums; a window is valid only if every row
        # in it (and its end temperature) is known.
        cx = np.vstack([np.zeros(X.shape[1]), np.cumsum(Xz, axis=0)])
        cb = np.concatenate([[0], np.cumsum(bad)])
        starts = np.arange(0, n - window)
        ok = (cb[starts + window] - cb[starts] == 0) & ~np.isnan(t_i[starts + window])
        starts = starts[ok]
        used.append(int(starts.size))
        if starts.size == 0:
            continue
        design = cx[starts + window] - cx[starts]
        target = t_i[starts + window] - t_i[starts]
        theta[a:b] = signed_lstsq(design, target, signs)
    return theta, used


def _segments(
    temps: np.ndarray, t_out: np.ndarray, u: np.ndarray, horizon: int, stride: int
) -> np.ndarray:
    """Start indices where the initial state and all inputs are known."""
    n = temps.shape[0]
    bad_in = np.isnan(t_out) | np.isnan(u).any(axis=1)
    cb = np.concatenate([[0], np.cumsum(bad_in)])
    starts = np.arange(0, n - horizon, stride)
    ok = ~np.isnan(temps[starts]).any(axis=1)
    ok &= cb[starts + horizon] - cb[starts] == 0
    return starts[ok]


def simulate_batch(
    theta: np.ndarray,
    layout: Layout,
    x0: np.ndarray,
    t_out: np.ndarray,
    u: np.ndarray,
    step_h: float,
) -> np.ndarray:
    """Simulate many segments at once.

    ``x0`` is (S, R); ``t_out`` is (S, H); ``u`` is (S, H, I). Returns the
    states after each step, shape (S, H, R).
    """
    A, g_out, G, c = layout.matrices(theta)
    S, H = t_out.shape
    out = np.empty((S, H, x0.shape[1]))
    x = x0
    for k in range(H):
        dx = x @ A.T + t_out[:, k, None] * g_out + u[:, k, :] @ G.T + c
        x = x + step_h * dx
        out[:, k] = x
    return out


def refine(
    theta0: np.ndarray,
    layout: Layout,
    temps: np.ndarray,
    t_out: np.ndarray,
    u: np.ndarray,
    step_h: float,
    horizon: int,
    stride: int,
    max_iter: int = 40,
    prior_weight: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Stage 2: projected Levenberg-Marquardt on open-loop segment errors.

    A weak pull towards ``theta0`` keeps parameters the data cannot pin down
    (e.g. a coupling that never had a temperature difference across it) from
    wandering. Returns the refined vector and its RMSE (K).
    """
    starts = _segments(temps, t_out, u, horizon, stride)
    if starts.size == 0:
        return theta0, float("nan")
    offs = np.arange(1, horizon + 1)
    x0 = temps[starts]
    to = t_out[starts[:, None] + offs - 1]
    uu = u[starts[:, None] + offs - 1]
    target = temps[starts[:, None] + offs]  # (S, H, R)
    mask = ~np.isnan(target)
    tgt = np.where(mask, target, 0.0)
    lo, hi = layout.bounds()
    scale = np.maximum(np.abs(theta0), 1e-3)
    n_obs = max(1, int(mask.sum()))
    w_prior = np.sqrt(prior_weight * n_obs / theta0.size)

    def residuals(theta: np.ndarray) -> np.ndarray:
        sim = simulate_batch(theta, layout, x0, to, uu, step_h)
        r = np.where(mask, sim - tgt, 0.0).ravel()
        prior = w_prior * (theta - theta0) / scale
        return np.concatenate([r, prior])

    theta = np.clip(theta0, lo, hi)
    r = residuals(theta)
    cost = float(r @ r)
    lam = 1e-2
    eps = 1e-6 * scale
    for _ in range(max_iter):
        J = np.empty((r.size, theta.size))
        for p in range(theta.size):
            t2 = theta.copy()
            t2[p] += eps[p]
            J[:, p] = (residuals(t2) - r) / eps[p]
        JtJ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(8):
            H = JtJ + lam * np.diag(np.diag(JtJ) + 1e-12)
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            cand = np.clip(theta + step, lo, hi)
            rc = residuals(cand)
            cc = float(rc @ rc)
            if cc < cost:
                improved = True
                converged = (cost - cc) < 1e-7 * cost
                theta, r, cost = cand, rc, cc
                lam = max(lam / 3, 1e-7)
                break
            lam *= 10
        if not improved or converged:
            break
    sim = simulate_batch(theta, layout, x0, to, uu, step_h)
    err = (sim - tgt)[mask]
    return theta, float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
