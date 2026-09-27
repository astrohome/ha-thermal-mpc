"""Parameter estimation for the RC model.

Each room has a measured air temperature ``T`` and, optionally, a hidden
thermal-mass temperature ``M`` (walls, floor, furniture) that exchanges heat
only with that room's air::

    dT/dt = g_out (T_out - T) + sum_j g_j (T_j - T) + h (M - T) + G u + c
    dM/dt = k (T - M) + S u_mass

``u_mass`` are inputs that heat the mass directly (sun on the floor). ``M``
is driven only by measured signals, so it can be reconstructed by a
first-order filter (:func:`observe_mass`). For a given ``k``, writing ``F``
for a unit-gain low-pass with rate ``k``::

    M = F(T) + sum_q (s_q / k) F(u_q)

so the air equation is linear in ``h`` and ``w_q = h s_q / k``.

Estimation runs in two stages:

1. **Integral equation error.** Regress the temperature change over a window
   of ``m`` steps on the *sum* of the regressors over that window. Sensor
   noise enters once per window instead of once per step, so it no longer
   swamps slow heat losses, and the problem stays linear (sign-constrained
   LS). The mass rate ``k`` is picked from a small grid (or no mass at all)
   by open-loop error on the training data.
2. **Output-error refinement.** Minimise the multi-step open-loop prediction
   error directly (projected Levenberg-Marquardt over overlapping segments),
   refining everything except the mass rates. That is the error that matters when the
   model is used to plan ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lsq import FREE, NEGATIVE, POSITIVE, signed_lstsq

# Candidate mass time constants (h) for stage 1; None = no mass node.
MASS_TAU_GRID_H: tuple[float | None, ...] = (None, 2.0, 4.0, 8.0, 16.0, 32.0)
MAX_MASS_COUPLING = 4.0  # 1/h, keeps forward Euler at 5 min well stable
MIN_MASS_K, MAX_MASS_K = 1 / 72, 1.0  # mass time constant between 1 h and 3 d


@dataclass
class Layout:
    """Flat parameter vector <-> per-room coefficients.

    Per room: ``[g_out, g_neighbours..., h, s_mass..., gains..., offset]``;
    then one mass rate ``k`` per room at the end of the vector.
    """

    rooms: list[str]
    neighbours: dict[str, list[str]]
    inputs: list[str]
    signs: dict[str, str]  # input -> sign
    mass: list[bool] = field(default_factory=list)  # per room; empty = none
    mass_inputs: list[str] = field(default_factory=list)  # subset of inputs

    def has_mass(self, k: int) -> bool:
        """Whether room ``k`` has a mass node."""
        return bool(self.mass) and self.mass[k]

    def blocks(self) -> list[tuple[str, int, int]]:
        """(room, start, stop) slices of the per-room part of the vector."""
        out, pos = [], 0
        for r in self.rooms:
            n = 1 + len(self.neighbours[r]) + 1 + len(self.mass_inputs)
            n += len(self.inputs) + 1
            out.append((r, pos, pos + n))
            pos += n
        return out

    @property
    def k_start(self) -> int:
        """Index of the first mass rate."""
        return self.blocks()[-1][2] if self.rooms else 0

    @property
    def size(self) -> int:
        """Total number of parameters."""
        return self.k_start + len(self.rooms)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds implied by physics."""
        lo, hi = [], []
        for k, r in enumerate(self.rooms):
            for s in [POSITIVE] * (1 + len(self.neighbours[r])):
                lo.append(0.0 if s == POSITIVE else -np.inf)
                hi.append(np.inf)
            lo.append(0.0)
            hi.append(MAX_MASS_COUPLING if self.has_mass(k) else 0.0)
            for _ in self.mass_inputs:
                lo.append(0.0)
                hi.append(np.inf if self.has_mass(k) else 0.0)
            for s in [self.signs[i] for i in self.inputs] + [FREE]:
                lo.append(0.0 if s == POSITIVE else -np.inf)
                hi.append(0.0 if s == NEGATIVE else np.inf)
        for k in range(len(self.rooms)):
            lo.append(MIN_MASS_K if self.has_mass(k) else 0.0)
            hi.append(MAX_MASS_K if self.has_mass(k) else 0.0)
        return np.array(lo), np.array(hi)

    @property
    def mass_input_idx(self) -> list[int]:
        """Positions of the mass inputs within ``inputs``."""
        return [self.inputs.index(q) for q in self.mass_inputs]

    def matrices(self, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return ``(A, g_out, G, c, h, k, S)``.

        Air: ``dT/dt = A T + g_out T_out + h M + G u + c`` (``A`` already
        includes ``-h`` on the diagonal). Mass: ``dM/dt = k (T - M) + S u_m``.
        """
        n, idx = len(self.rooms), {r: k for k, r in enumerate(self.rooms)}
        A = np.zeros((n, n))
        g_out = np.zeros(n)
        G = np.zeros((n, len(self.inputs)))
        c = np.zeros(n)
        h = np.zeros(n)
        nq = len(self.mass_inputs)
        S = np.zeros((n, nq))
        for k, (r, a, _) in enumerate(self.blocks()):
            g_out[k] = theta[a]
            A[k, k] -= theta[a]
            for j, o in enumerate(self.neighbours[r]):
                g = theta[a + 1 + j]
                A[k, idx[o]] += g
                A[k, k] -= g
            b = a + 1 + len(self.neighbours[r])
            h[k] = theta[b]
            A[k, k] -= theta[b]
            S[k] = theta[b + 1 : b + 1 + nq]
            b += 1 + nq
            G[k] = theta[b : b + len(self.inputs)]
            c[k] = theta[b + len(self.inputs)]
        kk = theta[self.k_start : self.k_start + n].copy()
        return A, g_out, G, c, h, kk, S


def observe_mass(
    temps: np.ndarray,
    k: np.ndarray,
    step_h: float,
    u_mass: np.ndarray | None = None,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct mass temperatures from measured signals.

    ``M`` follows ``dM/dt = k (T - M) + S u_mass``; while ``T`` is unknown it
    holds, and unknown mass inputs count as zero. Starts at the first known
    air temperature of each room.
    """
    n, r = temps.shape
    a = np.clip(np.asarray(k, float) * step_h, 0.0, 1.0)
    drive = np.zeros((n, r))
    if u_mass is not None and S is not None and S.size:
        drive = np.where(np.isnan(u_mass), 0.0, u_mass) @ S.T * step_h
    out = np.empty((n, r))
    first = np.array(
        [
            temps[~np.isnan(temps[:, j]), j][0]
            if (~np.isnan(temps[:, j])).any()
            else 0.0
            for j in range(r)
        ]
    )
    m = first.copy()
    for t in range(n):
        out[t] = m
        x = temps[t]
        ok = ~np.isnan(x)
        m = np.where(ok, m + a * (np.where(ok, x, 0.0) - m) + drive[t], m)
    return out


def _lowpass(x: np.ndarray, k: float, step_h: float) -> np.ndarray:
    """Unit-gain first-order low-pass of a 1-D series (NaN treated as 0)."""
    a = min(1.0, k * step_h)
    x = np.where(np.isnan(x), 0.0, x)
    out = np.empty_like(x)
    f = 0.0
    for t in range(len(x)):
        out[t] = f
        f += a * (x[t] - f)
    return out


def integral_fit(
    temps: np.ndarray,
    t_out: np.ndarray,
    u: np.ndarray,
    layout: Layout,
    step_h: float,
    window: int,
    k: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Stage 1: sign-constrained LS on ``window``-step integrated equations.

    ``k`` gives the mass rate per room (ignored for rooms without mass).
    Returns the flat parameter vector and the number of windows per room.
    """
    n, n_rooms = temps.shape
    k = np.zeros(n_rooms) if k is None else np.asarray(k, float)
    mass = observe_mass(temps, k, step_h)
    theta = np.zeros(layout.size)
    theta[layout.k_start :] = np.where(
        [layout.has_mass(j) for j in range(n_rooms)], k, 0.0
    )
    used = []
    idx = {r: j for j, r in enumerate(layout.rooms)}
    for r, a, b in layout.blocks():
        j = idx[r]
        t_i = temps[:, j]
        feats = [t_out - t_i]
        signs = [POSITIVE]
        for o in layout.neighbours[r]:
            feats.append(temps[:, idx[o]] - t_i)
            signs.append(POSITIVE)
        with_mass = layout.has_mass(j)
        if with_mass:
            feats.append(mass[:, j] - t_i)
            signs.append(POSITIVE)
            for q in layout.mass_input_idx:
                feats.append(_lowpass(u[:, q], k[j], step_h))
                signs.append(POSITIVE)
        for jj, name in enumerate(layout.inputs):
            feats.append(u[:, jj])
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
        coef = signed_lstsq(design, target, signs)
        split = 1 + len(layout.neighbours[r])
        nq = len(layout.mass_inputs)
        if with_mass:
            # w_q = h s_q / k  ->  s_q = w_q k / h
            h_i = coef[split]
            w = coef[split + 1 : split + 1 + nq]
            coef[split + 1 : split + 1 + nq] = w * k[j] / h_i if h_i > 1e-9 else 0.0
        else:  # keep the vector layout: h = 0, s = 0
            coef = np.concatenate([coef[:split], np.zeros(1 + nq), coef[split:]])
        theta[a:b] = coef
    return theta, used


@dataclass
class Segments:
    """Training segments for open-loop evaluation."""

    starts: np.ndarray
    x0: np.ndarray  # (S, R)
    t_out: np.ndarray  # (S, H)
    u: np.ndarray  # (S, H, I)
    target: np.ndarray  # (S, H, R), NaN -> 0
    mask: np.ndarray  # (S, H, R)

    @classmethod
    def build(
        cls,
        temps: np.ndarray,
        t_out: np.ndarray,
        u: np.ndarray,
        horizon: int,
        stride: int,
    ) -> Segments:
        """Start where the initial state and all inputs are known."""
        n = temps.shape[0]
        bad_in = np.isnan(t_out) | np.isnan(u).any(axis=1)
        cb = np.concatenate([[0], np.cumsum(bad_in)])
        starts = np.arange(0, max(0, n - horizon), stride)
        ok = ~np.isnan(temps[starts]).any(axis=1)
        ok &= cb[starts + horizon] - cb[starts] == 0
        starts = starts[ok]
        offs = np.arange(1, horizon + 1)
        target = temps[starts[:, None] + offs]
        mask = ~np.isnan(target)
        return cls(
            starts=starts,
            x0=temps[starts],
            t_out=t_out[starts[:, None] + offs - 1],
            u=u[starts[:, None] + offs - 1],
            target=np.where(mask, target, 0.0),
            mask=mask,
        )

    def rmse(self, sim: np.ndarray) -> float:
        """Root-mean-square error of a simulation against the targets."""
        err = (sim - self.target)[self.mask]
        return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def simulate_batch(
    theta: np.ndarray,
    layout: Layout,
    x0: np.ndarray,
    t_out: np.ndarray,
    u: np.ndarray,
    step_h: float,
    m0: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate many segments at once.

    ``x0``/``m0`` are (S, R) air/mass states (``m0`` defaults to ``x0``);
    ``t_out`` is (S, H); ``u`` is (S, H, I). Returns air temperatures after
    each step, shape (S, H, R).
    """
    A, g_out, G, c, h, k, Sm = layout.matrices(theta)
    n_seg, H = t_out.shape
    qi = layout.mass_input_idx
    out = np.empty((n_seg, H, x0.shape[1]))
    x = x0
    m = x0 if m0 is None else m0
    for step in range(H):
        us = u[:, step, :]
        dx = x @ A.T + t_out[:, step, None] * g_out + h * m + us @ G.T + c
        dm = k * (x - m) + (us[:, qi] @ Sm.T if qi else 0.0)
        x = x + step_h * dx
        m = m + step_h * dm
        out[:, step] = x
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

    Mass rates ``k`` stay fixed (stage 1 picks them). With ``k`` fixed the
    segment-start mass is linear in the sun-to-mass gains,
    ``M0 = F(T) + (s / k) F(u)``, so it is recomputed exactly for every
    candidate and the objective stays consistent. A weak pull towards
    ``theta0`` keeps parameters the data cannot pin down (e.g. a coupling
    that never saw a temperature difference) from wandering. Returns the
    refined vector and its training RMSE (K).
    """
    seg = Segments.build(temps, t_out, u, horizon, stride)
    if seg.starts.size == 0:
        return theta0, float("nan")
    lo, hi = layout.bounds()
    ks = slice(layout.k_start, layout.size)
    lo[ks] = hi[ks] = theta0[ks]
    scale = np.maximum(np.abs(theta0), 1e-3)
    n_obs = max(1, int(seg.mask.sum()))
    w_prior = np.sqrt(prior_weight * n_obs / theta0.size)

    k = theta0[ks]
    base = observe_mass(temps, k, step_h)[seg.starts]  # F(T) at segment starts
    qi = layout.mass_input_idx
    # F_k(u_q) at segment starts, per room: (S, R, Q)
    fu = np.zeros((seg.starts.size, len(layout.rooms), len(qi)))
    for j in range(len(layout.rooms)):
        if k[j] > 0:
            for col, q in enumerate(qi):
                fu[:, j, col] = _lowpass(u[:, q], k[j], step_h)[seg.starts]
    inv_k = np.where(k > 0, 1.0 / np.where(k > 0, k, 1.0), 0.0)

    def m0_of(th: np.ndarray) -> np.ndarray:
        if not qi:
            return base
        Sm = layout.matrices(th)[6]
        return base + np.einsum("srq,rq->sr", fu, Sm * inv_k[:, None])

    def residuals(th: np.ndarray) -> np.ndarray:
        sim = simulate_batch(th, layout, seg.x0, seg.t_out, seg.u, step_h, m0_of(th))
        r = np.where(seg.mask, sim - seg.target, 0.0).ravel()
        return np.concatenate([r, w_prior * (th - theta0) / scale])

    theta = np.clip(theta0, lo, hi)
    free = np.flatnonzero(hi > lo)  # skip parameters pinned by bounds
    r = residuals(theta)
    cost = float(r @ r)
    lam = 1e-2
    eps = 1e-6 * scale
    for _ in range(max_iter):
        J = np.zeros((r.size, free.size))
        for col, p in enumerate(free):
            t2 = theta.copy()
            t2[p] += eps[p]
            J[:, col] = (residuals(t2) - r) / eps[p]
        JtJ = J.T @ J
        g = J.T @ r
        improved = converged = False
        for _ in range(8):
            Hm = JtJ + lam * np.diag(np.diag(JtJ) + 1e-12)
            try:
                step = np.linalg.solve(Hm, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            cand = theta.copy()
            cand[free] = np.clip(theta[free] + step, lo[free], hi[free])
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

    sim = simulate_batch(theta, layout, seg.x0, seg.t_out, seg.u, step_h, m0_of(theta))
    return theta, seg.rmse(sim)


def _observe(
    theta: np.ndarray, layout: Layout, temps: np.ndarray, u: np.ndarray, step_h: float
) -> np.ndarray:
    mats = layout.matrices(theta)
    u_mass = u[:, layout.mass_input_idx] if layout.mass_inputs else None
    return observe_mass(temps, mats[5], step_h, u_mass, mats[6])


def open_loop_error(
    theta: np.ndarray,
    layout: Layout,
    temps: np.ndarray,
    u: np.ndarray,
    seg: Segments,
    step_h: float,
) -> float:
    """Training open-loop RMSE of a parameter vector (used to pick mass)."""
    m0 = _observe(theta, layout, temps, u, step_h)[seg.starts]
    return seg.rmse(simulate_batch(theta, layout, seg.x0, seg.t_out, seg.u, step_h, m0))
