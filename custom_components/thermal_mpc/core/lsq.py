"""Sign-constrained linear least squares in plain numpy."""

from __future__ import annotations

import numpy as np

POSITIVE, NEGATIVE, FREE = "positive", "negative", "free"


def nnls(A: np.ndarray, b: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """Lawson-Hanson non-negative least squares: min ||Ax - b|| s.t. x >= 0."""
    _, n = A.shape
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    tol = 10 * np.finfo(float).eps * np.linalg.norm(A, 1) * max(A.shape)
    max_iter = max_iter or 30 * n
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        if passive.all() or not (w[~passive] > tol).any():
            break
        passive[np.argmax(np.where(passive, -np.inf, w))] = True
        for _ in range(max_iter):
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if (z[passive] > tol).all():
                break
            neg = passive & (z <= tol)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            passive &= x > tol
        x = z
        w = A.T @ (b - A @ x)
    return x


def signed_lstsq(A: np.ndarray, b: np.ndarray, signs: list[str]) -> np.ndarray:
    """Least squares where each coefficient is positive, negative or free.

    Free coefficients are projected out so the rest is a plain NNLS problem,
    then recovered by ordinary least squares on the residual.
    """
    flip = np.array([-1.0 if s == NEGATIVE else 1.0 for s in signs])
    A = A * flip
    free = np.array([s == FREE for s in signs])
    # Scale columns for conditioning; positive scaling keeps the signs.
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    A = A / norms

    x = np.zeros(A.shape[1])
    A_f, A_c = A[:, free], A[:, ~free]
    if free.any():
        u, s, _ = np.linalg.svd(A_f, full_matrices=False)
        q = u[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else u[:, :0]

        def project(m: np.ndarray) -> np.ndarray:
            return m - q @ (q.T @ m)

        x_c = nnls(project(A_c), project(b)) if A_c.shape[1] else np.zeros(0)
        x_f = np.linalg.lstsq(A_f, b - A_c @ x_c, rcond=None)[0]
        x[free], x[~free] = x_f, x_c
    else:
        x = nnls(A, b)
    return x / norms * flip
