"""Pluggable forecast basis for Fast-SAM3D's cached step-skipping.

Fast-SAM3D forecasts a cached feature at a *skipped* sampling step with a Taylor
expansion ``sum_i (1/i!) * Delta^i * x^i`` (the monomial basis ``x^i`` — TaylorSeer,
inherited from Fast-TRELLIS). This module lets that basis be swapped for the
**dual-scaled physicist's Hermite** basis ``Htilde_i(-x)`` (**HiCache**,
arXiv:2508.16984 — the basis used by faster-trellis / hermit-trellis2):

    Htilde_n(x) = sigma^n * H_n(sigma * x),   sigma in (0, 1)
    H_0 = 1,  H_1 = 2x,  H_{k+1} = 2x H_k - 2k H_{k-1}

Same cached derivatives ``Delta^i``, same ``1/i!`` weights — ONLY the basis function
changes. Order 0 is identical (``x^0 == Htilde_0 == 1``), so the cached-value reuse
term is unchanged; the dual scaling bounds the high-order terms, a strictly more
stable forecast than the monomial. Our TRELLIS ablation showed Hermite wins on
CD / F1 at equal cost — this ports that win onto Fast-SAM3D with no other change.

Select at runtime (no code edit), so the original TaylorSeer stays the default:

    GF_FORECAST_BASIS = taylor | hermite      (default: taylor)
    GF_HERMITE_SIGMA  = float in (0, 1)       (default: 0.5)
"""
import math
import os


def physicists_hermite(n: int, x: float) -> float:
    """Physicist's Hermite ``H_n(x)`` via the stable recurrence (scalar)."""
    if n <= 0:
        return 1.0
    h_prev, h_curr = 1.0, 2.0 * x
    for k in range(1, n):
        h_prev, h_curr = h_curr, 2.0 * x * h_curr - 2.0 * k * h_prev
    return h_curr


def forecast_basis() -> str:
    return os.environ.get("GF_FORECAST_BASIS", "taylor").lower()


def basis_term(i: int, x_dist) -> float:
    """Basis value for derivative order ``i`` at forward distance ``x_dist`` (number of
    steps since the last compute step). Drop-in replacement for ``x_dist ** i``.

    taylor  -> x_dist ** i            (monomial, the original)
    hermite -> Htilde_i(-x_dist)      (HiCache; order-0 == 1 == x_dist**0)
    """
    if forecast_basis() == "hermite":
        sigma = float(os.environ.get("GF_HERMITE_SIGMA", "0.5"))
        return (sigma ** i) * physicists_hermite(i, sigma * (-float(x_dist)))
    return x_dist ** i
