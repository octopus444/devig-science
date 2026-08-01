"""Reference Shin (1992/1993) devigger for two-outcome markets — TEST ONLY.

This deliberately lives under ``tests/`` and is NOT part of the public
library. On a two-way book Shin's method returns exactly the same fair
probabilities as the additive method already in ``devig.py`` (derived below,
and checked numerically in ``test_shin_equivalence.py``), so shipping it as a
fourth ``devig`` method would just be a second name for something the library
already computes. It exists here only so the test can confirm the equivalence
by solving Shin *independently* and comparing.

The Shin model
--------------
Shin models a book that faces a fraction ``z`` of insider (perfectly
informed) traders, with ``0 < z < 1``. Writing the raw implied probabilities
as ``imp_i = 1 / odds_i`` and the booksum (overround) as ``B = sum_i imp_i``,
the model's forward relation between a quoted implied probability and the
true probability ``p_i`` is

    imp_i = sqrt( B * p_i * ( z + (1 - z) * p_i ) ),

which inverts to the estimator

    p_i = ( sqrt( z**2 + 4 (1 - z) * imp_i**2 / B ) - z ) / ( 2 (1 - z) ).

``z`` is pinned by requiring the recovered probabilities to sum to 1.
Summing the estimator over the ``n`` outcomes and clearing the denominator:

    sum_i sqrt( z**2 + 4 (1 - z) imp_i**2 / B )
        = 2 (1 - z) + n z
        = 2 + (n - 2) z.

For a two-outcome market (``n = 2``) the right-hand side collapses to a
constant — the ``z`` terms cancel:

    sum_i sqrt( z**2 + 4 (1 - z) imp_i**2 / B ) = 2.                 (C2)

(This is the standard two-outcome Shin constraint. It is exactly ``2`` on the
right, not ``2 - z``; all of the ``z`` dependence sits inside the roots.)

Why it equals the additive method on two outcomes
-------------------------------------------------
Square the estimator (multiply out ``2(1 - z) p_i + z = sqrt(...)`` and
simplify) to get the clean per-outcome identity

    imp_i**2 / B = p_i * ( p_i + z (1 - p_i) ).                      (*)

Now take the additive solution the library uses,
``p_a = (1 + imp_a - imp_b) / 2`` and ``p_b = 1 - p_a``. Subtract (*) for the
two outcomes. Because ``p_a + p_b = 1`` and ``p_a - p_b = imp_a - imp_b``:

    (imp_a**2 - imp_b**2) / B = p_a**2 - p_b**2
                              = (p_a - p_b)(p_a + p_b)
                              = imp_a - imp_b,

and the left side is ``(imp_a - imp_b)(imp_a + imp_b) / B = imp_a - imp_b``
as well, since ``imp_a + imp_b = B``. So the *difference* of the two (*)
equations is an identity — it holds for ANY ``z``. That leaves only their
*sum* to fix a single scalar:

    z = [ (imp_a**2 + imp_b**2)/B - (p_a**2 + p_b**2) ] / ( 2 p_a p_b ),

which lands in ``(0, 1)`` for a booked two-way market. Hence the additive
probabilities satisfy Shin's whole system exactly: the equivalence is
analytic, not a numerical accident. This module never uses that shortcut —
it solves (C2) for ``z`` by bisection from scratch — so the test verifies the
equality from the opposite direction.

Solver note
-----------
``g(z) = sum_i sqrt(z**2 + 4(1-z) imp_i**2 / B) - 2`` satisfies, for ``B > 1``:

  * ``g(0) = 2 (sqrt(B) - 1) > 0``;
  * ``g(z) -> 0`` from below as ``z -> 1``, with ``g'(1) > 0`` because
    ``imp_a**2 + imp_b**2 < imp_a + imp_b = B`` (each ``imp_i < 1``).

So there is a single interior root, bracketed by ``g(0) > 0`` and
``g(1 - eps) < 0``, and plain bisection converges to it. No SciPy.
"""

from __future__ import annotations

import math


def shin_z(
    odds_a: float,
    odds_b: float,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 200,
) -> float:
    """Solve the two-outcome Shin constraint (C2) for the insider fraction z.

    Returns ``z`` in ``[0, 1)``. Returns ``0.0`` for a fair/underround book
    (``B <= 1``), where the estimator degenerates to the raw implied probs.
    """
    imp_a = 1.0 / odds_a
    imp_b = 1.0 / odds_b
    book = imp_a + imp_b

    def g(z: float) -> float:
        return (
            math.sqrt(z * z + 4.0 * (1.0 - z) * imp_a * imp_a / book)
            + math.sqrt(z * z + 4.0 * (1.0 - z) * imp_b * imp_b / book)
            - 2.0
        )

    if g(0.0) <= 0.0:
        return 0.0  # B <= 1: no vig, no insider component to solve for.

    lo, hi = 0.0, 1.0 - 1e-9
    if g(hi) >= 0.0:  # structural guarantee for B > 1; guard anyway.
        raise RuntimeError(
            f"Shin bracket failed for odds {odds_a}/{odds_b}: "
            f"g(0)={g(0.0)!r}, g(hi)={g(hi)!r}"
        )
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) <= tolerance:
            return mid
        if gm > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shin_probabilities(
    odds_a: float,
    odds_b: float,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 200,
) -> tuple[float, float]:
    """Return ``(p_a, p_b)``, the Shin fair probabilities for a two-way book.

    Solves ``z`` via :func:`shin_z`, then applies the Shin estimator. Kept
    fully independent of the additive closed form so the equivalence test has
    something real to compare against.
    """
    imp_a = 1.0 / odds_a
    imp_b = 1.0 / odds_b
    book = imp_a + imp_b
    z = shin_z(odds_a, odds_b, tolerance=tolerance, max_iterations=max_iterations)

    if z <= 0.0:
        return imp_a, imp_b  # degenerate fair book -> raw implied probs.

    def p(imp: float) -> float:
        return (
            math.sqrt(z * z + 4.0 * (1.0 - z) * imp * imp / book) - z
        ) / (2.0 * (1.0 - z))

    return p(imp_a), p(imp_b)
