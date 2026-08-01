"""Devigging decimal odds into fair probabilities, and turning a fair
probability into the highest price that still clears an EV threshold.

Context
-------
"Devigging" (removing the vig / overround) is the act of stripping a
bookmaker's margin out of quoted odds so that what remains is an estimate
of the *true* probability the bookmaker believes in.

This is generic two-way devigging math. It is written for the workflow of
anchoring on a **sharp book** rather than a consensus line — Pinnacle is the
canonical example of such a book. The distinction matters:

  * A *consensus* line averages many books. It is dominated by soft books
    that shade their prices to balance action or to exploit recreational
    bettors, so the average is biased and slow.
  * A *sharp* book (Pinnacle being the usual example) runs a low margin,
    welcomes winning bettors, and moves on sharp money. Its closing line is,
    empirically, one of the best public predictors of outcome frequency. So
    instead of blending it away, we take the sharp book's number as the
    reference "truth" and ask a single question: does another venue offer a
    price that beats that fair value by enough to be worth the risk?

Everything here is pure: no I/O, no network, no clock, no Qt, no keys.
That is deliberate — the probability math is the part worth trusting and
testing in isolation, independent of how odds are fetched or how orders
are placed.
"""

from __future__ import annotations

import math
from typing import Literal

Method = Literal["additive", "multiplicative", "power"]

__all__ = [
    "devig",
    "overround",
    "required_price",
    "power_exponent",
    "ConvergenceError",
    "Method",
]


class ConvergenceError(RuntimeError):
    """Raised when the power-method root solve fails to bracket or converge.

    Kept distinct from ``ValueError`` (which signals bad *input*) so callers
    can tell "you gave me nonsense" apart from "the numerics ran out of
    iterations on otherwise-valid input".
    """


def _check_odds(value: float, name: str) -> float:
    """Validate a single decimal-odds input.

    Decimal odds represent total return per unit staked, so anything at or
    below 1.0 implies a non-positive payout and cannot be a real quote. NaN
    and infinities are rejected up front because they would silently poison
    every downstream division rather than failing loudly here.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 1.0:
        raise ValueError(f"{name} must be > 1.0 (decimal odds), got {v!r}")
    return v


def overround(odds_a: float, odds_b: float) -> float:
    """Return the book's overround (a.k.a. vig, juice) for a two-way market.

    The implied probability of a decimal quote is ``1 / odds``. For a fair,
    margin-free book the two implied probabilities sum to exactly 1.0. Real
    books quote them to sum to *more* than 1.0; the excess is the margin the
    book keeps. So ``overround`` is simply that sum:

        overround = 1/odds_a + 1/odds_b

    A value of 1.03 means a 3% book margin. It is the quantity every
    devigging method has to redistribute or remove.
    """
    a = _check_odds(odds_a, "odds_a")
    b = _check_odds(odds_b, "odds_b")
    return (1.0 / a) + (1.0 / b)


def _solve_power_k(
    imp_a: float,
    imp_b: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Find k > 0 with ``imp_a**k + imp_b**k == 1`` by bracketing + bisection.

    The sum ``s(k) = imp_a**k + imp_b**k`` is *strictly decreasing* in k
    whenever both implied probabilities are in (0, 1) — which valid decimal
    odds guarantee — because each term ``imp**k`` decays as k grows. That
    monotonicity is what makes a simple bracket-and-bisect correct and
    guaranteed to converge: there is at most one root, and once we have a k
    where s < 1 and a k where s > 1, the root is trapped between them.

    ``s(1) - 1`` is exactly the overround minus one:
      * == 0  -> the book is already fair, the exponent is 1.0.
      * > 0   -> a real (vigged) book; the root lies at k > 1, so we start
                 the upper bound at 2 and double it until s drops below 1.
      * < 0   -> the two implied probabilities sum to *less* than 1, i.e. an
                 underround / arbitrage, not a booked market. Power devig is
                 not defined for it here, so we reject it rather than solve a
                 k < 1 that would silently mean something different.
    """
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a positive finite number, got {tolerance!r}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")

    def residual(k: float) -> float:
        return imp_a ** k + imp_b ** k - 1.0

    r1 = residual(1.0)
    if abs(r1) <= tolerance:
        return 1.0
    if r1 < 0.0:
        raise ValueError(
            "power devig requires overround >= 1 (a booked two-way market); "
            "the implied probabilities sum to less than 1 (an underround / "
            "arbitrage), which is outside this method's scope"
        )

    # r1 > 0: the root is at k > 1. Expand the upper bound geometrically.
    k_hi = 2.0
    bracketed = False
    for _ in range(max_iterations):
        if residual(k_hi) < 0.0:
            bracketed = True
            break
        k_hi *= 2.0
    if not bracketed:
        raise ConvergenceError(
            f"could not bracket the power exponent within {max_iterations} "
            f"expansions (reached k_hi={k_hi})"
        )

    # Bisection on [1, k_hi]: residual(1) > 0, residual(k_hi) < 0.
    k_lo = 1.0
    k_mid = 0.5 * (k_lo + k_hi)
    for _ in range(max_iterations):
        k_mid = 0.5 * (k_lo + k_hi)
        r_mid = residual(k_mid)
        if abs(r_mid) <= tolerance:
            return k_mid
        if r_mid > 0.0:
            k_lo = k_mid
        else:
            k_hi = k_mid
    raise ConvergenceError(
        f"power exponent did not reach tolerance {tolerance} within "
        f"{max_iterations} bisections (k~={k_mid}, residual={residual(k_mid)})"
    )


def power_exponent(
    odds_a: float,
    odds_b: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Return the single exponent ``k`` that power-devigs this odds pair.

    Exposed separately from :func:`devig` so the numerical result itself is
    inspectable and testable: ``k`` equals 1.0 for a fair book and rises
    above 1.0 as the overround grows. See :func:`_solve_power_k` for the
    solve and :func:`devig` (``method="power"``) for the reasoning behind the
    method.

    Raises
    ------
    ValueError
        For invalid odds, a non-positive ``tolerance``, ``max_iterations`` < 1,
        or an underround (implied sum < 1).
    ConvergenceError
        If the solver cannot bracket or converge within ``max_iterations``.
    """
    imp_a = 1.0 / _check_odds(odds_a, "odds_a")
    imp_b = 1.0 / _check_odds(odds_b, "odds_b")
    return _solve_power_k(imp_a, imp_b, tolerance, max_iterations)


def devig(
    odds_a: float,
    odds_b: float,
    method: Method = "multiplicative",
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> tuple[float, float]:
    """Strip the margin out of a two-way decimal-odds pair.

    Returns ``(prob_a, prob_b)``, the fair probabilities, which sum to 1.0
    (within floating-point tolerance) for every supported method.

    The methods differ only in *how* the overround is removed, and that
    difference encodes a different assumption about where a book hides its
    margin — which is exactly the favourite–longshot question:

    ``method="multiplicative"`` (default)
        Divide each implied probability by the overround:

            prob_i = (1/odds_i) / overround

        This scales both sides down proportionally, so in *absolute* terms it
        removes more margin from the favourite (the larger implied
        probability) than from the longshot. It is the right default when you
        believe the book's margin is loaded roughly in proportion to
        probability. It tends to *under-correct* the favourite–longshot bias,
        i.e. it leaves longshots looking a little more likely than they truly
        are.

    ``method="additive"``
        Remove an equal *absolute* share of the excess from each side:

            excess = overround - 1
            prob_i = (1/odds_i) - excess/2

        This assumes the margin is a flat amount spread evenly across
        outcomes. For a two-outcome book the subtraction collapses to a
        clean closed form (``imp_i = 1/odds_i``):

            prob_b = (1 + imp_b - imp_a) / 2      (and symmetrically prob_a)

        Because valid decimal odds force ``imp_a < 1`` and ``imp_b > 0``, that
        numerator is strictly positive, and by symmetry each probability is
        below 1 — so additive devigging on a two-way market *always* returns
        a valid probability in (0, 1). There is no lopsided-book case that
        drives a side negative here; that degeneracy is real but only for
        three-or-more-outcome markets, where an equal split of the excess can
        push a thin outcome below zero. Relative to multiplicative, additive
        leaves the favourite with a higher fair probability and the longshot
        with a lower one, so it corrects favourite–longshot bias more
        aggressively.

    ``method="power"``
        Find a single exponent ``k > 0`` such that

            imp_a**k + imp_b**k = 1

        and return ``(imp_a**k, imp_b**k)``. ``k`` is solved numerically
        (bracket + bisection, no SciPy) via :func:`_solve_power_k`, and comes
        out as exactly 1.0 for a fair book and above 1.0 once there is vig.

        Why raise both sides to a *power* rather than shift or scale them?
        Because a common exponent shrinks the two implied probabilities by
        different *relative* amounts: the relative change of ``imp**k`` with
        k is governed by ``ln(imp)``, and ``|ln(imp)|`` is larger for the
        smaller number. So power devigging pulls the longshot down harder
        than the favourite, in proportion to how much of a longshot it is.
        That is precisely the shape of the empirical favourite–longshot bias
        (longshots are systematically over-bet and thus over-priced), so
        power devigging *models that bias explicitly* instead of assuming the
        margin is flat (additive) or proportional (multiplicative). It is the
        method most commonly used to devig sharp two-way lines.

    None of these methods is "correct" in an absolute sense; they are three
    priors about margin structure, from cheapest to most bias-aware. Pick
    multiplicative when you trust proportional margin, additive for a flat
    spread, and power when you specifically want to correct favourite–longshot
    bias.

    Parameters
    ----------
    tolerance, max_iterations
        Only used by ``method="power"``; control the root solve. Ignored by
        the other methods.

    Raises
    ------
    ValueError
        If either odds value is non-finite or <= 1.0, if ``method`` is not a
        supported string, or (power only) if the solver arguments are invalid
        or the market is an underround.
    ConvergenceError
        If ``method="power"`` and the solver fails to converge.
    """
    imp_a = 1.0 / _check_odds(odds_a, "odds_a")
    imp_b = 1.0 / _check_odds(odds_b, "odds_b")
    book = imp_a + imp_b

    if method == "multiplicative":
        return imp_a / book, imp_b / book
    if method == "additive":
        excess = book - 1.0
        return imp_a - excess / 2.0, imp_b - excess / 2.0
    if method == "power":
        k = _solve_power_k(imp_a, imp_b, tolerance, max_iterations)
        return imp_a ** k, imp_b ** k
    raise ValueError(
        f"unknown method {method!r}; expected 'additive', 'multiplicative', "
        f"or 'power'"
    )


def required_price(fair_prob: float, edge: float) -> float:
    """Highest price that still clears an EV edge over the fair probability.

    On a $1-notional binary contract (pays $1 if the outcome hits), buying a
    share at price ``P`` when the fair probability is ``q`` returns, per unit
    staked:

        return_on_stake = q / P - 1

    Requiring that this reach a target ``edge`` (expressed as a fraction of
    stake, so 0.05 is +5% EV) and solving for the break-even price gives:

        P_max = q / (1 + edge)

    So ``required_price`` answers: "given the sharp book's fair value ``q``, what is
    the most I can pay on the other venue and still expect ``edge`` of
    profit per dollar risked?" Any resting price at or below the returned
    value is a trade worth taking; anything above it is not. ``edge=0``
    returns ``fair_prob`` itself — the zero-EV break-even.

    Raises
    ------
    ValueError
        If ``fair_prob`` is non-finite or outside [0, 1], or if ``edge`` is
        non-finite or <= -1 (which would make the price non-positive).
    """
    q = float(fair_prob)
    e = float(edge)
    if not math.isfinite(q):
        raise ValueError(f"fair_prob must be finite, got {fair_prob!r}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"fair_prob must be in [0, 1], got {q!r}")
    if not math.isfinite(e):
        raise ValueError(f"edge must be finite, got {edge!r}")
    if e <= -1.0:
        raise ValueError(f"edge must be > -1, got {e!r}")
    return q / (1.0 + e)
