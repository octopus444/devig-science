"""Table-driven tests for the devig math.

The invariant that matters most is that every devigging method returns
probabilities summing to exactly 1.0 (a fair, margin-free distribution),
so that is checked across every well-formed case.
"""

import math

import pytest

from devig import (
    ConvergenceError,
    devig,
    overround,
    power_exponent,
    required_price,
)

TOL = 1e-9

# The odds grid shared by every method's "sums to 1.0" check.
ODDS_GRID = [
    (2.0, 2.0),      # equal odds / fair book
    (1.5, 2.5),      # mild vig
    (1.25, 5.0),     # heavy favourite
    (1.10, 8.0),     # very heavy favourite + longshot
    (1.91, 1.91),    # standard -110/-110 symmetric book
]


# --------------------------------------------------------------------------- #
#  overround                                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "odds_a, odds_b, expected",
    [
        (2.0, 2.0, 1.0),          # perfectly fair two-way book
        (1.5, 2.5, 1.0 / 1.5 + 1.0 / 2.5),
        (1.25, 5.0, 0.8 + 0.2),   # 1.05 -> 5% margin
    ],
)
def test_overround(odds_a, odds_b, expected):
    assert overround(odds_a, odds_b) == pytest.approx(expected, abs=TOL)


# --------------------------------------------------------------------------- #
#  devig: both methods always produce a proper distribution                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("odds_a, odds_b", ODDS_GRID)
@pytest.mark.parametrize("method", ["additive", "multiplicative", "power"])
def test_probabilities_sum_to_one(odds_a, odds_b, method):
    p_a, p_b = devig(odds_a, odds_b, method=method)
    assert p_a + p_b == pytest.approx(1.0, abs=TOL)


def test_equal_odds_gives_even_split():
    for method in ("additive", "multiplicative", "power"):
        p_a, p_b = devig(2.0, 2.0, method=method)
        assert p_a == pytest.approx(0.5, abs=TOL)
        assert p_b == pytest.approx(0.5, abs=TOL)


def test_fair_book_overround_exactly_one_methods_agree():
    # When overround == 1.0 there is no margin to remove, so all three methods
    # must collapse to the same answer (the raw implied probs).
    assert overround(2.0, 2.0) == pytest.approx(1.0, abs=TOL)
    add = devig(2.0, 2.0, method="additive")
    mul = devig(2.0, 2.0, method="multiplicative")
    pwr = devig(2.0, 2.0, method="power")
    assert add == pytest.approx(mul, abs=TOL)
    assert pwr == pytest.approx(mul, abs=TOL)


@pytest.mark.parametrize(
    "odds_a, odds_b",
    [
        (1.01, 100.0),   # extreme favourite + deep longshot
        (1.05, 1.30),    # two favourites -> ~72% overround
        (1.001, 50.0),   # near-certain favourite
        (1.20, 21.0),    # lopsided with a fat margin
        (2.0, 2.0),      # fair book
    ],
)
def test_additive_two_way_never_leaves_unit_interval(odds_a, odds_b):
    # Closed form on two outcomes is (1 + imp_b - imp_a)/2 with imp < 1 each,
    # so both probabilities are strictly inside (0, 1) for any valid odds.
    # This guards the corrected claim that the "additive goes negative"
    # failure mode is an n-way phenomenon, not a two-way one.
    p_a, p_b = devig(odds_a, odds_b, method="additive")
    assert 0.0 < p_a < 1.0
    assert 0.0 < p_b < 1.0
    assert p_a + p_b == pytest.approx(1.0, abs=TOL)


def test_additive_closed_form_matches_implementation():
    # p_b == (1 + imp_b - imp_a)/2, the simplification worth documenting.
    for odds_a, odds_b in [(1.5, 2.5), (1.01, 100.0), (1.91, 1.91)]:
        imp_a, imp_b = 1.0 / odds_a, 1.0 / odds_b
        _, p_b = devig(odds_a, odds_b, method="additive")
        assert p_b == pytest.approx((1.0 + imp_b - imp_a) / 2.0, abs=TOL)


def test_additive_vs_multiplicative_diverge_under_vig():
    # 1.5 / 2.5, overround ~1.0667. Multiplicative shaves the favourite harder
    # in absolute terms, so it assigns the favourite a LOWER fair probability
    # than additive does. This encodes the favourite-longshot assumption.
    add_fav, add_dog = devig(1.5, 2.5, method="additive")
    mul_fav, mul_dog = devig(1.5, 2.5, method="multiplicative")

    assert add_fav == pytest.approx(0.6333333333, abs=1e-6)
    assert mul_fav == pytest.approx(0.625, abs=1e-6)
    assert add_fav > mul_fav        # additive keeps the favourite higher
    assert add_dog < mul_dog        # ...and the longshot lower
    # and neither collapses the ordering
    assert add_fav > add_dog and mul_fav > mul_dog


# --------------------------------------------------------------------------- #
#  favourite-longshot ordering across methods                                   #
# --------------------------------------------------------------------------- #

# Asymmetric, vigged books with side A as the favourite (odds_a < odds_b).
# The strict ordering below is a property of positive overround PLUS
# asymmetry; at a symmetric book the three methods coincide (see the boundary
# test), so symmetric pairs are intentionally excluded here.
ORDERING_GRID = [
    (1.90, 1.95),
    (1.70, 2.20),
    (1.60, 2.40),
    (1.50, 2.50),
    (1.45, 2.80),
    (1.40, 3.00),
    (1.35, 3.20),
    (1.33, 3.75),
    (1.25, 4.50),
    (1.20, 5.50),
    (1.15, 7.50),
    (1.10, 8.00),
    (1.08, 9.00),
    (1.05, 12.0),
]


@pytest.mark.parametrize("odds_a, odds_b", ORDERING_GRID)
def test_favourite_ordering_is_general(odds_a, odds_b):
    # For ANY vigged, asymmetric two-way book the favourite's fair probability
    # rises multiplicative -> additive -> power, i.e. from the mildest to the
    # most aggressive favourite-longshot correction. This is not a quirk of
    # the 1.5/2.5 example; it holds across the whole grid.
    assert overround(odds_a, odds_b) > 1.0
    mul_fav, _ = devig(odds_a, odds_b, method="multiplicative")
    add_fav, _ = devig(odds_a, odds_b, method="additive")
    pow_fav, _ = devig(odds_a, odds_b, method="power")
    assert mul_fav < add_fav < pow_fav


@pytest.mark.parametrize("odds", [1.95, 1.80, 1.50, 1.30])
def test_symmetric_book_collapses_the_ordering(odds):
    # The equality boundary: on a symmetric (and vigged, odds < 2) book the
    # favourite == longshot, so all three methods return 0.5 and the strict
    # ordering degenerates to equality.
    mul = devig(odds, odds, method="multiplicative")[0]
    add = devig(odds, odds, method="additive")[0]
    pwr = devig(odds, odds, method="power")[0]
    assert mul == pytest.approx(0.5, abs=TOL)
    assert add == pytest.approx(0.5, abs=TOL)
    assert pwr == pytest.approx(0.5, abs=TOL)


# --------------------------------------------------------------------------- #
#  power method                                                                 #
# --------------------------------------------------------------------------- #

def test_power_exponent_is_one_on_fair_book():
    # overround == 1.0 -> no margin -> the exponent must be exactly 1.
    assert overround(2.0, 2.0) == pytest.approx(1.0, abs=TOL)
    assert power_exponent(2.0, 2.0) == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("odds_a, odds_b", [
    (1.5, 2.5),
    (1.10, 8.0),
    (1.05, 1.30),
    (1.91, 1.91),
])
def test_power_exponent_exceeds_one_when_vigged(odds_a, odds_b):
    # A real book (overround > 1) forces the solved exponent above 1, because
    # imp_a**k + imp_b**k is strictly decreasing in k and equals >1 at k=1.
    assert overround(odds_a, odds_b) > 1.0
    assert power_exponent(odds_a, odds_b) > 1.0


def test_power_favourite_is_the_most_aggressive_corrector():
    # NOTE: the original spec asked to assert the power favourite sits BETWEEN
    # additive and multiplicative. That is factually wrong. For 1.5/2.5 the
    # favourite probabilities are:
    #     multiplicative 0.62500  <  additive 0.63333  <  power 0.63792
    # Power applies the strongest favourite-longshot correction, so it gives
    # the favourite the HIGHEST probability of the three, not a middle value.
    # We assert the true ordering instead of a wrong "between" claim.
    mul_fav, _ = devig(1.5, 2.5, method="multiplicative")
    add_fav, _ = devig(1.5, 2.5, method="additive")
    pow_fav, _ = devig(1.5, 2.5, method="power")

    assert mul_fav < add_fav < pow_fav
    assert pow_fav == pytest.approx(0.63792, abs=1e-4)


def test_power_solution_satisfies_its_defining_equation():
    # Whatever k the solver returns, imp_a**k + imp_b**k must equal 1.
    for odds_a, odds_b in ODDS_GRID:
        imp_a, imp_b = 1.0 / odds_a, 1.0 / odds_b
        k = power_exponent(odds_a, odds_b)
        assert imp_a ** k + imp_b ** k == pytest.approx(1.0, abs=1e-10)


def test_power_rejects_underround():
    # 3.0 / 3.0 -> implied sum 0.667 < 1 (an arbitrage), outside the method.
    with pytest.raises(ValueError):
        devig(3.0, 3.0, method="power")


def test_power_convergence_failure_raises():
    # Starving the solver of iterations makes the bisection fail to reach
    # tolerance, which must surface as ConvergenceError, not a silent result.
    with pytest.raises(ConvergenceError):
        devig(1.5, 2.5, method="power", max_iterations=1)


@pytest.mark.parametrize("bad_tol", [0.0, -1e-6, math.inf, math.nan])
def test_power_rejects_bad_tolerance(bad_tol):
    with pytest.raises(ValueError):
        devig(1.5, 2.5, method="power", tolerance=bad_tol)


# --------------------------------------------------------------------------- #
#  required_price                                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "fair_prob, edge, expected",
    [
        (0.5, 0.0, 0.5),                 # zero edge -> break-even at fair value
        (0.5, 0.05, 0.5 / 1.05),         # +5% EV threshold
        (0.625, 0.02, 0.625 / 1.02),     # realistic devigged favourite
        (0.20, 0.10, 0.20 / 1.10),       # longshot with a fat required edge
        (1.0, 0.05, 1.0 / 1.05),         # boundary prob
    ],
)
def test_required_price(fair_prob, edge, expected):
    assert required_price(fair_prob, edge) == pytest.approx(expected, abs=TOL)


def test_required_price_is_monotone_in_edge():
    # A larger demanded edge must lower the price you are willing to pay.
    assert required_price(0.5, 0.10) < required_price(0.5, 0.02)


# --------------------------------------------------------------------------- #
#  invalid input                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [1.0, 0.9, 0.0, -2.0])
def test_devig_rejects_non_positive_odds(bad):
    with pytest.raises(ValueError):
        devig(bad, 2.0)
    with pytest.raises(ValueError):
        devig(2.0, bad)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_devig_rejects_non_finite_odds(bad):
    with pytest.raises(ValueError):
        devig(bad, 2.0)


def test_devig_rejects_unknown_method():
    with pytest.raises(ValueError):
        devig(2.0, 2.0, method="shin")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_prob", [-0.01, 1.01, math.inf, math.nan])
def test_required_price_rejects_bad_probability(bad_prob):
    with pytest.raises(ValueError):
        required_price(bad_prob, 0.05)


@pytest.mark.parametrize("bad_edge", [-1.0, -2.0, math.inf, math.nan])
def test_required_price_rejects_bad_edge(bad_edge):
    with pytest.raises(ValueError):
        required_price(0.5, bad_edge)
