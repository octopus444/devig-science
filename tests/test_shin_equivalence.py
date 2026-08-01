"""Verify that Shin's method equals the additive method on two-way books.

The equivalence is proven analytically in ``_shin_ref.py``; here we confirm
it numerically by solving Shin independently (bisection on its own
constraint) and comparing to the library's additive output across a wide
grid of odds.

The test only has teeth if the grid contains pairs where additive and
multiplicative visibly disagree — otherwise "Shin == additive" and
"Shin == multiplicative" would be indistinguishable. ``test_grid_has_power``
asserts that such pairs are present.
"""

import pytest

from devig import devig

from _shin_ref import shin_probabilities

# 20+ pairs, mild vig (near-pick'em) through heavy favourites, plus a few
# with the longshot listed first to exercise argument symmetry.
SHIN_GRID = [
    (1.95, 1.95),
    (1.90, 1.90),
    (2.00, 1.80),
    (1.80, 2.00),
    (1.90, 1.95),
    (1.70, 2.20),
    (1.60, 2.40),
    (1.50, 2.50),
    (2.50, 1.50),
    (1.45, 2.80),
    (1.40, 3.00),
    (1.35, 3.20),
    (1.33, 3.75),
    (1.30, 3.50),
    (1.25, 4.50),
    (1.20, 6.00),
    (1.15, 7.50),
    (1.10, 8.00),
    (1.10, 11.0),
    (1.08, 9.00),
    (1.05, 10.0),
    (1.05, 12.0),
    (2.10, 1.75),
]

# A pair chosen so additive and multiplicative differ a lot: if the Shin
# solver were secretly reproducing multiplicative (the classic way this test
# could be fooled), it would miss additive by ~0.02, far above 1e-6.
HIGH_SIGNAL_PAIR = (1.05, 10.0)


def test_grid_is_large_enough():
    assert len(SHIN_GRID) >= 20


@pytest.mark.parametrize("odds_a, odds_b", SHIN_GRID)
def test_shin_equals_additive(odds_a, odds_b):
    shin_a, shin_b = shin_probabilities(odds_a, odds_b)
    add_a, add_b = devig(odds_a, odds_b, method="additive")
    assert shin_a == pytest.approx(add_a, abs=1e-6)
    assert shin_b == pytest.approx(add_b, abs=1e-6)


def test_grid_has_power():
    # There must exist a pair where additive and multiplicative differ by far
    # more than the 1e-6 equivalence tolerance, so matching additive is a
    # meaningful result and not trivially true.
    gaps = [
        abs(devig(a, b, method="additive")[0] - devig(a, b, method="multiplicative")[0])
        for a, b in SHIN_GRID
    ]
    assert max(gaps) > 1e-3


def test_high_signal_pair_matches_additive_not_multiplicative():
    a, b = HIGH_SIGNAL_PAIR
    shin_a, _ = shin_probabilities(a, b)
    add_a, _ = devig(a, b, method="additive")
    mul_a, _ = devig(a, b, method="multiplicative")

    # additive and multiplicative are genuinely far apart here...
    assert abs(add_a - mul_a) > 1e-2
    # ...and Shin sits on additive, not multiplicative.
    assert shin_a == pytest.approx(add_a, abs=1e-6)
    assert abs(shin_a - mul_a) > 1e-2


@pytest.mark.parametrize("odds_a, odds_b", SHIN_GRID)
def test_shin_probabilities_are_a_valid_distribution(odds_a, odds_b):
    p_a, p_b = shin_probabilities(odds_a, odds_b)
    assert 0.0 < p_a < 1.0
    assert 0.0 < p_b < 1.0
    assert p_a + p_b == pytest.approx(1.0, abs=1e-9)
