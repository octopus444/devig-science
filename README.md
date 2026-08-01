# devig-science

Small, pure-Python library for **removing a bookmaker's margin from two-way
decimal odds** and turning the resulting fair probability into a maximum
tradeable price. The math is generic — no book-specific code.

No I/O, no network, no clock, no framework. Just the probability math, fully
type-hinted and tested — the part that is worth trusting in isolation.

## The problem

A bookmaker never quotes the true probability of an event. It quotes odds
whose implied probabilities deliberately sum to **more** than 100%. That
excess is the *overround* (a.k.a. the vig or juice) — the book's margin.

To reason about value you first have to remove that margin and recover an
estimate of the true probabilities. This is **devigging**.

The intended workflow anchors on a **sharp book** as the reference line.
Pinnacle is the canonical example — it runs a thin margin and does not limit
winning bettors, so its line is one of the best public estimators of real
outcome frequency — but nothing in the library is specific to it. The
workflow is:

1. Take the sharp book's two-way decimal odds.
2. Devig them into a fair probability for each side.
3. Compute the highest price another venue could offer at which the trade
   still clears a required EV edge.

Steps 1–3 map to the core functions in `devig.py` — `overround`, `devig`,
and `required_price` — with `power_exponent` exposing the power method's
numerical solve on its own.

## The formulas (plain math)

Let the two decimal quotes be `A` and `B`. Their **implied probabilities**
are `1/A` and `1/B`.

**Overround** — the total book, and the quantity to remove:

```
overround = 1/A + 1/B
```

For a fair, margin-free book this equals `1.0`. A value of `1.03` means a
3% margin.

**Multiplicative devig** (default) — scale both implied probabilities down
by the overround:

```
prob_A = (1/A) / overround
prob_B = (1/B) / overround
```

**Additive devig** — remove an equal absolute slice of the excess from each
side:

```
excess = overround - 1
prob_A = (1/A) - excess/2
prob_B = (1/B) - excess/2
```

**Power devig** — find a single exponent `k > 0` such that the powered
implied probabilities renormalise themselves to 1:

```
solve for k:   (1/A)**k + (1/B)**k = 1
prob_A = (1/A)**k
prob_B = (1/B)**k
```

`k` is `1.0` for a fair book and grows above `1.0` as the margin grows.
There is no closed form, so it is solved numerically (see below).

All three methods return probabilities that sum to `1.0`.

**Required price** — given a fair probability `q` and a demanded edge `e`
(as a fraction of stake), the highest price you can pay on a $1-notional
binary contract and still expect `e` profit per dollar risked:

```
required_price = q / (1 + e)
```

Any resting price at or below `required_price` is a trade worth taking;
`e = 0` returns `q` itself (the zero-EV break-even).

## The three methods — a worked example

Take odds **1.50** and **2.50**.

```
1/1.50 = 0.66667
1/2.50 = 0.40000
overround = 1.06667   (a 6.67% margin)
excess    = 0.06667
```

| Method             | favourite (A) | longshot (B) |
|--------------------|--------------:|-------------:|
| multiplicative     | 0.62500       | 0.37500      |
| additive (= Shin)  | 0.63333       | 0.36667      |
| power (k≈1.109)    | 0.63792       | 0.36208      |

All three sum to 1.0, but they disagree about **where the margin lived** —
and they line up in a clear order for the favourite,
`multiplicative < additive < power`, from the mildest to the most aggressive
favourite–longshot correction:

- **Multiplicative** scales proportionally, so in absolute terms it strips
  *more* margin from the favourite. It leaves the longshot looking slightly
  more probable. Use it when you believe the book loads its margin roughly
  in proportion to probability. It tends to **under-correct** the
  favourite–longshot bias.
- **Additive** subtracts the same absolute amount from each side, leaving
  the favourite higher and the longshot lower. It corrects the
  favourite–longshot bias more aggressively. On a two-way market it has a
  clean closed form,

  ```
  prob_b = (1 + imp_b - imp_a) / 2        (imp_i = 1/odds_i)
  ```

  whose numerator is strictly positive for any valid odds (`imp_a < 1`,
  `imp_b > 0`), so additive devigging here **always** yields a probability
  in `(0, 1)`. The often-repeated "additive devig can go negative" warning
  is real only for three-plus-outcome markets, where an equal split of the
  excess can push a thin outcome below zero — it cannot happen on two
  outcomes.
- **Power** raises both implied probabilities to a common exponent `k`. The
  exponent acts on the two numbers by *different relative amounts* — the
  relative effect scales with `ln(imp)`, which is larger for the smaller
  (longshot) number — so it shrinks the longshot harder than the favourite.
  That is the exact shape of the empirical favourite–longshot bias, so power
  devigging **models the bias explicitly** rather than assuming a flat or
  proportional margin. Here `k ≈ 1.109` pushes the favourite to `0.63792`,
  the highest of the three, which is why it is the standard choice for
  devigging sharp two-way lines.

None is "correct"; they are three transparent priors about margin structure,
from cheapest to most bias-aware. Pick the one whose assumption matches the
market you are pricing.

### Ordered by how hard they correct the bias

Ranking the favourite's fair probability from the mildest to the most
aggressive favourite–longshot correction (same 1.50 / 2.50 example):

```
multiplicative    0.62500   margin assumed proportional, under-corrects
additive (= Shin) 0.63333   margin assumed flat
power             0.63792   bias corrected explicitly, most aggressive
```

This ordering — `multiplicative < additive < power` for the favourite — is a
**general property of any book with a positive overround and two distinct
sides**, not a coincidence of this example. (On a symmetric book the three
collapse to the same 0.5.) It is verified across the whole odds grid by
`test_favourite_ordering_is_general` in `tests/test_devig.py`.

### The numerical solve

Power devig has no closed form, so `k` is found by a small hand-rolled
root finder (no SciPy dependency):

- The function `s(k) = (1/A)**k + (1/B)**k` is **strictly decreasing** in `k`
  because each base is below 1, so there is at most one root and simple
  bracketing is safe.
- `s(1) - 1` is just the overround minus one. If it is `0` the book is fair
  and `k = 1`; if it is positive the root lies at `k > 1`, so the upper
  bracket starts at 2 and doubles until `s` drops below 1; a negative value
  would mean an underround (arbitrage) and is rejected as out of scope.
- The root is then pinned by **bisection** between `1` and that upper bound.
  `tolerance` and `max_iterations` are tunable; exhausting the iteration
  budget raises `ConvergenceError` rather than returning a wrong number.

## Applying the edge threshold

Suppose the devigged favourite is `q = 0.625` and you require a `2%` edge:

```
required_price(0.625, 0.02) = 0.625 / 1.02 = 0.6127
```

If another venue rests that outcome at or below `0.6127`, the trade clears
your threshold. At `0.63` it does not.

## Why the choice of method matters more at long odds

The three methods differ only in how they redistribute a fixed overround, so
how much they disagree depends entirely on where in the book you look. Near
the centre they nearly coincide; in the tails they diverge sharply. The
following three lines, from a balanced market to an extreme one, show the
progression. All figures are computed by the library; each table gives the
fair probability and fair decimal odds assigned to the **longshot** side,
with the solved power exponent `k` noted per case.

**Case A — balanced.** `1.87 / 1.87`, overround `1.0695`, `k = 1.10737`.

| method         | longshot fair prob | longshot fair odds |
|----------------|-------------------:|-------------------:|
| multiplicative | 0.50000            | 2.00000            |
| additive       | 0.50000            | 2.00000            |
| power          | 0.50000            | 2.00000            |

**Case B — moderate.** `1.50 / 2.40`, overround `1.0833`, `k = 1.13737`.

| method         | longshot fair prob | longshot fair odds |
|----------------|-------------------:|-------------------:|
| multiplicative | 0.38462            | 2.60000            |
| additive       | 0.37500            | 2.66667            |
| power          | 0.36945            | 2.70671            |

**Case C — extreme.** `1.01 / 15.00`, overround `1.0568`, `k = 1.54466`.

| method         | longshot fair prob | longshot fair odds |
|----------------|-------------------:|-------------------:|
| multiplicative | 0.06309            | 15.85149           |
| additive       | 0.03828            | 26.12069           |
| power          | 0.01525            | 65.56356           |

In Case A the three methods return exactly `0.50000` for both sides: at the
centre of the book the choice of prior is irrelevant. In Case B they have
begun to separate, but the longshot's fair odds still fall within about 4% of
each other (`2.60` to `2.71`). In Case C — the *same* two-number input line —
the longshot's fair odds range from `15.85` to `65.56`, a spread of more than
four to one, driven by nothing but the devigging prior.

That spread turns directly into a disagreement about edge. Suppose another
venue quotes the Case C longshot at decimal `16.667`. The measured edge is
`fair_prob * 16.667 - 1`:

| method         | longshot fair prob | edge @ 16.667 |
|----------------|-------------------:|--------------:|
| multiplicative | 0.06309            | +5.1%         |
| additive       | 0.03828            | −36.2%        |
| power          | 0.01525            | −74.6%        |

The same trade is marginally positive under one prior and heavily negative
under the other two, with nothing changed but the method.

The reason is arithmetic. In the tail the longshot carries only a sliver of
the book, so reallocating a fixed slice of overround is a small *absolute*
change to a small number — and therefore a large *relative* change, which is
exactly what fair odds (the reciprocal of probability) magnify. At the centre
each side holds roughly half the book, the reallocation is a rounding error
by comparison, and the reciprocal barely moves.

The practical consequence is blunt: method choice is near-irrelevant in the
middle of the book and dominant in the tails. An edge figure reported on long
odds is meaningless unless the devigging method is named alongside it, and any
approach that trades long odds has to justify its prior rather than inherit
whatever default a tool happens to ship.

## Usage

```python
from devig import devig, overround, required_price

prob_a, prob_b = devig(1.50, 2.50, method="multiplicative")
q = prob_a                       # 0.625
max_price = required_price(q, 0.02)   # 0.6127...

# power devig, with the numerical solve exposed if you want the exponent
from devig import power_exponent
prob_a, prob_b = devig(1.50, 2.50, method="power")   # 0.63792, 0.36208
k = power_exponent(1.50, 2.50)                        # ~1.109
```

## Shin's method — a proven special case

Shin's method devigs under an explicit model of insider trading: it assumes a
fraction `z` of the money is perfectly informed and backs out the true
probabilities the book must be defending against. It is often presented as a
more sophisticated alternative to the flat/proportional methods above.

**On a two-outcome market it is not a distinct method at all — it reduces
exactly to additive devigging.** Solving Shin's insider fraction `z` from its
own constraint and applying its estimator returns the same fair
probabilities as `additive`, to machine precision (worst-case ~1e-16 across a
wide grid). This is an analytic identity, not a numerical near-miss; the
derivation is written out in `tests/_shin_ref.py`, and
`tests/test_shin_equivalence.py` confirms it numerically against an
independent Shin solver, including on heavy-favourite books where additive
and multiplicative differ substantially (so the test genuinely distinguishes
them).

That is exactly why this library does **not** ship a fourth `shin` method: it
would be a second name for `additive`. The equivalence breaks for three or
more outcomes, where Shin and additive genuinely diverge — see future work.

## Limitations

This is intentionally a small, honest core, not a full pricing engine:

- **Two-outcome markets only.** The additive and multiplicative formulas
  here are written for a two-way book. Three-plus-way markets (draws,
  multi-runner fields) need the n-way generalisation — and there additive
  devigging genuinely *can* drive a thin outcome's probability negative, so
  it needs a more careful (clamped or renormalised) margin split. On two
  outcomes that failure mode does not exist (see above).
- **No live-odds latency handling.** The math is a pure snapshot. It says
  nothing about staleness, line movement between read and execution, or
  which of two disagreeing quotes to trust.
- **No model of the reference book's own margin structure across sports.**
  Real margins are not uniform — they vary by sport, league, and market type,
  and are not applied identically to favourites and longshots. Treating the
  overround as a single number to split evenly (additive), proportionally
  (multiplicative), or by a single exponent (power) is still a
  simplification of that structure.
- **No selection, sizing, or execution logic.** This library decides
  neither *what* to trade nor *how much* — only what a price is worth once
  you already have a fair probability.

## Possible future work

- **n-way generalisation** — extending all methods to markets with three or
  more outcomes (draws, multi-runner fields). This is also where Shin stops
  coinciding with additive and becomes a genuinely distinct estimator, and
  where additive needs clamping to stay non-negative.

## Development

```bash
pip install -r requirements.txt
pytest
```

## License

MIT
