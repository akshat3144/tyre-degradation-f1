"""Per-circuit degradation rates must reach the strategy layer.

The circuit slopes run from -0.081 s/lap at Suzuka to +0.082 at Barcelona -- a
spread wider than the one separating C1 from C5. Dropping them is not a
rounding difference; it is the single largest term in the model. A strategy
caller that fits without circuit effects and reads the global rate gives every
circuit the season average. These tests pin the ``Fit`` accessors that carry the
circuit term (``rate_for`` / ``interval_for``) and that it moves the optimum.
"""

from __future__ import annotations

import pytest

from tyredeg.artifacts.schema import Interval
from tyredeg.models.mixed import Fit


def make_fit(**kw) -> Fit:
    """A Fit with two compounds and two circuits, one harsh and one kind."""
    base = dict(
        rates={"C3": Interval(0.05, 0.01, 0.09), "C4": Interval(0.03, -0.01, 0.07)},
        curvature={},
        offsets={},
        n_laps={"C3": 100, "C4": 100},
        n_runs={"C3": 10, "C4": 10},
        events={},
        context="race",
        quadratic=False,
        converged=True,
        n_obs=200,
        n_drivers=20,
        circuit_slope={"Harsh": 0.08, "Kind": -0.04},
    )
    base.update(kw)
    return Fit(**base)


# ---------------------------------------------------------------------------
# the Fit API itself
# ---------------------------------------------------------------------------


def test_rate_for_adds_the_circuit_slope():
    f = make_fit()
    assert f.rate_for("C3", "Harsh") == pytest.approx(0.13)
    assert f.rate_for("C3", "Kind") == pytest.approx(0.01)


def test_rate_for_falls_back_to_global_for_an_unknown_circuit():
    """A circuit we have never raced has no slope, and the global rate is the
    honest answer rather than an error."""
    assert make_fit().rate_for("C3", "Nowhere") == pytest.approx(0.05)


def test_interval_for_is_centred_on_rate_for():
    """The number a screen SHOWS has to be the number the optimiser USED.
    Displaying the global band beside a per-circuit rate is what made the
    playbook's compound table contradict its own plan."""
    f = make_fit()
    for ev in ("Harsh", "Kind", "Nowhere"):
        for c in ("C3", "C4"):
            assert f.interval_for(c, ev).mean == pytest.approx(f.rate_for(c, ev))


def test_interval_for_shifts_the_band_and_does_not_narrow_it():
    """The circuit term is a fitted mean shift. It moves the interval; it does
    not make the per-circuit slope better determined than the global one."""
    f = make_fit()
    g, h = f.rates["C3"], f.interval_for("C3", "Harsh")
    assert h.width == pytest.approx(g.width)
    assert h.lo == pytest.approx(g.lo + 0.08)
    assert h.hi == pytest.approx(g.hi + 0.08)


def test_interval_for_stays_ordered():
    """Interval.__post_init__ raises on an unordered band, so a sign error in
    the shift would surface here rather than as a 500 in the API."""
    f = make_fit()
    iv = f.interval_for("C4", "Kind")
    assert iv.lo <= iv.mean <= iv.hi


def test_a_fit_without_circuit_effects_is_identical_everywhere():
    """The failure mode, pinned. This is what the API was doing: a fit with no
    circuit slopes answers every track the same, and nothing in the response
    says so."""
    f = make_fit(circuit_slope={})
    assert f.rate_for("C3", "Harsh") == f.rate_for("C3", "Kind") == 0.05


def test_a_harsh_circuit_shortens_the_best_stint():
    """The consequence that reaches the pit wall. If this does not hold, the
    per-circuit term is not reaching the strategy layer."""
    from tyredeg.strategy.optimise import optimal_stint

    f = make_fit()
    harsh = optimal_stint("C3", f.rate_for("C3", "Harsh"), 22.0)
    kind = optimal_stint("C3", f.rate_for("C3", "Kind"), 22.0)
    assert harsh < kind
