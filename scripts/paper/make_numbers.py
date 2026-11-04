"""Write paper/manuscript/generated/numbers.tex: one \\newcommand per quoted number.

    python scripts/paper/make_numbers.py

Every macro is read from a JSON path (never typed in) and rounded to the same
number of places CLAIMS.md uses for that claim (the ``places`` of the matching
```claims`` check). Macros with no CLAIMS.md check ("extras": ns, cell counts,
the rest of a ladder) state their rounding here. The run fails if a numeric
check in CLAIMS.md section A has no macro, so a new claim cannot be forgotten.

Macro names use letters only (LaTeX); digits are spelled out (``RateCOne``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import re  # noqa: E402

from _paper_common import (  # noqa: E402
    ART_POWER,
    BENCH,
    COMPOUNDS,
    DESIGN,
    GENERATED,
    MGMT,
    MODEL,
    PIT,
    PUB4,
    RECOV,
    STRAT,
    TELE,
    TRANS,
    fmt_num,
    get,
    has,
    section_a_claims,
    shown,
    write_text,
)

PLACEHOLDER_TEXT = r"\textbf{??}"  # shown where a number's JSON key has not been produced yet
NAME_OK = re.compile(r"^[A-Za-z]+$")
WORD = {"C1": "One", "C2": "Two", "C3": "Three", "C4": "Four", "C5": "Five"}
SEASON = {"2022": "TwentyTwo", "2023": "TwentyThree", "2024": "TwentyFour", "2025": "TwentyFive", "2026": "TwentySix"}


@dataclass
class Macro:
    name: str
    text: str
    file: str
    path: tuple
    places: int
    signed: bool
    value: float
    claimed: bool  # backed by a CLAIMS.md check
    absolute: bool = False  # magnitude of a signed JSON value (prose such as "0.056 lower than")
    placeholder: bool = False  # the JSON key does not exist yet: text is PLACEHOLDER_TEXT, value is nan

    def tex(self) -> str:
        return rf"\newcommand{{\{self.name}}}{{{self.text}}}"


def _claim_places() -> tuple[dict, list[dict]]:
    claims = section_a_claims()
    by_key = {}
    for c in claims:
        if "value" in c or c.get("_pending"):
            by_key[(c["file"], tuple(c["path"]))] = int(c.get("places", 3))
    return by_key, claims


def build() -> list[Macro]:
    by_key, claims = _claim_places()
    macros: list[Macro] = []
    names: set[str] = set()
    paths: set[tuple] = set()

    def add(name, f, *path, signed=False, places=None, alias=False, absval=False, placeholder=False):
        key = (f, tuple(path))
        if absval:
            alias = True
        if not NAME_OK.match(name):
            raise ValueError(f"bad macro name {name!r}")
        if name in names:
            raise ValueError(f"duplicate macro name {name}")
        if key in paths and not alias:
            raise ValueError(f"path used twice: {key} ({name})")
        if placeholder:
            v = float("nan")
        else:
            v = get(f, *path)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{name}: {key} is {type(v).__name__}, not a number")
        claimed = key in by_key and not absval
        if places is None:
            if absval:
                raise ValueError(f"{name}: absolute macro needs explicit places")
            if claimed:
                places = by_key[key]
            elif isinstance(v, int):
                places = 0
            else:
                raise ValueError(f"{name}: extra macro on a float needs explicit places")
        elif claimed and by_key[key] != places:
            raise ValueError(f"{name}: places {places} differs from CLAIMS ({by_key[key]})")
        names.add(name)
        paths.add(key)
        if placeholder:
            macros.append(Macro(name, PLACEHOLDER_TEXT, f, tuple(path), places, signed, v, claimed, absval, True))
            return
        shown = abs(v) if absval else v
        macros.append(
            Macro(name, fmt_num(shown, places, signed), f, tuple(path), places, signed, float(v), claimed, absval)
        )

    def new(name, f, *path, **kw):
        """Add an extra only when neither the name nor the path is taken yet."""
        if name in names or (f, tuple(path)) in paths:
            return
        add(name, f, *path, **kw)

    # ---------------------------------------------------------------- A1 circuit variation
    cs = ["circuit_effect", "circuit_slope_sd_s_per_lap"]
    add("CircuitSD", MODEL, *cs, "estimate")
    add("CircuitSDLo", MODEL, *cs, "profile_likelihood_95", 0)
    add("CircuitSDHi", MODEL, *cs, "profile_likelihood_95", 1)
    add("CircuitSDBootLo", MODEL, *cs, "parametric_bootstrap_percentile_95", 0)
    add("CircuitSDBootHi", MODEL, *cs, "parametric_bootstrap_percentile_95", 1)
    t20 = ["circuit_effect", "effect_in_seconds_over_20_lap_stint", "circuit_sd"]
    add("CircuitSDsPerTwenty", MODEL, *t20, "seconds")
    add("CircuitSDsPerTwentyLo", MODEL, *t20, "profile_95", 0)
    add("CircuitSDsPerTwentyHi", MODEL, *t20, "profile_95", 1)
    add("CircuitLRStat", MODEL, "circuit_effect", "lr_test", "reml_lr_statistic")
    add("CircuitBootP", MODEL, "circuit_effect", "lr_test", "p_parametric_bootstrap")
    add("CircuitNBoot", MODEL, "circuit_effect", "lr_test", "n_bootstrap")
    add("CircuitRange", MODEL, "circuit_effect", "shrunken_predictions_not_an_effect_size", "range_s_per_lap", places=3)
    add("NCircuitsRaceFit", MODEL, "dataset", "race_fit", "n_events")
    add("NLapsRaceFit", MODEL, "dataset", "race_fit", "n_laps")
    add("NRunsRaceFit", MODEL, "dataset", "race_fit", "n_runs")
    add("NDriversRaceFit", MODEL, "dataset", "race_fit", "n_drivers")

    # A1, pooled 2022-26 nested model (persistent circuit slope + event-within-circuit slope).
    # The CircuitSD* macros above are the 2026-only fit, where each circuit is raced once, so
    # that SD mixes persistent and race-specific variation; the Persist*/Event* macros separate them.
    cv = ["circuit_vs_event"]
    add("PooledNEvents", MODEL, *cv, "design", "n_events")
    add("PooledNCircuits", MODEL, *cv, "design", "n_circuits")
    add("PooledNRepeat", MODEL, *cv, "design", "circuits_with_repeat_visits")
    add("PooledNLaps", MODEL, *cv, "design", "n_laps")
    add("PooledNRuns", MODEL, *cv, "design", "n_runs")
    new("PooledEventsPerCircuit", MODEL, *cv, "design", "events_per_circuit_median", places=0)
    for nm, key in (("Persist", "circuit_persistent"), ("Event", "event_specific")):
        sdp = [*cv, "sd_s_per_lap", key]
        add(f"{nm}SD", MODEL, *sdp, "estimate", places=4)
        add(f"{nm}SDLo", MODEL, *sdp, "profile_95", 0, places=4)
        add(f"{nm}SDHi", MODEL, *sdp, "profile_95", 1, places=4)
        add(f"{nm}SDBootLo", MODEL, *sdp, "bootstrap_95", 0, places=4)
        add(f"{nm}SDBootHi", MODEL, *sdp, "bootstrap_95", 1, places=4)
    sh = [*cv, "persistent_share_of_slope_variance"]
    add("PersistShare", MODEL, *sh, "estimate", places=2)
    add("PersistShareBootLo", MODEL, *sh, "bootstrap_95", 0, places=2)
    add("PersistShareBootHi", MODEL, *sh, "bootstrap_95", 1, places=2)
    s20 = [*cv, "seconds_over_20_laps"]
    add("PersistSDsPerTwenty", MODEL, *s20, "circuit_persistent", places=2)
    add("PersistSDsPerTwentyLo", MODEL, *s20, "circuit_persistent_profile_95", 0, places=2)
    add("PersistSDsPerTwentyHi", MODEL, *s20, "circuit_persistent_profile_95", 1, places=2)
    add("EventSDsPerTwenty", MODEL, *s20, "event_specific", places=2)
    add("EventSDsPerTwentyLo", MODEL, *s20, "event_specific_profile_95", 0, places=2)
    add("EventSDsPerTwentyHi", MODEL, *s20, "event_specific_profile_95", 1, places=2)
    add("PooledLRCircuit", MODEL, *cv, "circuit_component_test", "reml_lr", places=1)
    add("PooledPCircuit", MODEL, *cv, "circuit_component_test", "p_parametric_bootstrap", places=3)
    add("PooledNBootCircuit", MODEL, *cv, "circuit_component_test", "n_bootstrap")
    add("PooledLREvent", MODEL, *cv, "event_component_test", "reml_lr", places=1)
    for c in COMPOUNDS:
        add(f"PooledRateC{WORD[c]}", MODEL, *cv, "compound_rates_pooled_s_per_lap", c, "mean", places=3)
    add("PooledCompoundSD", MODEL, *cv, "compound_rate_sd_s_per_lap", places=4)
    cvv = [*cv, "compound_vs_variance"]
    add("PooledCompoundSDBootLo", MODEL, *cvv, "compound_sd_raw", "bootstrap_95", 0, places=4)
    add("PooledCompoundSDBootHi", MODEL, *cvv, "compound_sd_raw", "bootstrap_95", 1, places=4)
    add("PooledCompoundSDCorrHi", MODEL, *cvv, "compound_sd_noise_corrected", "bootstrap_95", 1, places=4)
    add("CircMinusCompound", MODEL, *cvv, "circuit_minus_compound_raw", "median", places=4)
    add("CircMinusCompoundLo", MODEL, *cvv, "circuit_minus_compound_raw", "bootstrap_95", 0, places=4)
    add("CircMinusCompoundHi", MODEL, *cvv, "circuit_minus_compound_raw", "bootstrap_95", 1, places=4)
    add("CircGreaterCompoundP", MODEL, *cvv, "circuit_minus_compound_raw", "p_circuit_greater", places=3)
    add("SensNoTwentyFiveCircuitSD", MODEL, *cv, "sensitivity", "without_2025", "circuit_sd", places=4)
    add("SensVerifiedCircuitSD", MODEL, *cv, "sensitivity", "verified_nominations_only", "circuit_sd", places=4)
    add("SensVerifiedNEvents", MODEL, *cv, "sensitivity", "verified_nominations_only", "n_events")

    # ---------------------------------------------------------------- A2 compound rates
    rates = ["shipped_fit", "rates_s_per_lap"]
    for c in COMPOUNDS:
        w = WORD[c]
        add(f"RateC{w}", MODEL, *rates, c, "mean")
        add(f"RateC{w}Lo", MODEL, *rates, c, "lo")
        add(f"RateC{w}Hi", MODEL, *rates, c, "hi")
        add(f"RateC{w}N", MODEL, *rates, c, "n_runs")
        add(f"RateC{w}NLaps", MODEL, *rates, c, "n_laps")
    ac = ["shipped_fit", "adjacent_contrasts_s_per_lap", "C5-C4"]
    add("ContrastCFiveCFour", MODEL, *ac, "mean", signed=True)
    add("ContrastCFiveCFourLo", MODEL, *ac, "lo", signed=True)
    add("ContrastCFiveCFourHi", MODEL, *ac, "hi", signed=True)
    we = ["compound_within_event", "within_event_fixed_circuit", "trend_s_per_lap_per_step_softer"]
    add("WithinEventTrend", MODEL, *we, "mean", signed=True)
    add("WithinEventTrendLo", MODEL, *we, "lo", signed=True)
    add("WithinEventTrendHi", MODEL, *we, "hi", signed=True)
    s22 = ["season_robustness", "2022", "trend_per_step_softer", "within_event"]
    add("TrendSeasonTwentyTwo", MODEL, *s22, "mean", signed=True)
    add("TrendSeasonTwentyTwoLo", MODEL, *s22, "lo", signed=True)
    add("TrendSeasonTwentyTwoHi", MODEL, *s22, "hi", signed=True)
    add("TotalLossTwentyCOne", MODEL, "shipped_fit", "total_loss_20_laps_s", "C1", places=2)
    add("TotalLossTwentyCFive", MODEL, "shipped_fit", "total_loss_20_laps_s", "C5", places=2)

    # ---------------------------------------------------------------- A3 ablation
    rows = ["ablation", "rows"]
    for c in COMPOUNDS:
        w = WORD[c]
        add(f"NaiveC{w}", MODEL, *rows, c, "naive_run_clustered", "mean", signed=True)
        add(f"FuelCorrC{w}", MODEL, *rows, c, "practitioner_fuel_corrected_within_stint", "mean", signed=True)
        add(f"DeconC{w}", MODEL, *rows, c, "deconfounded", "mean", signed=True)
    add("AblGapCOne", MODEL, "ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C1")
    add("AblGapCThree", MODEL, "ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C3")
    add("FuelCoefficient", MODEL, "ablation", "fuel_s_per_kg_central")

    # ---------------------------------------------------------------- A4 interval widths
    wr = ["inference_comparison", "width_ratio_vs_B"]
    add("WidthNoCircuitCThree", MODEL, *wr, "A_lap_level_no_circuit", "C3")
    add("WidthNoCircuitCOne", MODEL, *wr, "A_lap_level_no_circuit", "C1")
    add("WidthShippedCThree", MODEL, *wr, "D_run_intercept_and_slope_SHIPPED", "C3")
    add("WidthShippedCOne", MODEL, *wr, "D_run_intercept_and_slope_SHIPPED", "C1")
    lag = ["inference_comparison", "residual_lag1_autocorrelation_within_runs"]
    add("LagOneCircuitModel", MODEL, *lag, "B_lap_level_circuit")
    add("LagOneShipped", MODEL, *lag, "D_run_intercept_and_slope_SHIPPED")

    # ---------------------------------------------------------------- A5 calibration
    cov = ["rate_calibration", "leave_one_event_out", "variants"]
    c80 = [*cov, "full_predictive", "coverage", "0.80"]
    add("CovEightyLOCO", MODEL, *c80, "coverage")
    add("CovEightyK", MODEL, *c80, "k")
    add("CovEightyN", MODEL, *c80, "n")
    add("CovEightyWilsonLo", MODEL, *c80, "wilson_95", 0)
    add("CovEightyWilsonHi", MODEL, *c80, "wilson_95", 1)
    add("CovNinetyFiveLOCO", MODEL, *cov, "full_predictive", "coverage", "0.95", "coverage")
    add("CovEightyNoCircuit", MODEL, *cov, "no_circuit_variance", "coverage", "0.80", "coverage")
    add("CovEightySBC", MODEL, "rate_calibration", "simulation_based_calibration", "coverage_of_true_rate", "0.80", "pooled")
    add("CovEightyLapLevel", MODEL, "lap_level_calibration", "coverage_80")

    # ---------------------------------------------------------------- A6 power
    # (1) min_detectable_effect: final. Analytic (known variances) and simulated (refit) minimum detectable
    # effect at 80% power; the realistic design is Pirelli's actual 2026 nominations at our actual run counts.
    an = ["min_detectable_effect", "analytic_known_variance"]
    sim = ["min_detectable_effect", "simulated_refit"]
    act = "at_actual_pair_runs"
    add("NRunsCOneCTwo", MODEL, *an, "realistic_C1_C2", act, "n_stints")
    for pair, w in (("C1_C2", "COneCTwo"), ("C2_C3", "CTwoCThree"), ("C3_C4", "CThreeCFour"), ("C4_C5", "CFourCFive")):
        add(f"MDEReal{w}", MODEL, *an, f"realistic_{pair}", act, "point", "mde_80_s_per_lap")
    r12 = [*an, "realistic_C1_C2", act]
    add("MDERealCOneCTwoCircLo", MODEL, *r12, "circuit_sd_low", "mde_80_s_per_lap")
    add("MDERealCOneCTwoCircHi", MODEL, *r12, "circuit_sd_high", "mde_80_s_per_lap")
    add("MDERealCOneCTwoStress", MODEL, *r12, "circuit_high_and_run_slope_x1.5", "mde_80_s_per_lap")
    add("MDERealCOneCTwoRound", MODEL, *r12, "point", "mde_80_s_per_lap", places=2, absval=True)  # "about 0.02"
    add("NStintsBalanced", MODEL, *an, "balanced", "at_N_832", "n_stints")
    add("MDEBalanced", MODEL, *an, "balanced", "at_N_832", "point", "mde_80_s_per_lap")
    add("MDEBalancedStress", MODEL, *an, "balanced", "at_N_832", "circuit_high_and_run_slope_x1.5", "mde_80_s_per_lap")
    add("MDESplit", MODEL, *an, "split", "at_N_832", "point", "mde_80_s_per_lap")
    add("MDESplitLo", MODEL, *an, "split", "at_N_832", "circuit_sd_low", "mde_80_s_per_lap")
    add("MDESplitHi", MODEL, *an, "split", "at_N_832", "circuit_sd_high", "mde_80_s_per_lap")
    sr = [*sim, "realistic_C1_C2"]
    add("MDESimRealCOneCTwo", MODEL, *sr, "mde_80_wald_z", "crossing")
    add("MDESimRealCOneCTwoLo", MODEL, *sr, "mde_80_wald_z", "mc_ci95", 0)
    add("MDESimRealCOneCTwoHi", MODEL, *sr, "mde_80_wald_z", "mc_ci95", 1)
    add("MDESimRealCOneCTwoT", MODEL, *sr, "mde_80_t_df12", "crossing")
    add("MDESimReps", MODEL, *sr, "reps")
    add("PowerEffectSize", ART_POWER, "effect_size", places=3)
    add("PowerStintsCappello", ART_POWER, "benchmark_n")

    # (2) power_extended: PENDING (a long run merges it into model.json later). Until the key exists every
    # macro below is the flagged placeholder (bold ??) (never a number), and build_manuscript_assets warns;
    # tests/test_paper_assets.py fails if a placeholder is committed once the key is present.
    px_ready = has(MODEL, "power_extended")
    px = ["power_extended"]
    eff = (("Six", "0.006"), ("Ten", "0.01"), ("Twenty", "0.02"), ("Thirty", "0.03"))

    def pxadd(name, *path, **kw):
        add(name, MODEL, *px, *path, placeholder=not px_ready, **kw)

    for w, e in eff:
        pxadd(f"PXRunsBal{w}", "analytic_crossing_80pct_stints", f"effect_{e}", "balanced", "point", places=0)
        pxadd(f"PXRunsReal{w}", "analytic_crossing_80pct_stints", f"effect_{e}", "realistic_C1_C2", "point", places=0)
        pxadd(f"PXSplitCeil{w}", "split_design_power_ceiling", f"effect_{e}", "point", places=3)
    pxadd("PXSplitCeilLoSDThirty", "split_design_power_ceiling", "effect_0.03", "circuit_sd_low", places=3)
    for w, e in eff[2:3]:  # simulation against the analytic crossing, at 0.02 s/lap
        for dn, tag in (("balanced", "Bal"), ("realistic_C1_C2", "Real")):
            sc = ["simulated", f"effect_{e}", dn]
            pxadd(f"PXSimRuns{tag}{w}", *sc, "crossing_80_wald_z", "crossing", places=0)
            pxadd(f"PXSimRuns{tag}{w}Lo", *sc, "crossing_80_wald_z", "mc_ci95", 0, places=0)
            pxadd(f"PXSimRuns{tag}{w}Hi", *sc, "crossing_80_wald_z", "mc_ci95", 1, places=0)
    pxadd("PXSimReps", "simulated", "effect_0.02", "balanced", "reps")
    for dn, tag in (("balanced", "Bal"), ("realistic_C1_C2", "Real")):
        pxadd(f"PXSize{tag}", "type_I_error_at_zero_effect", dn, "wald_z_reject_rate", places=3)
        pxadd(f"PXSizeT{tag}", "type_I_error_at_zero_effect", dn, "t_df12_reject_rate", places=3)
    pxadd("PXSizeReps", "type_I_error_at_zero_effect", "balanced", "reps")
    pxadd("PXSizeMCSE", "type_I_error_at_zero_effect", "balanced", "wald_z_mc_se", places=3)

    # ---------------------------------------------------------------- A7 race vs practice
    # PRIMARY design: used sets (FreshTyre false) in both contexts, tyre ages 6-16 (referee_additions).
    ud = ["referee_additions", "designs", "used_sets_only_FreshTyre_false"]
    lg = [*ud, "level_gap"]
    ms = [*lg, "MEDIUM+SOFT"]
    add("RaceToPracticeRatio", MGMT, *ms, "ratio")
    add("RaceToPracticeRatioLo", MGMT, *ms, "ratio_ci95", 0)
    add("RaceToPracticeRatioHi", MGMT, *ms, "ratio_ci95", 1)
    add("NMatchedCellsMS", MGMT, *ms, "n")
    add("LevelGapMS", MGMT, *ms, "gap", signed=True)
    add("LevelGapMSLo", MGMT, *ms, "gap_ci95", 0, signed=True)
    add("LevelGapMSHi", MGMT, *ms, "gap_ci95", 1, signed=True)
    add("MatchedRaceSlopeMS", MGMT, *ms, "race", places=4)
    add("MatchedPracticeSlopeMS", MGMT, *ms, "practice", places=4)
    add("NMatchedCellsAll", MGMT, *lg, "n_cells")
    add("NMatchedEventSeasons", MGMT, *lg, "n_event_seasons")
    add("LevelGapAll", MGMT, *lg, "ALL", "gap", signed=True)
    add("LevelGapAllLo", MGMT, *lg, "ALL", "gap_ci95", 0, signed=True)
    add("LevelGapAllHi", MGMT, *lg, "ALL", "gap_ci95", 1, signed=True)
    add("RaceToPracticeRatioAll", MGMT, *lg, "ALL", "ratio", places=2)
    for nm, k in (("Hard", "HARD"), ("Medium", "MEDIUM"), ("Soft", "SOFT")):
        add(f"Gap{nm}", MGMT, *lg, k, "gap", places=3, signed=True)
    # robustness designs
    sa = ["referee_additions", "designs", "start_age_proxy_matched", "level_gap"]
    add("StartAgeGapMS", MGMT, *sa, "MEDIUM+SOFT", "gap", signed=True)
    add("StartAgeGapMSLo", MGMT, *sa, "MEDIUM+SOFT", "gap_ci95", 0, signed=True)
    add("StartAgeGapMSHi", MGMT, *sa, "MEDIUM+SOFT", "gap_ci95", 1, signed=True)
    add("StartAgeRatioMS", MGMT, *sa, "MEDIUM+SOFT", "ratio", places=2)
    add("StartAgeRatioMSLo", MGMT, *sa, "MEDIUM+SOFT", "ratio_ci95", 0, places=2)
    add("StartAgeRatioMSHi", MGMT, *sa, "MEDIUM+SOFT", "ratio_ci95", 1, places=2)
    add("StartAgeCellsMS", MGMT, *sa, "MEDIUM+SOFT", "n")
    add("StartAgeCellsAll", MGMT, *sa, "n_cells")
    add("StartAgeEventSeasons", MGMT, *sa, "n_event_seasons")
    ix = ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap"]
    add("IntersectGapMS", MGMT, *ix, "MEDIUM+SOFT", "gap", signed=True)
    add("IntersectGapMSLo", MGMT, *ix, "MEDIUM+SOFT", "gap_ci95", 0, signed=True)
    add("IntersectGapMSHi", MGMT, *ix, "MEDIUM+SOFT", "gap_ci95", 1, signed=True)
    add("IntersectRatioMS", MGMT, *ix, "MEDIUM+SOFT", "ratio", places=2)
    add("IntersectRatioMSLo", MGMT, *ix, "MEDIUM+SOFT", "ratio_ci95", 0, places=2)
    add("IntersectRatioMSHi", MGMT, *ix, "MEDIUM+SOFT", "ratio_ci95", 1, places=2)
    add("IntersectCellsMS", MGMT, *ix, "MEDIUM+SOFT", "n")
    add("IntersectCellsAll", MGMT, *ix, "n_cells")
    add("IntersectEventSeasons", MGMT, *ix, "n_event_seasons")
    um = ["post_review_additions", "level_gap_unmatched_pooled", "MEDIUM+SOFT"]
    add("UnmatchedRatioMS", MGMT, *um, "ratio", places=2)
    add("UnmatchedGapMS", MGMT, *um, "gap", places=3, signed=True)
    add("NUnmatchedCellsMS", MGMT, *um, "n")
    # placebo (FP2 against FP1 and FP3 together), same used-set design; its orientation is arbitrary
    pl = [*ud, "level_placebo_fp2_minus_fp1fp3"]
    add("PlaceboGapMS", MGMT, *pl, "MEDIUM+SOFT", "gap", signed=True, places=3)
    add("PlaceboGapMSLo", MGMT, *pl, "MEDIUM+SOFT", "gap_ci95", 0, signed=True, places=3)
    add("PlaceboGapMSHi", MGMT, *pl, "MEDIUM+SOFT", "gap_ci95", 1, signed=True, places=3)
    add("NPlaceboCellsMS", MGMT, *pl, "MEDIUM+SOFT", "n")
    add("NPlaceboCells", MGMT, *pl, "n_cells")
    pa = [*ud, "placebo_adjusted_gap", "MEDIUM+SOFT"]
    add("PlaceboAdjGapMS", MGMT, *pa, "placebo_adjusted_gap", signed=True, places=3)
    add("PlaceboAdjGapMSLo", MGMT, *pa, "placebo_adjusted_gap_ci95_joint_bootstrap", 0, signed=True, places=3)
    add("PlaceboAdjGapMSHi", MGMT, *pa, "placebo_adjusted_gap_ci95_joint_bootstrap", 1, signed=True, places=3)
    add("PlaceboShareMS", MGMT, *pa, "placebo_share_of_gap", places=2)
    # why start-age matching is not the primary design (counts of runs)
    ss = ["referee_additions", "set_state_counts"]
    add("RaceRunsStartAge", MGMT, *ss, "race", "start_age_ge_5", "n_runs")
    add("RaceRunsStartAgeUsed", MGMT, *ss, "race", "start_age_ge_5", "genuinely_used_set_FreshTyre_false")
    add("RaceRunsStartAgeGap", MGMT, *ss, "race", "start_age_ge_5", "new_set_but_run_starts_after_stint_start_gap_split")
    add("PracRunsStartAge", MGMT, *ss, "practice_race_sims", "start_age_5_to_9", "n_runs")
    add("PracRunsStartAgeUsed", MGMT, *ss, "practice_race_sims", "start_age_5_to_9", "genuinely_used_set_FreshTyre_false")
    add("RaceUsedRuns", MGMT, *ss, "race", "n_used_set_runs")
    add("RaceAllRuns", MGMT, *ss, "race", "n_runs")
    add("PracUsedRuns", MGMT, *ss, "practice_race_sims", "n_used_set_runs")
    add("PracAllRuns", MGMT, *ss, "practice_race_sims", "n_runs")
    # reconciliation of the race-only compound trend across estimators
    rc = ["referee_additions", "reconcile_race_compound_trend", "explanation"]
    rcs = ["referee_additions", "reconcile_race_compound_trend", "settings", "all_seasons|window_6_16",
           "race_only_trend_by_estimator", "within_run"]
    add("ReconWithinRunAll", MGMT, *rcs, "beta_per_step_softer", places=3, signed=True)
    add("ReconWithinRunAllLo", MGMT, *rcs, "ci95", 0, places=3, signed=True)
    add("ReconWithinRunAllHi", MGMT, *rcs, "ci95", 1, places=3, signed=True)
    add("ReconWithinRunTwentySix", MGMT, *rc, "race_only_trend_2026_window_6_16", "within_run", places=3, signed=True)
    add("ReconCrossSectionalAll", MGMT, *rc, "race_only_trend_all_seasons_window_6_16", "xs", places=3, signed=True)
    add("ReconFieldDemeanedAll", MGMT, *rc, "race_only_trend_all_seasons_window_6_16", "field_origin", places=3, signed=True)

    # ---------------------------------------------------------------- A8 nursing not established
    tw = ["post_review_additions", "twfe_gradient"]
    plo = ["post_review_additions", "level_placebo_fp2_vs_fp1fp3_matched_pooled"]
    for nm, k in (("SoftHard", "SOFT-HARD"), ("SoftMedium", "SOFT-MEDIUM"), ("MediumHard", "MEDIUM-HARD")):
        add(f"DidTWFE{nm}", MGMT, *tw, k, "estimate_s_per_lap", signed=True)
        add(f"DidTWFE{nm}Lo", MGMT, *tw, k, "ci95", 0, signed=True)
        add(f"DidTWFE{nm}Hi", MGMT, *tw, k, "ci95", 1, signed=True)
        add(f"DidTWFE{nm}NEvents", MGMT, *tw, k, "n_events")
        new(f"DidTWFE{nm}PPerm", MGMT, *tw, k, "p_perm_two_sided", places=3)
    hm = ["referee_additions", "twfe_primary_holm", "holm_adjusted_p_two_sided"]
    for nm, k in (("SoftHard", "SOFT-HARD"), ("SoftMedium", "SOFT-MEDIUM"), ("MediumHard", "MEDIUM-HARD")):
        add(f"DidTWFE{nm}HolmP", MGMT, *hm, k, places=3)
        new(f"DidTWFE{nm}RaceContrast", MGMT, *tw, k, "race_contrast_mean", places=3, signed=True)
    udp = ["post_review_additions", "uniform_damping"]
    add("UniformDampingImplied", MGMT, *udp, "implied_additive_beta_per_step_if_race_slope_were_k_times_practice", "0.5", signed=True)
    add("UniformDampingObserved", MGMT, *udp, "observed_beta", signed=True)
    add("PlaceboSoftMedium", MGMT, *plo, "gap_SOFT_minus_MEDIUM", "estimate", signed=True)
    add("PlaceboSoftMediumLo", MGMT, *plo, "gap_SOFT_minus_MEDIUM", "ci95", 0, signed=True)
    add("PlaceboSoftMediumHi", MGMT, *plo, "gap_SOFT_minus_MEDIUM", "ci95", 1, signed=True)
    tr = ["primary", "trend"]
    add("NursingTrend", MGMT, *tr, "beta_per_softness_step", signed=True)
    add("NursingTrendLo", MGMT, *tr, "ci95_cluster_bootstrap", 0, signed=True)
    add("NursingTrendHi", MGMT, *tr, "ci95_cluster_bootstrap", 1, signed=True)
    add("NursingPOneSided", MGMT, *tr, "p_perm_one_sided_less")
    add("NursingPTwoSided", MGMT, *tr, "p_perm_two_sided")
    add("NursingMDE", MGMT, "post_review_additions", "power_additive_trend", "mde80_power_full_rule")
    add("MultiverseSpecs", MGMT, "multiverse_summary", "n_specs")
    add("MultiverseNegative", MGMT, "multiverse_summary", "n_beta_negative")
    add("MultiverseExcludesZero", MGMT, "multiverse_summary", "n_ci_excludes_zero_negative")

    # ---------------------------------------------------------------- A9 telemetry
    add("TelePrimaryEvents", TELE, "primary", "n_events")
    add("TelePrimaryLaps", TELE, "primary", "n_laps")
    add("TeleExtensionEvents", TELE, "extension", "n_events")
    for tag, blk in (("Primary", "primary"), ("Extension", "extension")):
        ta = [blk, "proxies", "coast_s", "trend_all"]
        add(f"Tele{tag}Slope", TELE, *ta, "slope", places=3, signed=True)
        add(f"Tele{tag}SlopeLo", TELE, *ta, "ci95_event_bootstrap", 0, places=3, signed=True)
        add(f"Tele{tag}SlopeHi", TELE, *ta, "ci95_event_bootstrap", 1, places=3, signed=True)
        add(f"Tele{tag}POneSided", TELE, *ta, "p_perm_one_sided", places=3)
        add(f"Tele{tag}PTwoSided", TELE, *ta, "p_perm_two_sided", places=3)
        add(f"Tele{tag}NEventsTrend", TELE, *ta, "n_events")

    # ---------------------------------------------------------------- A10 forecast
    hf = ["holdout_frozen"]
    sb = [*hf, "scoreboard"]
    add("HoldoutCells", TRANS, *hf, "n_cells")
    add("HoldoutWeekends", TRANS, *hf, "n_weekends")
    add("FrozenFactor", TRANS, *hf, "factor_frozen_from_2026")
    add("MaeScalar", TRANS, *sb, "scalar", "mae")
    add("MaeScalarLo", TRANS, *sb, "scalar", "mae_ci_lo")
    add("MaeScalarHi", TRANS, *sb, "scalar", "mae_ci_hi")
    add("MaeNaive", TRANS, *sb, "naive", "mae")
    add("MaeNaiveLo", TRANS, *sb, "naive", "mae_ci_lo")
    add("MaeNaiveHi", TRANS, *sb, "naive", "mae_ci_hi")
    add("MaeZero", TRANS, *sb, "zero", "mae")
    add("MaeConstant", TRANS, *sb, "const", "mae")
    add("MaeStack", TRANS, *sb, "stack", "mae")
    add("MaeShrink", TRANS, *sb, "shrink50", "mae")
    for nm, k in (("Constant", "const"), ("History", "history"), ("Stack", "stack")):
        add(f"DiffScalarVs{nm}", TRANS, *hf, "scalar_vs", k, "mean_diff", signed=True)
        add(f"DiffScalarVs{nm}Lo", TRANS, *hf, "scalar_vs", k, "ci_lo", signed=True)
        add(f"DiffScalarVs{nm}Hi", TRANS, *hf, "scalar_vs", k, "ci_hi", signed=True)
    add("MaeTwentySixScalar", TRANS, "s2026_loeo", "scoreboard", "scalar", "mae")
    add("MaeTwentySixNaive", TRANS, "s2026_loeo", "scoreboard", "naive", "mae")
    add("MaeNested", TRANS, "nested", "mae_nested")
    add("NestedOptimism", TRANS, "nested", "optimism_of_shipped")
    add("TwentySixCells", TRANS, "data", "cells_2026")
    # rest of the ladder (not in CLAIMS; tables and figure use the same JSON)
    for nm, k in (("Pooled", "pooled"), ("History", "history"), ("Compound", "compound")):
        new(f"Mae{nm}", TRANS, *sb, k, "mae", places=4)
    for nm, k in (("Zero", "zero"), ("Constant", "const"), ("Pooled", "pooled"), ("History", "history"),
                  ("Compound", "compound"), ("Shrink", "shrink50"), ("Stack", "stack")):
        new(f"Mae{nm}Lo", TRANS, *sb, k, "mae_ci_lo", places=4)
        new(f"Mae{nm}Hi", TRANS, *sb, k, "mae_ci_hi", places=4)

    # ---------------------------------------------------------------- A11 pit loss
    bc = ["by_category"]
    add("PitLossGreen", PIT, *bc, "green", "median_s")
    add("PitLossGreenLo", PIT, *bc, "green", "median_ci95", 0)
    add("PitLossGreenHi", PIT, *bc, "green", "median_ci95", 1)
    add("PitStopsGreen", PIT, *bc, "green", "n_stops")
    add("PitRacesGreen", PIT, *bc, "green", "n_races")
    add("PitRatioVSC", PIT, *bc, "vsc", "ratio_to_green")
    add("PitRatioVSCLo", PIT, *bc, "vsc", "ratio_ci95", 0)
    add("PitRatioVSCHi", PIT, *bc, "vsc", "ratio_ci95", 1)
    add("PitStopsVSC", PIT, *bc, "vsc", "n_stops")
    add("PitRatioSC", PIT, *bc, "safety_car", "ratio_to_green")
    add("PitRatioSCLo", PIT, *bc, "safety_car", "ratio_ci95", 0)
    add("PitRatioSCHi", PIT, *bc, "safety_car", "ratio_ci95", 1)
    add("PitStopsSC", PIT, *bc, "safety_car", "n_stops")
    add("PitRacesSC", PIT, *bc, "safety_car", "n_races")
    add("PitRatioYellow", PIT, *bc, "yellow", "ratio_to_green")
    add("PitRatioVSCTwentySix", PIT, "by_season", "2026", "vsc", "ratio")
    add("PitRatioSCTwentySix", PIT, "by_season", "2026", "safety_car", "ratio")
    add("PitRatioSCNoFilter", PIT, "console_default", "filter_sensitivity", "safety_car_ratio_no_filter")
    add("PitSCOutsideFilter", PIT, "console_default", "filter_sensitivity", "safety_car_stops_outside_sanity_filter")
    add("PitConsoleFraction", PIT, "console_default", "fraction")
    add("PitRatioSCNoFilterLo", PIT, "truncation_effect", "safety_car", "unfiltered", "ratio_ci95", 0)
    add("PitRatioSCNoFilterHi", PIT, "truncation_effect", "safety_car", "unfiltered", "ratio_ci95", 1)

    # ---------------------------------------------------------------- A12 strategy
    pb = ["paired_vs_baselines", "practice_scalar/plugin+support"]
    add("StratRacesOOT", STRAT, *pb, "lap_rule", "stop_match_mode", "n_races")
    add("StratDiffLapRule", STRAT, *pb, "lap_rule", "stop_match_mode", "mean_a_minus_b", signed=True)
    add("StratDiffLapRuleLo", STRAT, *pb, "lap_rule", "stop_match_mode", "ci95", 0, signed=True)
    add("StratDiffLapRuleHi", STRAT, *pb, "lap_rule", "stop_match_mode", "ci95", 1, signed=True)
    add("StratDiffAlwaysOne", STRAT, *pb, "always_1", "stop_match_mode", "mean_a_minus_b", signed=True)
    add("StratDiffAlwaysOneLo", STRAT, *pb, "always_1", "stop_match_mode", "ci95", 0, signed=True)
    add("StratDiffAlwaysOneHi", STRAT, *pb, "always_1", "stop_match_mode", "ci95", 1, signed=True)
    sc = ["shipped_claim_rechecked"]
    add("StratProductMatches", STRAT, *sc, "product_in_sample_with_bounds", "matches")
    add("StratProductOf", STRAT, *sc, "product_in_sample_with_bounds", "of")
    add("StratAlwaysOneMatches", STRAT, *sc, "always_one_stop", "matches")
    add("StratAlwaysOneOf", STRAT, *sc, "always_one_stop", "of")
    rl = [*sc, "race_length_rule_best_threshold_fitted_on_these_events"]
    add("StratRuleThreshold", STRAT, *rl, "threshold_laps")
    add("StratRuleMatches", STRAT, *rl, "matches")
    add("StratRuleOf", STRAT, *rl, "of")
    lo = ["leave_one_race_out_sensitivity", "paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode"]
    add("StratLOROLapRule", STRAT, *lo, "mean_a_minus_b", signed=True)
    add("StratLOROLapRuleLo", STRAT, *lo, "ci95", 0, signed=True)
    add("StratLOROLapRuleHi", STRAT, *lo, "ci95", 1, signed=True)
    add("StratLORORaces", STRAT, *lo, "n_races")

    # ---------------------------------------------------------------- A13 benchmark
    t3 = ["table3"]
    add("TableThreeHard", BENCH, *t3, "their_ramp", "v_hard", "mean")
    add("TableThreeMedium", BENCH, *t3, "their_ramp", "v_medium", "mean")
    add("TableThreeHardPub", BENCH, *t3, "published", "hard", "mean", places=3)
    add("TableThreeMediumPub", BENCH, *t3, "published", "medium", "mean", places=3)
    add("PMediumGtHard", BENCH, *t3, "their_ramp", "p_medium_gt_hard")
    add("FuelDefectKg", BENCH, "fuel_defect", "austria", "max_abs_kg")
    add("PMediumGtHardCorrected", BENCH, *t3, "corrected_fuel", "p_medium_gt_hard")
    add("FuelDefectDeltaCRPS", BENCH, "fuel_defect", "skewt_cv_effect", "delta_crps", signed=True)
    hb = ["head_to_head", "season_vs_published", "B_their_folds_vs_published"]
    add("BenchWinsVsRepoSSM", BENCH, *hb, "ours_wins")
    add("BenchCRPSOurs", BENCH, *hb, "ours_mean")
    add("BenchCRPSSSM", BENCH, *hb, "theirs_mean")
    add("BenchDiffSSM", BENCH, *hb, "mean_diff_ours_minus_theirs", signed=True)
    add("BenchDiffSSMLo", BENCH, *hb, "mean_diff_ci95", 0, signed=True)
    add("BenchDiffSSMHi", BENCH, *hb, "mean_diff_ci95", 1, signed=True)
    add("BenchSignPSSM", BENCH, *hb, "sign_test_p_two_sided")
    add("BenchNRaces", BENCH, *hb, "n_races")
    add("AustriaCRPSOurs", BENCH, "head_to_head", "austria", "ours_their_folds")
    add("AustriaCRPSTheirs", BENCH, "head_to_head", "austria", "theirs_repo_csv")
    av = ["head_to_head", "austria_paired_vs_reproduced_theirs"]
    add("AustriaDiff", BENCH, *av, "mean_paired_diff", signed=True)
    add("AustriaDiffLo", BENCH, *av, "ci95", 0, signed=True)
    add("AustriaDiffHi", BENCH, *av, "ci95", 1, signed=True)
    sk = ["skewt_cv_season_matched_vs_mismatched"]
    add("SkewTDelta", BENCH, *sk, "mean_diff_matched_minus_normal", signed=True)
    add("SkewTBetter", BENCH, *sk, "races_matched_better")
    add("SkewTRaces", BENCH, *sk, "n_races")
    add("SkewTAustriaDelta", BENCH, "skewt_cv_austria", "iter10000_theirs", "matched_minus_mismatched_crps", signed=True)
    jt = ["journal_table4"]
    add("JournalRacesScored", BENCH, *jt, "coverage", "ours_scored")
    add("JournalRacesTotal", BENCH, *jt, "coverage", "journal_races")
    add("JournalWinsSSM", BENCH, *jt, "common_races", "crps_vs_ssm", "ours_wins")
    add("JournalCRPSSSM", BENCH, *jt, "common_races", "crps_vs_ssm", "theirs_mean")
    ar = [*jt, "common_races", "crps_vs_ar1"]
    add("JournalWinsAR", BENCH, *ar, "ours_wins")
    add("JournalCRPSAR", BENCH, *ar, "theirs_mean")
    add("JournalDiffAR", BENCH, *ar, "mean_diff_ours_minus_theirs", signed=True)
    add("JournalDiffARLo", BENCH, *ar, "mean_diff_ci95", 0, signed=True)
    add("JournalDiffARHi", BENCH, *ar, "mean_diff_ci95", 1, signed=True)
    add("JournalSignPAR", BENCH, *ar, "sign_test_p_two_sided")
    add("JournalRMSPESSM", BENCH, *jt, "common_races", "rmspe_vs_ssm", "theirs_mean")
    add("JournalRMSPEOurs", BENCH, *jt, "common_races", "rmspe_vs_ssm", "ours_mean")
    add("JournalRMSPEAR", BENCH, *jt, "common_races", "rmspe_vs_ar1", "theirs_mean")
    add("JournalSSMBeatsARCommon", BENCH, *jt, "common_races", "journal_ssm_beats_ar1_on_these_races", "crps")
    add("JournalSSMBeatsARAll", BENCH, *jt, "table4_internal_checks", "ssm_beats_ar1", "crps")
    add("JournalPrintedMeanAR", BENCH, *jt, "table4_internal_checks", "crps_ar1", "printed")
    add("JournalMeanARAll", BENCH, *jt, "table4_internal_checks", "crps_ar1", "mean_all_19")
    add("JournalAustriaAR", PUB4, "races", "Austrian Grand Prix", "crps_ar1")
    add("JournalAustriaRMSPEAR", PUB4, "other_published_tables_austria", "table2_rmspe_total", "ar1")
    # rest of the head-to-head, for the table
    new("BenchOursRMSPE", BENCH, *jt, "common_races", "rmspe_vs_ssm", "ours_mean", places=3)
    new("JournalWinsRMSPESSM", BENCH, *jt, "common_races", "rmspe_vs_ssm", "ours_wins")
    new("JournalWinsRMSPEAR", BENCH, *jt, "common_races", "rmspe_vs_ar1", "ours_wins")

    # ---------------------------------------------------------------- A14 design sweep
    kr = ["2022-2025", "keep-runs"]
    add("KeepRunsDiff", DESIGN, *kr, "diff_vs_shipped", signed=True)
    add("KeepRunsDiffLo", DESIGN, *kr, "diff_ci", 0, signed=True)
    add("KeepRunsDiffHi", DESIGN, *kr, "diff_ci", 1, signed=True)
    add("KeepRunsCommonCells", DESIGN, *kr, "n_common_with_shipped")
    wu = ["2026", "warmup1"]
    add("WarmupDiff", DESIGN, *wu, "diff_vs_shipped", signed=True)
    add("WarmupDiffLo", DESIGN, *wu, "diff_ci", 0, signed=True)
    add("WarmupDiffHi", DESIGN, *wu, "diff_ci", 1, signed=True)
    add("WarmupSignFlipP", DESIGN, *wu, "signflip_p")
    add("PracticeLapsShipped", DESIGN, "parity", "shipped_laps")
    add("PracticeLapsLegacy", DESIGN, "parity", "legacy_laps")

    # ---------------------------------------------------------------- A15 dataset size
    add("NLapsCleanTwentySix", MODEL, "dataset", "n_laps_total")
    add("NEventsInFileTwentySix", MODEL, "dataset", "n_events_in_file")
    for y, w in SEASON.items():
        add(f"NLapsSeason{w}", MODEL, "season_robustness", y, "n_laps")
        add(f"NEventsSeason{w}", MODEL, "season_robustness", y, "n_events")
        add(f"NRunsSeason{w}", MODEL, "season_robustness", y, "n_runs")

    # ---------------------------------------------------------------- extras used by the manuscript text
    vsd = ["shipped_fit", "variance_sd"]
    new("SDDriverEvent", MODEL, *vsd, "driver_event", places=2)
    new("SDRunIntercept", MODEL, *vsd, "run", places=2)
    new("SDRunSlope", MODEL, *vsd, "run_slope", places=3)
    new("SDResidual", MODEL, *vsd, "resid", places=2)
    new("CompoundRateSD", MODEL, "circuit_effect", "effect_in_seconds_over_20_lap_stint",
        "sd_of_the_five_compound_rates_s_per_lap", places=4)
    new("EngineRateDiff", MODEL, "engine_agreement", "max_abs_rate_difference", places=6)
    for y, w in SEASON.items():
        new(f"CircuitSDSeason{w}", MODEL, "season_robustness", y, "circuit_slope_sd_s_per_lap", places=4)
    new("ReproCorr", BENCH, "head_to_head", "their_model_reproduction", "corr", places=4)
    new("ReproRaces", BENCH, "head_to_head", "their_model_reproduction", "n_races")
    new("TeleJoinFraction", TELE, "access", "primary_join", "fraction_reduced", places=2)

    # magnitudes of the race-minus-practice gap, for prose such as "0.056 s/lap lower than"
    add("LevelGapMSAbs", MGMT, *ms, "gap", places=3, absval=True)
    add("LevelGapMSAbsMin", MGMT, *ms, "gap_ci95", 1, places=3, absval=True)
    add("LevelGapMSAbsMax", MGMT, *ms, "gap_ci95", 0, places=3, absval=True)

    # the abstract rounds to three places; these are the same JSON values as the
    # four-place macros above (all positive, so the magnitude is the value)
    sdp = [*cv, "sd_s_per_lap", "circuit_persistent"]
    add("PersistSDAbs", MODEL, *sdp, "estimate", places=3, absval=True)
    add("PersistSDLoAbs", MODEL, *sdp, "profile_95", 0, places=3, absval=True)
    add("PersistSDHiAbs", MODEL, *sdp, "profile_95", 1, places=3, absval=True)
    add("EventSDAbs", MODEL, *cv, "sd_s_per_lap", "event_specific", "estimate", places=3, absval=True)
    # run-level slope variation in the same pooled model: the largest component
    add("RunSlopeSD", MODEL, *cv, "sd_s_per_lap", "run_slope", places=4)
    add("RunSlopeSDAbs", MODEL, *cv, "sd_s_per_lap", "run_slope", places=3, absval=True)
    add("PooledCompoundSDAbs", MODEL, *cv, "compound_rate_sd_s_per_lap", places=3, absval=True)
    add("MaeScalarAbs", TRANS, *sb, "scalar", "mae", places=3, absval=True)
    add("MaeConstantAbs", TRANS, *sb, "const", "mae", places=3, absval=True)

    # neighbouring-compound contrasts and the gap each fit would detect (from its own covariance)
    pair = {"C2-C1": "TwoOne", "C3-C2": "ThreeTwo", "C4-C3": "FourThree", "C5-C4": "FiveFour"}
    for fit_key, pre in (("pooled_2022_26_nested", "Pool"), ("shipped_2026", "Six")):
        for k, w in pair.items():
            ct = ["detectable_from_fit", fit_key, "contrasts", k]
            add(f"{pre}Con{w}", MODEL, *ct, "estimate", places=3, signed=True)
            add(f"{pre}Con{w}Lo", MODEL, *ct, "lo", places=3, signed=True)
            add(f"{pre}Con{w}Hi", MODEL, *ct, "hi", places=3, signed=True)
            add(f"{pre}MDE{w}", MODEL, *ct, "mde80", places=3)
            add(f"{pre}MDET{w}", MODEL, *ct, "mde80_t_reference", places=3)

    # correlated random effects (Mundlak): CRE minus RE slope shifts and the joint test
    mk = ["mundlak_check"]
    for fit_key, pre in (("pooled_2022_26_nested", "Pool"), ("shipped_2026", "Six")):
        add(f"Mundlak{pre}P", MODEL, *mk, fit_key, "wald_run_means_zero", "p_bootstrap_t", places=2)
        for c, w in WORD.items():
            add(f"Mundlak{pre}Shift{w}", MODEL, *mk, fit_key, "slopes_s_per_lap", c, "cre_minus_re", "estimate",
                places=4, signed=True)

    # exact versus approximate within-transformation, same laps and engine
    ea = ["exact_vs_approximate"]
    for fit_key, pre in (("pooled_2022_26_nested", "Pool"), ("shipped_2026", "Six")):
        for c, w in WORD.items():
            add(f"Approx{pre}Rate{w}", MODEL, *ea, fit_key, "approximate", "rates_s_per_lap", c, "mean", places=3)
        add(f"Approx{pre}CompoundSD", MODEL, *ea, fit_key, "approximate", "compound_rate_sd_raw", places=4)
    add("ApproxPoolCircuitSD", MODEL, *ea, "pooled_2022_26_nested", "approximate", "variance_sd", "circuit_slope",
        places=4)
    add("PooledCompoundSDCorr", MODEL, *ea, "pooled_2022_26_nested", "exact", "compound_rate_sd_noise_corrected",
        places=4)

    # recovery study: bias and coverage by estimator and scenario
    scen = {"exogenous": "Exo", "pace": "Pace", "wear": "Wear"}
    est = {"eq1_exact_within": "Exact", "eq1_exact_within_mundlak": "ExactMundlak", "eq1_shipped": "Approx",
           "cross_section_cell": "Cross", "field_demeaned_ols": "Field", "fuel_within_run": "Fuel",
           "naive_pooled_ols": "Naive"}
    for s, sw in scen.items():
        for e, ew in est.items():
            mo = ["results", s, e, "mean_over_compounds"]
            add(f"Rec{ew}{sw}Bias", RECOV, *mo, "bias", places=3, signed=True)
            add(f"Rec{ew}{sw}Cov", RECOV, *mo, "coverage", places=2)
        add(f"RecExact{sw}CircBias", RECOV, "results", s, "eq1_exact_within", "circuit_sd", "bias", places=4,
            signed=True)
        add(f"RecExact{sw}WorstCov", RECOV, "results", s, "eq1_exact_within", "worst_compound", "coverage", places=2)
    add("RecExactWearBiasAbs", RECOV, "results", "wear", "eq1_exact_within", "mean_over_compounds", "bias",
        places=3, absval=True)
    # pooled model without 2026 (new technical rules and narrower tyres)
    rg = ["regulation_sensitivity", "without_2026"]
    add("NoTwentySixCircuitSD", MODEL, *rg, "circuit_sd", places=3)
    add("NoTwentySixEventSD", MODEL, *rg, "event_sd", places=3)
    add("NoTwentySixShare", MODEL, *rg, "share_circuit", places=2)
    add("NoTwentySixCompoundSD", MODEL, *rg, "compound_rate_sd", places=3)
    add("NoTwentySixRateCFive", MODEL, *rg, "compound_rates_s_per_lap", "C5", "mean", places=3)
    add("NoTwentySixNEvents", MODEL, *rg, "n_events")
    add("RecFuelExoBiasAbs", RECOV, "results", "exogenous", "fuel_within_run", "mean_over_compounds", "bias",
        places=3, absval=True)
    add("RecTrueCircuitSD", RECOV, "truth_circuit_slope_sd_s_per_lap", places=3)
    add("RecReps", RECOV, "results", "exogenous", "eq1_exact_within", "circuit_sd", "n")

    # ---------------------------------------------------------------- coverage guard
    missing = []
    skipped = 0
    for c in claims:
        if "value" not in c:
            skipped += 1
            continue
        if (c["file"], tuple(c["path"])) not in paths:
            missing.append(f"{c['file']} :: {c['path']}")
    if missing:
        raise RuntimeError(
            "CLAIMS.md section A numbers with no macro (add them to make_numbers.py):\n  " + "\n  ".join(missing)
        )
    return macros


HEADER = (
    "% GENERATED by scripts/paper/make_numbers.py from paper/results/*.json - do not edit.\n"
    "% One macro per quoted number, rounded as in paper/CLAIMS.md. Usable in text and math mode.\n"
)


def placeholders(macros: list[Macro]) -> list[str]:
    return [m.name for m in macros if m.placeholder]


def render(macros: list[Macro]) -> str:
    lines = [HEADER]
    ph = placeholders(macros)
    if ph:
        lines.append(
            f"% WARNING: {len(ph)} PLACEHOLDER macros below print as bold ?? because their JSON key\n"
            "% (power_extended in model.json) does not exist yet. Rerun make_numbers.py once it does;\n"
            "% tests/test_paper_assets.py fails on committed placeholders when the key is present.\n"
        )
    for m in macros:
        lines.append(m.tex())
    return "\n".join(lines) + "\n"


def main() -> int:
    macros = build()
    out = GENERATED / "numbers.tex"
    write_text(out, render(macros))
    n_claim = sum(m.claimed for m in macros)
    print(f"wrote {shown(out)}: {len(macros)} macros ({n_claim} tied to CLAIMS.md section A checks)")
    ph = placeholders(macros)
    if ph:
        print(f"WARNING: {len(ph)} placeholder macros (power_extended is not in model.json yet), e.g. {ph[0]}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
