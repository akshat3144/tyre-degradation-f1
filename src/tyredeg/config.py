"""Project-wide configuration and verified constants.

Every number in this file was checked against a primary source. Where a value is
uncertain, that is stated in the comment. Do not change these without a source.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
FASTF1_CACHE = DATA / "fastf1_cache"
PROCESSED = DATA / "processed"
ARTIFACTS = DATA / "artifacts"

STAN_DIR = Path(__file__).resolve().parent / "models" / "stan"
BENCHMARK = ROOT / "benchmark"

for _p in (FASTF1_CACHE, PROCESSED, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2026 season facts
# Source: fastf1.get_event_schedule(2026), verified 2026-09-03.
# ---------------------------------------------------------------------------

SEASON = 2026

#: Total rounds on the 2026 calendar.
#: NOTE: 23, not 22. Saudi Arabia was cancelled. Bahrain was NOT -- it was
#: relocated to Sepang, Malaysia (2-4 Oct) and still carries the Bahrain name.
N_ROUNDS_2026 = 23

#: Completed conventional weekends as of 2026-09-03: the only 2026 events with
#: an FP2 session, and therefore the only ones with PRACTICE race-simulation
#: long runs. That is what practice->race transfer and the management analysis
#: need, and it is the only thing this list should ever gate.
#:
#: It does NOT mean sprint weekends have no usable data. A sprint weekend is
#: FP1 + Sprint Qualifying on Friday, Sprint + Qualifying on Saturday, and a
#: FULL GRAND PRIX on Sunday -- so its race long runs exist and are as good as
#: any other race's. Measured on the four cached 2025 sprint weekends: 198 race
#: long runs, median 20 laps, against 17 for conventional races. Their FP1 gives
#: 34 runs at a median of 6 laps, which is sprint prep and not a race sim.
#:
#: An earlier version of this comment said sprint weekends "have none", full
#: stop, which silently excluded five 2026 races from the degradation fit.
CONVENTIONAL_2026 = (
    "Australian Grand Prix",   # R1,  08 Mar
    "Japanese Grand Prix",     # R3,  29 Mar
    "Monaco Grand Prix",       # R6,  07 Jun
    "Barcelona Grand Prix",    # R7,  14 Jun
    "Austrian Grand Prix",     # R8,  28 Jun
    "Belgian Grand Prix",      # R10, 19 Jul
    "Hungarian Grand Prix",    # R11, 26 Jul
)

#: Completed sprint weekends as of 2026-09-03. No FP2, so no PRACTICE long runs
#: -- but a full Grand Prix on Sunday like everyone else, so their RACE long runs
#: are in the degradation fit and contribute 5 of its 13 events. This list gates
#: practice-derived work only. See the note on CONVENTIONAL_2026 above; reading
#: "no long runs" as "no data" is exactly the bug that cost us five races.
SPRINT_2026 = (
    "Chinese Grand Prix",      # R2
    "Miami Grand Prix",        # R4
    "Canadian Grand Prix",     # R5
    "British Grand Prix",      # R9
    "Dutch Grand Prix",        # R12
)


# ---------------------------------------------------------------------------
# 2026 technical regulations
# ---------------------------------------------------------------------------

#: Dry compounds for 2026. Five, not six -- Pirelli dropped the C6 because the
#: C5-to-C6 gap was too small to be worth validating.
COMPOUNDS = ("HARD", "MEDIUM", "SOFT")          # as labelled in the timing feed
COMPOUND_CODES = {"HARD": 1, "MEDIUM": 2, "SOFT": 3}   # matches the benchmark's coding
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

#: Pirelli's per-event compound nomination, 2026.
#:
#: THIS MATTERS MORE THAN IT LOOKS. "HARD", "MEDIUM" and "SOFT" in the timing
#: feed are RELATIVE to each weekend's nomination, not physical compounds.
#: Pirelli picks three of C1-C5 per race, so a "HARD" at Monaco (C3) is softer
#: rubber than a "SOFT" at Suzuka (C3 too, but as the softest of C1/C2/C3).
#:
#: Pooling by label across events therefore averages together different tyres,
#: and it is what every public FastF1 tyre analysis we have seen does. It is
#: also what produced a physically backwards degradation ordering in our own
#: first pooled fit (Hard appearing to wear faster than Soft).
#:
#: Mapping to the C number lets us pool physically identical rubber instead.
#: C3 appears at all seven completed conventional weekends and C4 at six, so
#: there is real cross-event overlap to exploit.
#:
#: Sources: Pirelli press releases and per-race F1.com tyre previews, 2026.
COMPOUND_ALLOCATION_2026 = {
    "Australian Grand Prix": {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    "Japanese Grand Prix":   {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},
    "Monaco Grand Prix":     {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    "Barcelona Grand Prix":  {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
    "Austrian Grand Prix":   {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    "Belgian Grand Prix":    {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
    "Hungarian Grand Prix":  {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    # Not yet raced when the allocation was recorded:
    "Italian Grand Prix":    {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    "Spanish Grand Prix":    {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
}

#: Hardest to softest. Ordered so a model can treat the index as a scale.
C_COMPOUNDS = ("C1", "C2", "C3", "C4", "C5")

#: Race fuel load, kg. Down from 110 kg under the previous regulations.
#: CAUTION: sources disagree on whether this is a hard regulatory cap at race
#: start or the practical race load implied by the 3000 MJ/h energy-flow limit.
#: The magnitude is not in doubt. Do not cite a regulation article number.
FUEL_RACE_KG_2026 = 70.0
FUEL_RACE_KG_LEGACY = 110.0     # what the benchmark paper assumed (correct for 2025)

#: Lap-time sensitivity to car mass, seconds per lap per kg.
#: From the widely used 0.3-0.35 s/lap per 10 kg rule of thumb. Circuit-agnostic,
#: so treat as a prior to check a fitted value against, never as a fixed input.
#:
#: The benchmark's model fits only ~0.016, about half (0.0155 on their ramp,
#: 0.0164 with the fuel index corrected; paper/results/benchmark.json). This
#: comment used to call that "our headline evidence of confounding", and that
#: claim is withdrawn. It also used to say our own hierarchical state-space
#: model reads ~0.043, "about 30% HIGH"; that figure could not be reproduced on
#: any race we refit. Whole-race gamma at the correct 110 kg start is 0.020 at
#: Austria 2025 and 0.036 at Bahrain 2024 (a diffuse prior gives the same), so
#: it is race-dependent, and neither "recovers the physical value" nor "reads
#: high" holds as a general statement.
#:
#: What the design actually buys is that it never has to estimate fuel: the
#: within-transformation subtracts the (event, lap) mean, removing fuel and
#: track evolution together. So this constant is a sanity check on any
#: level-based fit -- theirs or ours -- and not a scoreboard.
MASS_SENSITIVITY_S_PER_KG = 0.033
MASS_SENSITIVITY_RANGE = (0.030, 0.035)


# ---------------------------------------------------------------------------
# Lap filtering
# ---------------------------------------------------------------------------

#: TrackStatus codes meaning the track was not fully green.
#: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending.
#: The benchmark filters "4|5|6|7" as a substring match -- we replicate that
#: exactly in the benchmark harness, but use explicit codes in our own pipeline.
NON_GREEN_CODES = ("4", "5", "6", "7")
GREEN = "1"

#: Minimum consecutive clean green laps in a stint to count as a "long run".
MIN_LONG_RUN_LAPS = 5


# ---------------------------------------------------------------------------
# Benchmark targets (Cappello & Hoegh 2025, arXiv:2512.00640)
# Hamilton, 2025 Austrian GP. See benchmark/README.md.
# ---------------------------------------------------------------------------

#: Their Table 3: per-compound degradation, s/lap, with 95% credible intervals.
#: The intervals overlap almost completely -- the model cannot separate Hard
#: from Medium. Reproduced by us on 2026-09-03 (Hard 0.0550, Medium 0.0555).
BENCHMARK_TABLE3 = {
    "HARD":   {"mean": 0.054, "lo": 0.004, "hi": 0.133},
    "MEDIUM": {"mean": 0.060, "lo": 0.009, "hi": 0.120},
}

#: Their Table 1 (RMSPE) and Table 2 (CRPS). Lower is better.
#: WARNING: their "RMSPE" is a plain RMSE in seconds, not a percentage error,
#: and its "Total" row is a SUM over 3 stints while the CRPS total is a MEAN.
#: Do not compare RMSPE totals across races with different stint counts.
BENCHMARK_RMSPE = {"arima": 1.520, "base": 1.169, "ext1": 1.187, "ext2": 1.218, "skew_t": 1.082}
BENCHMARK_CRPS = {"arima": 0.324, "base": 0.230, "ext1": 0.236, "ext2": 0.245, "skew_t": 0.202}

#: Our target. Note that beating 0.202 and separating the compounds are two
#: different jobs: their compound-specific model (ext1) scored WORSE than their
#: base model, and their best model has no compound structure at all. We need
#: pooling AND a heavy-tailed observation model to do both.
TARGET_CRPS = 0.202

#: Season-wide benchmark from their repo (19 races, Hamilton, skew-t vs ARIMA).
#: Not published in the paper. Gives us a win-rate to report instead of one number.
BENCHMARK_SEASON_2025 = {
    "n_races": 19,
    "n_stints": 51,
    "skewt_crps_mean": 0.238,
    "arima_crps_mean": 0.2997,
    "skewt_rmspe_mean": 0.4088,
    "arima_rmspe_mean": 0.4786,
    "skewt_crps_wins": 16,
}

#: Their skew-t CRPS per race, and the stint count behind each, from
#: ``Cross_Validation_Results/All_CV_results1.csv`` in their repo. Values are
#: (crps, n_stints).
#:
#: WHY THIS IS HERE AND WHY IT MATTERS
#:
#: We spent a while asserting that their repository published only a season
#: mean, so a per-race comparison "cannot be built without inventing their side
#: of it". That was simply wrong -- the per-race numbers are in their repo, and
#: writing the claim down instead of opening the file is the mistake.
#:
#: It matters because the honest comparison is much harder on us. Comparing our
#: median to their mean over different race sets flattered us; a race-by-race
#: table says we win 2 of the 15 races we both scored.
#:
#: Austria here is 0.2022 against the paper's Table 2 figure of 0.202, and our
#: reconstructed stint counts match theirs at 12 of 15 races, which is the
#: evidence that this is like-for-like rather than two different schemes.
#:
#: Transcribed rather than read from ``benchmark/upstream/``, which is fetched
#: locally and never committed because their repo carries no license. Nineteen
#: published numbers cited with attribution is a different thing from
#: redistributing their code.
BENCHMARK_SEASON_2025_PER_RACE = {
    "Abu Dhabi Grand Prix": (0.1768, 3),
    "Austrian Grand Prix": (0.2022, 3),
    "Azerbaijan Grand Prix": (0.2872, 2),
    "Bahrain Grand Prix": (0.1447, 3),
    "Canadian Grand Prix": (0.2158, 3),
    "Chinese Grand Prix": (0.1438, 3),
    "Emilia Romagna Grand Prix": (0.2064, 3),
    "Hungarian Grand Prix": (0.3111, 2),
    "Italian Grand Prix": (0.1284, 2),
    "Japanese Grand Prix": (0.1654, 2),
    "Las Vegas Grand Prix": (0.1991, 2),
    "Mexico City Grand Prix": (0.1985, 3),
    "Monaco Grand Prix": (0.4021, 3),
    "Qatar Grand Prix": (0.2470, 3),
    "Saudi Arabian Grand Prix": (0.1620, 2),
    "Singapore Grand Prix": (0.2398, 3),
    "Spanish Grand Prix": (0.2645, 4),
    "Sao Paulo Grand Prix": (0.2524, 3),
    "United States Grand Prix": (0.5750, 2),
}


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

#: CmdStan install shared by cmdstanpy (Python) and cmdstanr (R).
CMDSTAN_PATH = Path.home() / ".cmdstan" / "cmdstan-2.39.0"

#: Rtools bin directory. MUST be prepended to PATH before compiling Stan models:
#: C:/MinGW/bin is on this machine's system PATH carrying GCC 6.3.0 from 2016,
#: which shadows modern compilers and breaks every Stan build. See benchmark/README.md.
RTOOLS_BIN = Path("C:/rtools45/x86_64-w64-mingw32.static.posix/bin")
RTOOLS_USR_BIN = Path("C:/rtools45/usr/bin")
