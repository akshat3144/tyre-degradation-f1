# What public timing data can tell us about tyre degradation in Formula 1

Code and results for a study of what public lap-time data can and cannot
say about Formula 1 tyre degradation.

## Summary

A lap time mixes tyre wear with fuel burn, track evolution, traffic and driver
effects, and within one stint tyre age and fuel burn move together, so no estimator
can separate them from a single car. We pool the field and difference out
everything common to a lap (a within-(event, lap) design), then fit a mixed model
with run-level and circuit-level slopes so that intervals count stints, not laps.
The question is how far such estimates can be trusted: for ranking compounds, for
forecasting a race from that weekend's practice, and for pit strategy. The main
findings, each backed by a check in [`paper/CLAIMS.md`](paper/CLAIMS.md):

- **Degradation differs persistently by circuit, about as much as race to race, and far
  more than by compound.** Pooled over 2022-2026 (82 races, 23 circuits), the persistent
  circuit slope SD is 0.025 s/lap (95% profile interval [0.017, 0.037]; bootstrap p = 0.002),
  about 0.5 s over a 20-lap stint; race-to-race SD at a circuit is 0.024 [0.021, 0.029];
  the SD of the five compound rates is 0.005. The 2026-only circuit SD (0.044) mixes the
  two because each circuit was raced once. Rate intervals are approximately calibrated: leave-one-circuit-out coverage at a
  nominal 80% is 0.78 (28 of 36 circuit-compound groups).
- **Small compound differences are not detectable on this data.** In 2026 the
  nominally hardest tyre (C1) has the highest fitted rate and no adjacent pair
  separates at 95%. Under Pirelli's actual nominations the smallest gap detectable at
  80% power is 0.0175 s/lap for C1 against C2 (246 runs); a 0.006 s/lap gap would need
  about 2,100 runs of the pair. If compounds ran at disjoint circuits, no sample
  size would suffice, because circuit variance enters the contrast.
- **Naive slopes get the sign wrong.** Without controls, 3 of 5 compounds show
  negative degradation; the within-event estimator does not.
- **Race degradation is about half of practice degradation on matched used sets, and
  the data do not establish compound-specific tyre management.** The lead estimand
  is the race-minus-practice gap in the same within-run estimator, on used tyre sets
  (feed flag FreshTyre false) in both contexts, tyre ages 6-16. For MEDIUM+SOFT it is
  -0.056 s/lap (95% event-season bootstrap [-0.096, -0.019], 82 groups; race slope
  0.56 [0.36, 0.81] of practice); over all 105 groups in 67 event-seasons it is
  -0.059 [-0.093, -0.024]. Robustness: start-age matching gives -0.099
  [-0.154, -0.050], the intersection of both rules -0.058 [-0.106, -0.012]. A
  practice-versus-practice placebo (FP2 against FP1+FP3; its orientation is
  arbitrary) gives -0.006 [-0.053, 0.037], so the placebo-adjusted gap is -0.050
  [-0.106, +0.007]. The soft-minus-hard difference in differences (-0.071,
  permutation p = 0.028) is not significant after Holm adjustment over the three
  pairs (p = 0.085). See `referee_additions` in `paper/results/management.json`.
  A telemetry check (coast time) finds nothing.
- **Practice forecasts race degradation no better than a constant.** Out of season
  (2022-2025) the scaled practice forecast has MAE 0.036 s/lap against 0.112
  for "Sunday equals Friday", but is not distinguishable from a constant prior at 95%.
- **The strategy layer barely beats simple rules.** Out of time, over 38 races, its
  stop-count agreement with the field exceeds a race-length rule by 0.16
  (interval touches zero) and "always one stop" by 0.29.
- **A published benchmark is reproduced, but not beaten on its own task.** We ran
  the Cappello and Hoegh state-space model (Table 3 reproduces) and corrected its
  fuel ramp. On their 15 test races we win 2 on CRPS.

Several earlier claims did not survive re-analysis; see [Withdrawn claims](#withdrawn-claims).

## Repository layout

```
paper/
  CLAIMS.md            evidence registry: every claim, its source, caveats, and what was withdrawn
  results/             JSON written by the paper analyses (the numbers the paper quotes)
  manuscript/          generated numbers, tables and figures used by the paper
src/tyredeg/          the package
  data/                FastF1 caching, lap cleaning, long-run tagging, schedule, tyre allocation
  models/              within-event design, mixed/REML estimators, hierarchical Stan model
  validation/          power, calibration, practice-to-race transfer, scoring, benchmark helpers
  strategy/            pit-loss estimation, stint support bounds, optimiser, uncertainty
  paper/               analysis code behind paper/results (not imported by the packages above)
  artifacts/           typed schema for the pipeline's JSON artifacts
  config.py            paths and constants
scripts/
  01_cache_sessions.py ... 13_design_sweep.py   numbered pipeline stages
  run_pipeline.py      runs the core stages in order
  paper/               p1-p6 analyses, make_* asset builders, check_claims.py
benchmark/             reproduction of the published benchmark (R script, published Table 4)
data/                  allocation tables, schedule cache, artifacts (JSON); caches are gitignored
tests/                 offline tests
```

## Installation

Python 3.10 or newer. FastF1 3.8.3 requires pandas below 3.0, which `pyproject.toml` pins.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Stan-based parts (the hierarchical model and the benchmark reproduction) need
CmdStan 2.39 and a C++ toolchain: `python -m cmdstanpy.install_cmdstan`. The tests
and the claims check do not. A LaTeX distribution (`pdflatex`, `bibtex`) is needed
only to build the manuscript.

```bash
pytest -q                                  # offline; tests needing cached data skip themselves
python scripts/paper/check_claims.py       # every number in paper/CLAIMS.md against its JSON source
```

## Data

Raw timing data are not redistributed; they are fetched from the original services
and subject to their terms.

- **Lap timing:** the [FastF1](https://docs.fastf1.dev/) package (3.8.3), which
  caches sessions under `data/fastf1_cache/`. `python scripts/01_cache_sessions.py`
  downloads a season and writes `data/processed/laps*.parquet` (see the script's
  header for event and session options). The 2022-2025 holdout seasons use the same
  script with `--season`.
- **Telemetry:** the [OpenF1](https://openf1.org/) public API, because the live-timing
  host that FastF1 reads car data from returned HTTP 401 for every session when we
  tried. `python scripts/paper/p6_telemetry.py download` fetches and reduces it,
  resumably and rate-limited, into `data/processed/telemetry_proxies/`.
- **Tyre allocation** (which of C1-C5 each weekend uses) is in no feed. It is
  transcribed from Pirelli's per-race previews into `data/allocation.json` (2026) and
  `data/allocation_history.json` (2022-2025).
- **Benchmark code:** `bash scripts/fetch_reference.sh` clones the benchmark
  authors' repository into `benchmark/upstream/` and migrates their Stan models to
  current syntax in `benchmark/stan/`. Their code carries no licence, so neither is
  committed.

The committed JSON in `paper/results/` and `data/artifacts/` lets the claims check,
the paper assets and most tests run from a fresh clone with none of the above.

## Reproducing the results

Stage scripts write `data/artifacts/*.json` (see
[`data/artifacts/README.md`](data/artifacts/README.md)); the `scripts/paper/`
analyses write `paper/results/*.json`. Run in this order, from the repository root,
after the data are cached. Runtimes are approximate, on a laptop.

| step | command | writes | time |
|---|---|---|---|
| 1 | `python scripts/01_cache_sessions.py` | `data/processed/laps*.parquet` | network, tens of minutes |
| 2 | `python scripts/run_pipeline.py` | artifacts from stages 03-08 | about 6 min (validation dominates; `--skip-validation` takes under a minute) |
| 3 | `python scripts/10_pit_loss_by_status.py` | `paper/results/pit_loss.json` | under a minute |
| 4 | `python scripts/11_stint_pool.py` | `data/artifacts/stint_pool.json` | under a minute |
| 5 | `python scripts/12_session_skill.py` | `paper/results/session_skill.json` | under a minute |
| 6 | `python scripts/13_design_sweep.py` | `paper/results/design_sweep.json` | under a minute |
| 7 | `python scripts/paper/p3_model.py` | `model.json` | about 10 min |
| 8 | `python scripts/paper/p1_management.py` | `management.json` | 10-15 min |
| 9 | `python scripts/paper/p2_transfer.py` | `transfer.json` | 3-6 min |
| 10 | `python scripts/paper/p4_benchmark.py` | `benchmark.json` | about 1 h with fits; `--no-fits` uses cached fits; `--season` re-runs their model on 14 more races (hours) |
| 11 | `python scripts/paper/p5_strategy.py` | `strategy.json` | about 5 min |
| 12 | `python scripts/paper/p6_telemetry.py download` then `analyse` | `telemetry.json` | download: hours (resumable); analyse: minutes |
| 13 | `python scripts/paper/build_manuscript_assets.py` | `generated/*.tex`, `figures/*.pdf` | seconds |

`run_pipeline.py` accepts `--refresh` (step 1 first), `--event` and
`--skip-validation`. Seeds are fixed; reruns reproduce every number except
timestamp and runtime fields.

## Paper assets

The numbers, tables and figures the paper quotes are generated from the JSON
results into `paper/manuscript/generated/` and `paper/manuscript/figures/`:

```bash
python scripts/paper/build_manuscript_assets.py          # add --check to fail if stale
```

The manuscript itself will be added to this repository, with its citation, once the
paper is published.

## How the numbers are guarded

`paper/CLAIMS.md` lists each claim with its source file and path, and carries
machine-readable checks in fenced `claims` blocks, one per quoted number:

```
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "mean"], "value": 0.041, "places": 4}
```

`python scripts/paper/check_claims.py` compares each to the current JSON after
rounding to the stated places and exits non-zero on any mismatch. Section A is what
the paper may claim, B what was withdrawn, C the open limitations a referee will
raise, and D pending items. `tests/test_paper_assets.py` additionally requires that
every number in section A has a macro in `generated/numbers.tex` whose text equals
the claim, and that the committed tables are what a fresh build writes, so prose
cannot drift from the JSON.

## Withdrawn claims

Earlier versions of this work made claims that a hostile re-analysis did not
support: tyre nursing ("drivers nurse the soft tyre"), a 35.6% practice-to-race
improvement, FP2 as the best predictor, a warm-up fix, a circuit-versus-compound
spread test, strict compound ordering, a strategy call that "matches the teams",
a 0.84x virtual-safety-car cost and a 0.043 s/kg fuel coefficient, among others.
Each is listed in [`paper/CLAIMS.md`](paper/CLAIMS.md) section B with the old
value, what the re-analysis found, and where.

## Licence

MIT; see [`LICENSE`](LICENSE). Raw timing data are subject to the terms of FastF1
and OpenF1. The benchmark authors' code is not redistributed.

## Citation

Citation details will be added on publication.

```
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```
