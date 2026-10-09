# Benchmark — verification only

**This directory is not part of the product.** It exists so we can say "we ran
their model", not "we read their paper". Nothing in `src/tyredeg/` imports from
here, and the package installs and tests clean with no R present.

## What is here

```
upstream/   The authors' repo, unmodified. Reference only, never redistributed.
published_table4.json   Journal Table 4 (and Austria Tables 2, 3, 5, 6), hand-transcribed with page references.
stan/       Their four .stan models, migrated to Stan 2.32+ array syntax.
results/    Flat CSV copies of our reproduction (written by scripts/paper/p4_benchmark.py).
repro.R     The original R reproduction of Table 3 (R optional; Python is canonical).
```

## The paper

Cappello & Hoegh (2025), [arXiv:2512.00640](https://arxiv.org/abs/2512.00640) —
a Bayesian state-space model of tyre degradation, fitted to one driver across
three stints. Code: <https://github.com/colecappello12/F1_SSM_Paper>.

The journal version (Cappello & Hoegh 2026, J. Sports Analytics 12, doi 10.1177/22150218261446170, CC BY-NC 4.0) adds 19 races and an AR(1) baseline; its per-race Table 4 is transcribed, with attribution, in `published_table4.json` (the PDF itself is not redistributed). `src/tyredeg/paper/published_benchmark.py` compares our per-race CRPS with it; the section is `journal_table4` in `paper/results/benchmark.json`. Their SSM column equals the repo CSV to three decimals.

The arXiv preprint (CC BY 4.0) is downloaded to `paper/refs/` by `paper/refs/fetch.sh` and is not committed. **Their code carries no
license**, so it is fetched locally by `scripts/fetch_reference.sh` and never
committed.

## Setup (Python, canonical)

```bash
bash scripts/fetch_reference.sh                    # their repo -> benchmark/upstream/, migrated .stan -> benchmark/stan/
python scripts/paper/p4_benchmark.py               # Austria re-run + every audit below; writes paper/results/benchmark.json
python scripts/paper/p4_benchmark.py --season      # also re-run THEIR model on the other races (hours)
python scripts/08_benchmark.py                     # our head-to-head -> data/artifacts/benchmark.json
```

Their pipeline is ported to Python in `src/tyredeg/paper/benchmark_repro.py`
and run through cmdstanpy (CmdStan 2.39). R is not needed. Seeds are fixed
(their 1234 for the CV folds, 1 for the full-race fits, 0 for bootstraps).
Long fits are cached per fold in `data/processed/benchmark_fits/` so a run can
be resumed. `benchmark/upstream/` and `benchmark/stan/` hold their unlicensed
code and must not be committed.

On Windows put Rtools ahead of MinGW (`tyredeg.models.hierarchical.ensure_toolchain`
does it for you). Their `.stan` files use pre-2.32 array syntax; the migration
is mechanical (`array[TT] int Compound;`) and is applied by `fetch_reference.sh`.

## What the audit found

All numbers are in `paper/results/benchmark.json`.

| Claim | Result |
|---|---|
| Table 3 reproduces | Hard 0.0546 [0.0038, 0.1322], Medium 0.0552 [0.0065, 0.1154] vs published 0.054 [0.004, 0.133] and 0.060 [0.009, 0.120] |
| Their skew-t CV reproduces | Austria CRPS 0.2023, RMSE total 1.0815 (published 0.202 / 1.082). 11 more races at reduced iterations: correlation 0.9997 with their per-race CSV over 12 races, mean abs diff 0.0018 |
| Fuel ramp is misplaced | Up to 4.74 kg (mean 2.22 kg) at Austria; up to 20 kg elsewhere in the season |
| Fixing it matters | No. Rates move by -0.0018 (Hard) and -0.0005 (Medium) s/lap; the intervals still overlap; P(Medium > Hard) 0.528 -> 0.539 (within Monte-Carlo noise) |
| Skew-t scored on normal draws | **Not in the shipped CV.** `skewt_CV` overwrites Stan's `normal_rng` draw with `rsgt` skew-t draws. Scoring Stan's normal draws instead changes Austria CRPS by +0.0015 (0.2023 to 0.2038) and the 12-race mean by +0.0066 (3%) |
| `log_pd` omits the fuel term | True, and it is also not the skew-t. It is unused in their paper or code; lppd would shift by 215 nats on Austria |
| Their fitted fuel coefficient | 0.0155 (their ramp), 0.0164 (corrected) s/kg, about half the physical 0.030-0.035 |
| Ours | 0.020 [0.013, 0.027] at Austria 2025 and 0.036 [0.032, 0.039] at Bahrain 2024, both at the correct 110 kg start; a diffuse prior gives the same. The notes' 0.043 and the 0.0501 in an old comment do not reproduce |

### The Singapore number was our mistake

`2025_CV.qmd` removes Singapore laps after 58 before cross-validating. Our
scorer rebuilt the folds from our own lap feed, which kept them, so the
"3.83, one 32-second lap" was on laps their model was never tested on. Folds are
now taken from their retained laps. Our feed lacks a few laps theirs has (Austria
lap 30: 16 test laps to their 17); laps we cannot forecast are listed, not
dropped silently.

### Scoring

RMSPE and CRPS under their rolling-origin scheme. CRPS via `scoringrules`. The
cross-check against R's `scoringRules` (2.5e-11) was done with R installed;
`validation.scoring.crosscheck_python` re-checks the estimator without R
(closed form vs numerical integration, sample estimator vs the O(m^2) kernel
form, an analytic anchor), and `tests/test_paper_benchmark.py` runs it.

"RMSPE" is a plain RMSE in seconds, and Table 1's "Total" is a sum over stints
while the CRPS total is a mean.
