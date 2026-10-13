# Pipeline artifacts

JSON written by the numbered pipeline scripts in `scripts/` (stages 03-08 and 11).
They are committed so that `paper/CLAIMS.md`, `scripts/paper/make_numbers.py`,
`make_tables.py`, `make_figures.py` and the tests can run from a fresh clone
without refitting anything. The paper's own results live in `paper/results/`.

Shapes are defined in `src/tyredeg/artifacts/schema.py`; each script writes its
artifact through `schema.write()`, which rejects unknown names.

Anything that describes the dataset is written by the stage that *reads* the
dataset (for example `meta.json` by `03_fit_model.py`), so a count in the
provenance file cannot drift behind the model that produced it.

## What is here

| file | written by | contents |
|---|---|---|
| `meta.json` | `03_fit_model.py` | provenance and dataset size; what each fit read |
| `degradation.json` | `03_fit_model.py` | degradation rate per physical compound (C1-C5) |
| `ablation.json` | `03_fit_model.py` | naive versus within-event (deconfounded) estimates |
| `calibration.json` | `04_validate.py` | empirical coverage against nominal |
| `power.json` | `04_validate.py` | how many driver-stints separating adjacent compounds needs |
| `transfer.json` | `05_transfer.py` | practice to race, predicted versus actual (simple version; the paper's analysis is `paper/results/transfer.json`) |
| `strategy.json` | `06_strategy.py` | one event's strategy decision in full detail (worked example of the optimiser) |
| `management.json` | `07_management.py` | race against practice degradation by compound, legacy per-cell view (the paper's analysis is `paper/results/management.json`; the "drivers nurse softer tyres" claim is withdrawn, see `paper/CLAIMS.md`) |
| `benchmark.json` | `08_benchmark.py` | scored against the published benchmark model |
| `stint_pool.json` | `11_stint_pool.py` | genuine stint lengths per race and compound label, behind the strategy layer's stint floors and ceilings, so the support bounds can be rebuilt and audited |

`paper/CLAIMS.md` cites `ablation`, `benchmark`, `calibration`, `degradation`,
`management`, `meta` and `power` directly; the other files are working outputs.

## Regenerating

```bash
python scripts/run_pipeline.py              # stages 03, 04, 05, 06, 07, 08 from the cached laps
python scripts/11_stint_pool.py             # the stint pool (needs raw laps for every season)
```

Individual stages are numbered and safe to run alone. See the top-level README
for the full reproduction order, including the `paper/results/` files.
