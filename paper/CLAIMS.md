# CLAIMS: evidence registry for the paper

Generated 2026-10-30 (`model.json` and `recovery.json` regenerated 2026-11-01) from `paper/results/*.json` and `data/artifacts/*.json`. Every number below is wrapped in a check that `scripts/paper/check_claims.py` re-reads from the JSON (run by `tests/test_claims_registry.py`). If a result file is regenerated and a number moves, the test fails and this file must be updated: that is the point.

How to read an entry: **Claim** (what the paper may say, with n and a 95% interval), **Source** (JSON path), **Script** (what produces it), **Caveat** (what a referee will say). Machine-readable checks sit in the `claims` blocks; a path is a list of JSON keys/indexes. Rounding is stated per check (`places`).

**Literature rule:** only sources opened and recorded in `paper/refs/NOTES.md` (all in `paper/references.bib`) may be cited. Cappello & Hoegh is cited via the journal version (J. Sports Analytics 2026, read in full, bib key `cappelloHoegh2026`; NOTES section A) and, for the v1 numbers only, arXiv v1 (`cappello2025state`).

**Entries marked `[READS REGENERATED FILES]`** read `pit_loss.json`, `strategy.json` or `transfer.json`, which were regenerated on 30 Oct 2026 after the strategy-layer, pit-loss and transfer-stacking fixes (strategy evaluation made out-of-time, stint floors rebuilt, wide pit-loss filter, leakage fix in the stacked baseline). Their quoted numbers were re-verified against the regenerated files.

**2026-11-01 RE-ANALYSIS:** exact within-(event, lap) demeaning of compound columns replaced the approximate construction; the approximate one leaked slopes/offsets between compounds mixed on a lap (2026 C1 0.082 -> 0.041; pooled compound SD 0.0047 -> 0.0134; pooled rates now ordered by softness from C2 to C5). All model.json sections rerun; see model.json `exact_vs_approximate` and `recovery.json`. `paper/results/recovery.json` is the recovery study (simulated data with known truth): Eq. 1 with exact demeaning is unbiased under random and pace-dependent stops, and about -0.009 s/lap with 0.79 coverage under wear-triggered stops.


## A. Claims the paper can make

### A1. Degradation varies persistently by circuit, about as much as it varies race to race, and about twice as much as by compound

**Claim (primary, pooled 2022-2026).** A nested REML model with a persistent circuit random slope and an event-within-circuit random slope, fitted to 82 races at 23 circuits (20 with repeat visits; 69,395 race laps in 4,445 runs; physical C-number compounds), gives a persistent circuit slope SD of 0.0278 s/lap (95% profile-likelihood interval [0.0197, 0.0405]; parametric bootstrap [0.0157, 0.0380]), about 0.56 s over a 20-lap run (profile [0.39, 0.81]), and a race-specific (event within circuit) SD of 0.0240 s/lap (profile [0.0208, 0.0290]; bootstrap [0.0191, 0.0288]), about 0.48 s over 20 laps. The persistent share of slope variance is 0.57 (bootstrap [0.28, 0.75]). The persistent component is clearly present (REML likelihood-ratio 27.0; parametric-bootstrap p = 0.002 from 500 replicates, the floor 1/(B+1); do not quote 'p < 1e-5') and so is the race-specific one (LR 258.5). Both exceed the compound spread, by about a factor of two: the SD of the five pooled compound rates (C1 to C5, 0.044 to 0.073 s/lap, rising with softness from C2 to C5) is 0.0134 (bootstrap [0.0111, 0.0156]); the persistent circuit SD minus the compound SD has a bootstrap median of 0.0137 ([0.0022, 0.0253]) with bootstrap probability 0.992 that it is positive. Dropping 2025 (C-number re-specification) or keeping only verified nominations leaves the persistent SD at 0.0285 and 0.0277.

**Claim (secondary, 2026 only).** In 2026 every circuit was raced once, so the circuit slope SD (0.0498 s/lap; profile [0.0354, 0.0810]; bootstrap [0.0284, 0.0708]; about 1.00 s over 20 laps; LR 220.8, bootstrap p = 0.002; n = 13 circuits, 11,039 race laps, 55 to 242 runs per compound) mixes the persistent and race-specific components; do not read it as a property of the track.

**Source.** `paper/results/model.json -> circuit_vs_event` (primary), `-> circuit_effect` (2026 only)

**Script.** scripts/paper/p3_model.py (`tyredeg.paper.model_checks.circuit_vs_event`, `circuit_effect`)

**Caveat.** The persistent share is imprecise (bootstrap [0.28, 0.75]) and the circuit-component p-value sits at the bootstrap floor. C numbers are comparable within a season's range, not strictly across the 2025 re-specification; the 2022-2025 mapping comes from Pirelli nominations (80 of 82 events verified against two sources). The pooled fit keeps compounds C1 to C5 only, so it omits 38 laps in 2 runs on the C6 soft tyre of the 2025 Monaco race; the label-based season fits keep them (season rows of Table 1 sum to 69,433 laps in 4,447 runs against 69,395 in 4,445). Compounds are nested in circuits (Pirelli nominates by track), so circuit and compound effects are only partly separable. The compound-rate SD is a different kind of quantity from a variance component and includes sampling noise (noise-corrected upper bound 0.0143); the comparison is of spreads, with the bootstrap difference carrying the uncertainty. The persistent component does not say why: surface, layout and climate are candidate causes we do not identify. The per-circuit offsets of the 2026 fit are shrunken predictions: their range (0.191 s/lap) is NOT an effect size.

```claims
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "design", "n_events"], "value": 82, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "design", "n_circuits"], "value": 23, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "design", "circuits_with_repeat_visits"], "value": 20, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "design", "n_laps"], "value": 69395, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "design", "n_runs"], "value": 4445, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "circuit_persistent", "estimate"], "value": 0.0278, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "circuit_persistent", "profile_95", 0], "value": 0.0197, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "circuit_persistent", "profile_95", 1], "value": 0.0405, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "circuit_persistent", "bootstrap_95", 0], "value": 0.0157, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "circuit_persistent", "bootstrap_95", 1], "value": 0.038, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "event_specific", "estimate"], "value": 0.024, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "event_specific", "profile_95", 0], "value": 0.0208, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "event_specific", "profile_95", 1], "value": 0.029, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "event_specific", "bootstrap_95", 0], "value": 0.0191, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sd_s_per_lap", "event_specific", "bootstrap_95", 1], "value": 0.0288, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "persistent_share_of_slope_variance", "estimate"], "value": 0.57, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "persistent_share_of_slope_variance", "bootstrap_95", 0], "value": 0.28, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "persistent_share_of_slope_variance", "bootstrap_95", 1], "value": 0.75, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "circuit_persistent"], "value": 0.56, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "circuit_persistent_profile_95", 0], "value": 0.39, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "circuit_persistent_profile_95", 1], "value": 0.81, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "event_specific"], "value": 0.48, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "event_specific_profile_95", 0], "value": 0.42, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "seconds_over_20_laps", "event_specific_profile_95", 1], "value": 0.58, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "circuit_component_test", "reml_lr"], "value": 27.0, "places": 1}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "circuit_component_test", "p_parametric_bootstrap"], "value": 0.002, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "circuit_component_test", "n_bootstrap"], "value": 500, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "event_component_test", "reml_lr"], "value": 258.5, "places": 1}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rates_pooled_s_per_lap", "C1", "mean"], "value": 0.044, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rates_pooled_s_per_lap", "C2", "mean"], "value": 0.043, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rates_pooled_s_per_lap", "C3", "mean"], "value": 0.06, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rates_pooled_s_per_lap", "C4", "mean"], "value": 0.066, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rates_pooled_s_per_lap", "C5", "mean"], "value": 0.073, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_rate_sd_s_per_lap"], "value": 0.0134, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "compound_sd_raw", "bootstrap_95", 0], "value": 0.0111, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "compound_sd_raw", "bootstrap_95", 1], "value": 0.0156, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "compound_sd_noise_corrected", "bootstrap_95", 1], "value": 0.0143, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "circuit_minus_compound_raw", "median"], "value": 0.0137, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "circuit_minus_compound_raw", "bootstrap_95", 0], "value": 0.0022, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "circuit_minus_compound_raw", "bootstrap_95", 1], "value": 0.0253, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "compound_vs_variance", "circuit_minus_compound_raw", "p_circuit_greater"], "value": 0.992, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sensitivity", "without_2025", "circuit_sd"], "value": 0.0285, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sensitivity", "verified_nominations_only", "circuit_sd"], "value": 0.0277, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_vs_event", "sensitivity", "verified_nominations_only", "n_events"], "value": 80, "places": 0}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "estimate"], "value": 0.0498, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "profile_likelihood_95", 0], "value": 0.0354, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "profile_likelihood_95", 1], "value": 0.081, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "parametric_bootstrap_percentile_95", 0], "value": 0.0284, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "parametric_bootstrap_percentile_95", 1], "value": 0.0708, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "effect_in_seconds_over_20_lap_stint", "circuit_sd", "seconds"], "value": 1.0, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_effect", "effect_in_seconds_over_20_lap_stint", "circuit_sd", "profile_95", 0], "value": 0.71, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_effect", "effect_in_seconds_over_20_lap_stint", "circuit_sd", "profile_95", 1], "value": 1.62, "places": 2}
{"file": "paper/results/model.json", "path": ["circuit_effect", "lr_test", "reml_lr_statistic"], "value": 220.8, "places": 1}
{"file": "paper/results/model.json", "path": ["circuit_effect", "lr_test", "p_parametric_bootstrap"], "value": 0.002, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_effect", "lr_test", "n_bootstrap"], "value": 500, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "n_runs"], "value": 55, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "n_runs"], "value": 242, "places": 0}
```

### A2. Per-compound rates

**Claim.** Race degradation per physical compound, 2026, with every fixed-effect column demeaned exactly within (event, lap), s/lap with 95% intervals: C1 0.0410 [0.0079, 0.0742] (n = 55 runs); C2 0.0373 [0.0081, 0.0665] (n = 191 runs); C3 0.0544 [0.0258, 0.0830] (n = 242 runs); C4 0.0633 [0.0346, 0.0921] (n = 223 runs); C5 0.0385 [0.0090, 0.0680] (n = 109 runs). The ordering is C4 > C3 > C1 > C5 > C2, not monotone in hardness (`monotone_softer_degrades_faster` is False). No pair of rate intervals separates at 95% (`separated_pairs` has 0 entries); the adjacent contrasts, which share the random effects, exclude zero for C3-C2 (+0.0171 [+0.0041, +0.0300]), C4-C3 (+0.0089 [+0.0026, +0.0153]) and C5-C4 (-0.0248 [-0.0332, -0.0164]), not for C2-C1 (-0.0037 [-0.0216, +0.0141]). Within event, the 2026 trend per step softer is about zero: -0.0010 s/lap [-0.0047, +0.0028]. Pooled over 2022-2026 (82 races, nested model, C-number compounds) the rates are C1 0.044, C2 0.043, C3 0.060, C4 0.066, C5 0.073 s/lap, rising with softness from C2 to C5; C1 and C2 are indistinguishable (C2-C1 -0.0016 [-0.0062, +0.0031]). By season, the within-event trend per step softer (HARD/MEDIUM/SOFT labels) is +0.0238 [+0.0201, +0.0274] in 2022, +0.0101 [+0.0072, +0.0131] in 2023, +0.0020 [-0.0005, +0.0046] in 2024, +0.0127 [+0.0096, +0.0159] in 2025 and -0.0009 [-0.0046, +0.0029] in 2026.

**Source.** `paper/results/model.json -> shipped_fit, compound_within_event, season_robustness, circuit_vs_event (pooled rates), detectable_from_fit (pooled contrasts), exact_vs_approximate; data/artifacts/degradation.json -> separated_pairs`

**Script.** scripts/paper/p3_model.py; scripts/03_fit_model.py (artifact)

**Caveat.** The earlier reading that softer compounds degrade slower in 2026 came from the approximate column construction and no longer holds (B6; `exact_vs_approximate`). Do NOT claim C5 below C4 in 2026 is driver tyre-management (see A8 and B1). Cause unexplained; candidates are compound x race-phase pace offsets and selection in who stays on which tyre. C-numbers are ordinal within a season's range and were re-specified each year.

```claims
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "mean"], "value": 0.041, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "lo"], "value": 0.0079, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "hi"], "value": 0.0742, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "n_runs"], "value": 55, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C2", "mean"], "value": 0.0373, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C2", "lo"], "value": 0.0081, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C2", "hi"], "value": 0.0665, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C2", "n_runs"], "value": 191, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "mean"], "value": 0.0544, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "lo"], "value": 0.0258, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "hi"], "value": 0.083, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "n_runs"], "value": 242, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C4", "mean"], "value": 0.0633, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C4", "lo"], "value": 0.0346, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C4", "hi"], "value": 0.0921, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C4", "n_runs"], "value": 223, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C5", "mean"], "value": 0.0385, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C5", "lo"], "value": 0.009, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C5", "hi"], "value": 0.068, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C5", "n_runs"], "value": 109, "places": 0}
{"file": "paper/results/model.json", "path": ["shipped_fit", "adjacent_contrasts_s_per_lap", "C5-C4", "mean"], "value": -0.0248, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "adjacent_contrasts_s_per_lap", "C5-C4", "lo"], "value": -0.0332, "places": 4}
{"file": "paper/results/model.json", "path": ["shipped_fit", "adjacent_contrasts_s_per_lap", "C5-C4", "hi"], "value": -0.0164, "places": 4}
{"file": "paper/results/model.json", "path": ["compound_within_event", "within_event_fixed_circuit", "trend_s_per_lap_per_step_softer", "mean"], "value": -0.001, "places": 4}
{"file": "paper/results/model.json", "path": ["compound_within_event", "within_event_fixed_circuit", "trend_s_per_lap_per_step_softer", "lo"], "value": -0.0047, "places": 4}
{"file": "paper/results/model.json", "path": ["compound_within_event", "within_event_fixed_circuit", "trend_s_per_lap_per_step_softer", "hi"], "value": 0.0028, "places": 4}
{"file": "paper/results/model.json", "path": ["season_robustness", "2022", "trend_per_step_softer", "within_event", "mean"], "value": 0.0238, "places": 4}
{"file": "paper/results/model.json", "path": ["season_robustness", "2022", "trend_per_step_softer", "within_event", "lo"], "value": 0.0201, "places": 4}
{"file": "paper/results/model.json", "path": ["season_robustness", "2022", "trend_per_step_softer", "within_event", "hi"], "value": 0.0274, "places": 4}
```

### A3. Naive vs fuel-corrected vs deconfounded

**Claim.** Same laps, same physical-compound grouping, three estimators (s/lap): C1 naive -0.035 / fuel-corrected +0.034 / deconfounded +0.041; C2 naive +0.201 / fuel-corrected +0.038 / deconfounded +0.037; C3 naive -0.037 / fuel-corrected +0.048 / deconfounded +0.054; C4 naive +0.016 / fuel-corrected +0.033 / deconfounded +0.063; C5 naive -0.070 / fuel-corrected +0.016 / deconfounded +0.039. The naive no-controls slope has the wrong sign on 3 of 5 compounds (C1, C3, C5; the sign list is checked below). The fuel-corrected within-stint 'practitioner' regression is much closer to ours, differing by 0.007 s/lap at C1 and 0.006 at C3 (largest gap 0.030 at C4). Fuel coefficient used: 0.033 s/kg.

**Source.** `paper/results/model.json -> ablation; data/artifacts/ablation.json (same numbers)`

**Script.** scripts/paper/p3_model.py; scripts/03_fit_model.py (artifact)

**Caveat.** The honest claim is 'naive analyses can be wrong in sign', not 'deconfounding beats the careful analysis'. The practitioner comparator does not control track evolution or traffic, so it is a fair-but-not-best-practice comparator, and we have no ground truth for the rate.

```claims
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C1", "naive_run_clustered", "mean"], "value": -0.035, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C1", "practitioner_fuel_corrected_within_stint", "mean"], "value": 0.034, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C1", "deconfounded", "mean"], "value": 0.041, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C2", "naive_run_clustered", "mean"], "value": 0.201, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C2", "practitioner_fuel_corrected_within_stint", "mean"], "value": 0.038, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C2", "deconfounded", "mean"], "value": 0.037, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C3", "naive_run_clustered", "mean"], "value": -0.037, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C3", "practitioner_fuel_corrected_within_stint", "mean"], "value": 0.048, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C3", "deconfounded", "mean"], "value": 0.054, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C4", "naive_run_clustered", "mean"], "value": 0.016, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C4", "practitioner_fuel_corrected_within_stint", "mean"], "value": 0.033, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C4", "deconfounded", "mean"], "value": 0.063, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C5", "naive_run_clustered", "mean"], "value": -0.07, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C5", "practitioner_fuel_corrected_within_stint", "mean"], "value": 0.016, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C5", "deconfounded", "mean"], "value": 0.039, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C1"], "value": 0.007, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C3"], "value": 0.006, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "fuel_s_per_kg_central"], "value": 0.033, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "naive_negative_compounds"], "equals": ["C1", "C3", "C5"]}
```

### A4. Why the shipped intervals are wider than a lap-level fit

**Claim.** Treating laps as independent understates uncertainty. Interval width relative to a lap-level circuit model: no-circuit lap-level model 0.15x (C3) to 0.43x (C1); the shipped model (run intercept and run slope) is 1.32x (C1) to 1.39x (C5) wider; residual lag-1 autocorrelation within runs 0.36 (lap-level circuit model) falling to 0.22 (shipped).

**Source.** `paper/results/model.json -> inference_comparison`

**Script.** scripts/paper/p3_model.py

**Caveat.** Cluster-robust OLS comparators (E, F) disagree with each other; the shipped REML is one defensible choice, not the unique one. Engine agrees with statsmodels to 4e-6 s/lap in rates (model.json -> engine_agreement).

```claims
{"file": "paper/results/model.json", "path": ["inference_comparison", "width_ratio_vs_B", "A_lap_level_no_circuit", "C3"], "value": 0.15, "places": 2}
{"file": "paper/results/model.json", "path": ["inference_comparison", "width_ratio_vs_B", "A_lap_level_no_circuit", "C1"], "value": 0.43, "places": 2}
{"file": "paper/results/model.json", "path": ["inference_comparison", "width_ratio_vs_B", "D_run_intercept_and_slope_SHIPPED", "C3"], "value": 1.38, "places": 2}
{"file": "paper/results/model.json", "path": ["inference_comparison", "width_ratio_vs_B", "D_run_intercept_and_slope_SHIPPED", "C1"], "value": 1.32, "places": 2}
{"file": "paper/results/model.json", "path": ["inference_comparison", "residual_lag1_autocorrelation_within_runs", "B_lap_level_circuit"], "value": 0.36, "places": 2}
{"file": "paper/results/model.json", "path": ["inference_comparison", "residual_lag1_autocorrelation_within_runs", "D_run_intercept_and_slope_SHIPPED"], "value": 0.22, "places": 2}
```

### A5. Calibration of the rate intervals

**Claim.** Leave-one-circuit-out coverage of each circuit's own fitted rate by the pooled model's predictive interval: nominal 80% -> 0.667 (24/36 cells, Wilson 95% [0.50, 0.80]); nominal 95% -> 0.889. Dropping the circuit variance collapses 80% coverage to 0.389. Simulation-based calibration (400 datasets, data drawn from the fitted model): pooled coverage at 80% 0.762. Lap-level held-out coverage at 80% is 0.806 but lap scatter dominates, so it is an easy bar.

**Source.** `paper/results/model.json -> rate_calibration, lap_level_calibration; data/artifacts/calibration.json`

**Script.** scripts/paper/p3_model.py; scripts/04_validate.py (artifact)

**Caveat.** Quote the per-circuit coverage (about 67%, n = 36 cells, Wilson interval wide; the 80% intervals under-cover, and the Wilson upper bound only reaches 0.80) and not the 80.6% lap-level figure as the headline. SBC only checks the estimator against its own model, so it cannot detect misspecification.

```claims
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "coverage"], "value": 0.667, "places": 3}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "k"], "value": 24.0, "places": 0}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "n"], "value": 36, "places": 0}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "wilson_95", 0], "value": 0.5, "places": 2}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "wilson_95", 1], "value": 0.8, "places": 2}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.95", "coverage"], "value": 0.889, "places": 3}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "no_circuit_variance", "coverage", "0.80", "coverage"], "value": 0.389, "places": 3}
{"file": "paper/results/model.json", "path": ["rate_calibration", "simulation_based_calibration", "coverage_of_true_rate", "0.80", "pooled"], "value": 0.762, "places": 3}
{"file": "paper/results/model.json", "path": ["lap_level_calibration", "coverage_80"], "value": 0.806, "places": 3}
```

### A6. Power: what public timing data can separate

**Claim.** Under Pirelli's actual 2026 nominations (the "realistic" design: stints only on the compounds each event nominated; C3 is nominated everywhere, so any two compounds are linked by a chain of within-event contrasts) and the runs we hold, the minimum detectable compound gap at 80% power is 0.0190 s/lap for C1 against C2 (246 runs, the weakest-linked adjacent pair; simulation that refits the variances gives 0.019, Monte-Carlo 95% [0.0188, 0.0196], and 0.023 with a t reference on 12 df) and 0.0111 to 0.0139 s/lap for the other adjacent pairs (C2-C3 0.0131, C3-C4 0.0111, C4-C5 0.0139). Moving the circuit-SD to either end of its profile interval leaves the realistic figure essentially unchanged (0.0189 to 0.0190); raising the run-slope SD by 50% lifts it to 0.0262. A balanced design (both compounds at every circuit) at 832 stints detects 0.0080 (0.0111 under the stressed run-slope SD); a split design (disjoint circuits) at 832 stints only 0.0781 (0.0558 to 0.1265 over the circuit-SD interval). Compounds at disjoint circuits cannot be separated whatever the sample: circuit variance enters the contrast, so power has a ceiling (0.055 at 0.006 s/lap rising to 0.192 at 0.03 s/lap). Runs needed for 80% power (analytic): 0.006 s/lap needs 2,482 realistic C1-C2 runs (1,489 balanced); 0.01 needs 893 (536); 0.02 needs 223 (134); 0.03 needs 99 (60). Simulations that refit the variances match at 0.02: realistic 220 (Monte-Carlo 95% [210, 233]), balanced 138 [128, 147]; simulated points were run up to 2,000 runs (the 0.006 s/lap points above that cap, balanced at 2,236 and 3,276 runs and realistic at 2,470 to 5,460, are covered by the analytic curve). At zero effect the Wald z test rejects in 0.056 (balanced) and 0.050 (realistic) of 2,000 replicates (Monte-Carlo SE 0.005), close to the nominal 0.05.

**Source.** `paper/results/model.json -> min_detectable_effect` (final); `-> power_extended` (produced by `p3_model.py --only compound_spread_2026 power_extended`); `data/artifacts/power.json` (0.006 s/lap, 3 runs: the benchmark study). The first power analysis (`-> power`, 100 simulations per point, balanced-versus-split only; 1702 runs for 0.006 s/lap balanced) is superseded in the text and kept only in B8.

**Script.** scripts/paper/p3_model.py (`tyredeg.paper.model_checks.min_detectable_effect`, `.power_extended`; `tyredeg.validation.power`)

**Caveat.** All power figures come from the fitted model (simulated data), not from observed compound differences, and say what the design could detect, not what is there. "Analytic" uses the generalised-least-squares covariance at known variances (a lower bound on the detectable gap: no Monte-Carlo error); "simulated" refits the variances and applies the Wald test used in the paper, with Monte-Carlo error. The Wald z test holds its nominal size here (0.056 and 0.050 at nominal 0.05); the t(12) reference is conservative (0.032 and 0.011), so its minimum detectable effect is larger. The realistic design reuses the 2026 Pirelli nominations (supplementary table) and 20-lap stints; it does not model pit strategy or who runs which compound within an event. The run counts are driver-stints of the two compared compounds, not all runs.

```claims
{"file": "paper/results/model.json", "path": ["dataset", "race_fit", "n_runs"], "value": 820, "places": 0}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C1_C2", "at_actual_pair_runs", "n_stints"], "value": 246, "places": 0}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C1_C2", "at_actual_pair_runs", "point", "mde_80_s_per_lap"], "value": 0.019, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C2_C3", "at_actual_pair_runs", "point", "mde_80_s_per_lap"], "value": 0.0131, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C3_C4", "at_actual_pair_runs", "point", "mde_80_s_per_lap"], "value": 0.0111, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C4_C5", "at_actual_pair_runs", "point", "mde_80_s_per_lap"], "value": 0.0139, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C1_C2", "at_actual_pair_runs", "circuit_sd_low", "mde_80_s_per_lap"], "value": 0.0189, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C1_C2", "at_actual_pair_runs", "circuit_sd_high", "mde_80_s_per_lap"], "value": 0.019, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "realistic_C1_C2", "at_actual_pair_runs", "circuit_high_and_run_slope_x1.5", "mde_80_s_per_lap"], "value": 0.0262, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "balanced", "at_N_832", "n_stints"], "value": 832, "places": 0}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "balanced", "at_N_832", "point", "mde_80_s_per_lap"], "value": 0.008, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "balanced", "at_N_832", "circuit_high_and_run_slope_x1.5", "mde_80_s_per_lap"], "value": 0.0111, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "split", "at_N_832", "point", "mde_80_s_per_lap"], "value": 0.0781, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "split", "at_N_832", "circuit_sd_low", "mde_80_s_per_lap"], "value": 0.0558, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "analytic_known_variance", "split", "at_N_832", "circuit_sd_high", "mde_80_s_per_lap"], "value": 0.1265, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "simulated_refit", "realistic_C1_C2", "mde_80_wald_z", "crossing"], "value": 0.019, "places": 3}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "simulated_refit", "realistic_C1_C2", "mde_80_wald_z", "mc_ci95", 0], "value": 0.0188, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "simulated_refit", "realistic_C1_C2", "mde_80_wald_z", "mc_ci95", 1], "value": 0.0196, "places": 4}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "simulated_refit", "realistic_C1_C2", "mde_80_t_df12", "crossing"], "value": 0.023, "places": 3}
{"file": "paper/results/model.json", "path": ["min_detectable_effect", "simulated_refit", "realistic_C1_C2", "reps"], "value": 1000, "places": 0}
```

The `power_extended` keys below were merged into model.json by the 2026-11-01 rerun and are checked like the rest; before they existed the checker skipped them. Once the key exists a line without a value FAILS and asks for `python scripts/paper/check_claims.py --promote-pending`, which fills in the rounded values and moves the lines into a checked `claims` block. Lines with `equals` are checked as soon as the key exists. make_numbers.py reads the rounding from the `places` here.

```claims
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.006", "balanced", "point"], "places": 0, "value": 1489.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.006", "realistic_C1_C2", "point"], "places": 0, "value": 2482.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.006", "split", "point"], "equals": null}
{"file": "paper/results/model.json", "path": ["power_extended", "split_design_power_ceiling", "effect_0.006", "point"], "places": 3, "value": 0.055}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.01", "balanced", "point"], "places": 0, "value": 536.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.01", "realistic_C1_C2", "point"], "places": 0, "value": 893.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.01", "split", "point"], "equals": null}
{"file": "paper/results/model.json", "path": ["power_extended", "split_design_power_ceiling", "effect_0.01", "point"], "places": 3, "value": 0.065}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.02", "balanced", "point"], "places": 0, "value": 134.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.02", "realistic_C1_C2", "point"], "places": 0, "value": 223.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.02", "split", "point"], "equals": null}
{"file": "paper/results/model.json", "path": ["power_extended", "split_design_power_ceiling", "effect_0.02", "point"], "places": 3, "value": 0.112}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.03", "balanced", "point"], "places": 0, "value": 60.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.03", "realistic_C1_C2", "point"], "places": 0, "value": 99.0}
{"file": "paper/results/model.json", "path": ["power_extended", "analytic_crossing_80pct_stints", "effect_0.03", "split", "point"], "equals": null}
{"file": "paper/results/model.json", "path": ["power_extended", "split_design_power_ceiling", "effect_0.03", "point"], "places": 3, "value": 0.192}
{"file": "paper/results/model.json", "path": ["power_extended", "split_design_power_ceiling", "effect_0.03", "circuit_sd_low"], "places": 3, "value": 0.333}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "balanced", "crossing_80_wald_z", "crossing"], "places": 0, "value": 138.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "balanced", "crossing_80_wald_z", "mc_ci95", 0], "places": 0, "value": 128.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "balanced", "crossing_80_wald_z", "mc_ci95", 1], "places": 0, "value": 147.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "realistic_C1_C2", "crossing_80_wald_z", "crossing"], "places": 0, "value": 220.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "realistic_C1_C2", "crossing_80_wald_z", "mc_ci95", 0], "places": 0, "value": 210.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "realistic_C1_C2", "crossing_80_wald_z", "mc_ci95", 1], "places": 0, "value": 233.0}
{"file": "paper/results/model.json", "path": ["power_extended", "simulated", "effect_0.02", "balanced", "reps"], "places": 0, "value": 1000.0}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "balanced", "wald_z_reject_rate"], "places": 3, "value": 0.056}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "balanced", "t_df12_reject_rate"], "places": 3, "value": 0.032}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "realistic_C1_C2", "wald_z_reject_rate"], "places": 3, "value": 0.05}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "realistic_C1_C2", "t_df12_reject_rate"], "places": 3, "value": 0.011}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "balanced", "reps"], "places": 0, "value": 2000.0}
{"file": "paper/results/model.json", "path": ["power_extended", "type_I_error_at_zero_effect", "balanced", "wald_z_mc_se"], "places": 3, "value": 0.005}
```

### A7. Race degradation is about half of practice degradation on matched used sets

**Claim.** PRIMARY design (management.json `referee_additions`): the same within-run estimator in race and practice, tyre ages 6-16, used sets only (the feed's FreshTyre flag false) in both contexts. On MEDIUM+SOFT cells (82 cells) the pooled race slope is 0.0710 s/lap and the practice slope 0.1274; the gap is -0.0564 s/lap [-0.096, -0.019] and race degradation is 0.56 of practice degradation (95% event-season cluster bootstrap [0.36, 0.81]). Over all 105 cells in 67 event-seasons the gap is -0.0586 [-0.093, -0.024]. ROBUSTNESS rows: start-age matching (practice runs starting at age 5-9, race stints at age >= 5; the earlier headline) gives -0.0994 [-0.154, -0.050], ratio 0.38 [0.11, 0.63], 76 MEDIUM+SOFT cells (97 cells, 60 event-seasons); the intersection (used sets AND start-age matched) gives -0.0578 [-0.106, -0.012], ratio 0.61, 48 MEDIUM+SOFT cells (56 cells, 48 event-seasons); no matching at all gives ratio 0.43 and gap -0.079. A practice-versus-practice placebo (FP2 against FP1+FP3 together, same used-set design; its orientation is arbitrary) gives a MEDIUM+SOFT gap of -0.006 [-0.053, 0.037] (63 cells; 76 cells overall), so the placebo-adjusted gap is -0.050 [-0.106, +0.007] (placebo share of the gap 0.11).

**Why used sets are the primary design.** Start-age matching mostly selects race stints that resumed after a gap in the lap record rather than heat-cycled used sets: of 1,411 race runs starting at tyre age >= 5, 416 are used sets (FreshTyre false) and 908 are new-set stints split by a gap; in practice 374 of 407 runs starting at age 5-9 are used sets (`referee_additions.set_state_counts`).

**Source.** `paper/results/management.json -> referee_additions.designs.{used_sets_only_FreshTyre_false, start_age_proxy_matched, used_sets_and_start_age_matched}, referee_additions.set_state_counts, referee_additions.reconcile_race_compound_trend; post_review_additions.level_gap_unmatched_pooled`

**Script.** scripts/paper/p1_management.py (`tyredeg.paper.management`)

**Caveat.** Observational. Race stints and practice runs differ in fuel, traffic, run plan and purpose; the placebo bounds session-to-session differences but not these. The gap is the lead estimand because the ratio of two noisy slopes is unstable and depends on the assumed fuel correction (which cancels in the gap). The used-set design and the start-age design were both added after an independent review and are not in the original pre-registration (management.json -> post_review_additions.provenance, referee_additions.provenance). The Holm-adjusted pairwise tests (A8) use run fixed effects over all events with both contexts, not the used-set subset.

```claims
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "gap"], "value": -0.0564, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "gap_ci95", 0], "value": -0.096, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "gap_ci95", 1], "value": -0.019, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "ratio"], "value": 0.56, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 0], "value": 0.36, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 1], "value": 0.81, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "n"], "value": 82, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "race"], "value": 0.071, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "MEDIUM+SOFT", "practice"], "value": 0.1274, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "n_cells"], "value": 105, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "n_event_seasons"], "value": 67, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "ALL", "gap"], "value": -0.0586, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "ALL", "gap_ci95", 0], "value": -0.093, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "ALL", "gap_ci95", 1], "value": -0.024, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_gap", "ALL", "ratio"], "value": 0.53, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "gap"], "value": -0.0994, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "gap_ci95", 0], "value": -0.154, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "gap_ci95", 1], "value": -0.05, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "ratio"], "value": 0.38, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 0], "value": 0.12, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 1], "value": 0.62, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "MEDIUM+SOFT", "n"], "value": 76, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "n_cells"], "value": 97, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "start_age_proxy_matched", "level_gap", "n_event_seasons"], "value": 60, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "gap"], "value": -0.0578, "places": 4}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "gap_ci95", 0], "value": -0.106, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "gap_ci95", 1], "value": -0.012, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "ratio"], "value": 0.61, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 0], "value": 0.39, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "ratio_ci95", 1], "value": 0.9, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "MEDIUM+SOFT", "n"], "value": 48, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "n_cells"], "value": 56, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap", "n_event_seasons"], "value": 48, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_placebo_fp2_minus_fp1fp3", "MEDIUM+SOFT", "gap"], "value": -0.006, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_placebo_fp2_minus_fp1fp3", "MEDIUM+SOFT", "gap_ci95", 0], "value": -0.053, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_placebo_fp2_minus_fp1fp3", "MEDIUM+SOFT", "gap_ci95", 1], "value": 0.037, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_placebo_fp2_minus_fp1fp3", "MEDIUM+SOFT", "n"], "value": 63, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "level_placebo_fp2_minus_fp1fp3", "n_cells"], "value": 76, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "placebo_adjusted_gap", "MEDIUM+SOFT", "placebo_adjusted_gap"], "value": -0.05, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "placebo_adjusted_gap", "MEDIUM+SOFT", "placebo_adjusted_gap_ci95_joint_bootstrap", 0], "value": -0.106, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "placebo_adjusted_gap", "MEDIUM+SOFT", "placebo_adjusted_gap_ci95_joint_bootstrap", 1], "value": 0.007, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "designs", "used_sets_only_FreshTyre_false", "placebo_adjusted_gap", "MEDIUM+SOFT", "placebo_share_of_gap"], "value": 0.11, "places": 2}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "race", "start_age_ge_5", "n_runs"], "value": 1411, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "race", "start_age_ge_5", "genuinely_used_set_FreshTyre_false"], "value": 416, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "race", "start_age_ge_5", "new_set_but_run_starts_after_stint_start_gap_split"], "value": 908, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "practice_race_sims", "start_age_5_to_9", "n_runs"], "value": 407, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "practice_race_sims", "start_age_5_to_9", "genuinely_used_set_FreshTyre_false"], "value": 374, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "race", "n_used_set_runs"], "value": 563, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "race", "n_runs"], "value": 2770, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "practice_race_sims", "n_used_set_runs"], "value": 858, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "set_state_counts", "practice_race_sims", "n_runs"], "value": 968, "places": 0}
{"file": "paper/results/management.json", "path": ["referee_additions", "reconcile_race_compound_trend", "settings", "all_seasons|window_6_16", "race_only_trend_by_estimator", "within_run", "beta_per_step_softer"], "value": 0.009, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "reconcile_race_compound_trend", "settings", "all_seasons|window_6_16", "race_only_trend_by_estimator", "within_run", "ci95", 0], "value": -0.003, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "reconcile_race_compound_trend", "settings", "all_seasons|window_6_16", "race_only_trend_by_estimator", "within_run", "ci95", 1], "value": 0.021, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "reconcile_race_compound_trend", "explanation", "race_only_trend_2026_window_6_16", "within_run"], "value": -0.002, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "reconcile_race_compound_trend", "explanation", "race_only_trend_all_seasons_window_6_16", "field_origin"], "value": -0.026, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_gap_unmatched_pooled", "MEDIUM+SOFT", "ratio"], "value": 0.43, "places": 2}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_gap_unmatched_pooled", "MEDIUM+SOFT", "gap"], "value": -0.079, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_gap_unmatched_pooled", "MEDIUM+SOFT", "n"], "value": 118, "places": 0}
```

### A8. Nursing is not established (an honest negative)

**Claim.** Compound-specific tyre nursing is NOT established. The two-way-FE difference-in-differences (race contrast minus practice contrast, per event) gives SOFT-minus-HARD -0.0706 s/lap [-0.134, -0.013] (21 events, permutation p two-sided 0.028; Holm-adjusted over the three pairs 0.085, so not significant after adjustment) but SOFT-minus-MEDIUM -0.0055 [-0.048, +0.035] and MEDIUM-minus-HARD -0.0398 [-0.085, +0.005]. A uniform damping of every compound by the same factor would by itself imply an additive gradient of -0.019 per softness step at k = 0.5 (observed -0.028), and the practice-vs-practice placebo shows a SOFT-minus-MEDIUM gap of -0.124 [-0.207, -0.033]. Original pre-stated additive trend: -0.0280 s/lap per step [-0.063, +0.014], permutation p one-sided 0.070, two-sided 0.137; the test's minimum detectable effect is 0.060, so 'not supported' there is underpowered, not a null. Decision flags (computed from the pre-stated rule): H1 additive interaction supported = False.

**Source.** `paper/results/management.json -> post_review_additions.twfe_gradient, uniform_damping, power_additive_trend; primary.trend; decision`

**Script.** scripts/paper/p1_management.py

**Caveat.** This is a statement about what the data cannot show, with power caveats; do not write 'drivers do not nurse tyres'. 74-specification multiverse (management.json -> multiverse_summary): 70 of 74 betas are negative but the pre-stated decision rule is not met on the primary spec. The cross-sectional race estimator is not identified (selection on pit timing, negative SOFT slopes) and its output must not be cited.

```claims
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "estimate_s_per_lap"], "value": -0.0706, "places": 4}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "ci95", 0], "value": -0.134, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "ci95", 1], "value": -0.013, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "n_events"], "value": 21, "places": 0}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "p_perm_two_sided"], "value": 0.028, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-MEDIUM", "estimate_s_per_lap"], "value": -0.0055, "places": 4}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-MEDIUM", "ci95", 0], "value": -0.048, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-MEDIUM", "ci95", 1], "value": 0.035, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "MEDIUM-HARD", "estimate_s_per_lap"], "value": -0.0398, "places": 4}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "MEDIUM-HARD", "ci95", 0], "value": -0.085, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "MEDIUM-HARD", "ci95", 1], "value": 0.005, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "twfe_primary_holm", "holm_adjusted_p_two_sided", "SOFT-HARD"], "value": 0.085, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "twfe_primary_holm", "holm_adjusted_p_two_sided", "SOFT-MEDIUM"], "value": 0.807, "places": 3}
{"file": "paper/results/management.json", "path": ["referee_additions", "twfe_primary_holm", "holm_adjusted_p_two_sided", "MEDIUM-HARD"], "value": 0.182, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "twfe_gradient", "SOFT-HARD", "race_contrast_mean"], "value": 0.018, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "uniform_damping", "implied_additive_beta_per_step_if_race_slope_were_k_times_practice", "0.5"], "value": -0.019, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "uniform_damping", "observed_beta"], "value": -0.028, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_placebo_fp2_vs_fp1fp3_matched_pooled", "gap_SOFT_minus_MEDIUM", "estimate"], "value": -0.124, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_placebo_fp2_vs_fp1fp3_matched_pooled", "gap_SOFT_minus_MEDIUM", "ci95", 0], "value": -0.207, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "level_placebo_fp2_vs_fp1fp3_matched_pooled", "gap_SOFT_minus_MEDIUM", "ci95", 1], "value": -0.033, "places": 3}
{"file": "paper/results/management.json", "path": ["primary", "trend", "beta_per_softness_step"], "value": -0.028, "places": 4}
{"file": "paper/results/management.json", "path": ["primary", "trend", "ci95_cluster_bootstrap", 0], "value": -0.063, "places": 3}
{"file": "paper/results/management.json", "path": ["primary", "trend", "ci95_cluster_bootstrap", 1], "value": 0.014, "places": 3}
{"file": "paper/results/management.json", "path": ["primary", "trend", "p_perm_one_sided_less"], "value": 0.07, "places": 3}
{"file": "paper/results/management.json", "path": ["primary", "trend", "p_perm_two_sided"], "value": 0.137, "places": 3}
{"file": "paper/results/management.json", "path": ["post_review_additions", "power_additive_trend", "mde80_power_full_rule"], "value": 0.06, "places": 3}
{"file": "paper/results/management.json", "path": ["decision", "H1_additive_interaction_supported"], "equals": false}
```

### A9. Telemetry check of the nursing mechanism is null

**Claim.** Telemetry does not show the nursing signature. On the pre-specified primary proxy (coast time, predicted sign: race-minus-practice rises with softness), 2025-2026 over 26 events / 25045 laps, the verdict string reads: 'coast_s slope +0.039 s per softness step over 14 events (95% event-bootstrap CI -0.130 to +0.186; one-sided perm p 0.2981, two-sided 0.5953)'. Extension on 2023-2024 (34 events): 'coast_s slope -0.221 s per softness step over 16 events (95% event-bootstrap CI -0.443 to +0.007; one-sided perm p 0.9581, two-sided 0.0844)'. Both fail to support a positive slope; the extension is nominally in the OPPOSITE direction.

**Source.** `paper/results/telemetry.json -> primary.headline, extension.headline, verdict`

**Script.** scripts/paper/p6_telemetry.py (`tyredeg.paper.telemetry`)

**Caveat.** Data came from the OpenF1 public API: Formula 1's live-timing static host (FastF1's primary source) returned HTTP 401 for every session, so the source was switched AFTER the proxies and tests were frozen (proxies/thresholds unchanged). 97% of wanted laps joined. Coarse public proxies; absence of a signal here is not evidence of no management.

```claims
{"file": "paper/results/telemetry.json", "path": ["primary", "n_events"], "value": 26, "places": 0}
{"file": "paper/results/telemetry.json", "path": ["primary", "n_laps"], "value": 25045, "places": 0}
{"file": "paper/results/telemetry.json", "path": ["extension", "n_events"], "value": 34, "places": 0}
{"file": "paper/results/telemetry.json", "path": ["primary", "headline"], "equals": "coast_s slope +0.039 s per softness step over 14 events (95% event-bootstrap CI -0.130 to +0.186; one-sided perm p 0.2981, two-sided 0.5953)"}
{"file": "paper/results/telemetry.json", "path": ["extension", "headline"], "equals": "coast_s slope -0.221 s per softness step over 16 events (95% event-bootstrap CI -0.443 to +0.007; one-sided perm p 0.9581, two-sided 0.0844)"}
```

### A10. Practice-to-race forecast, out of season

**[READS REGENERATED FILES: transfer.json]**

**Claim.** Out of season (2022-2025, 70 event-compound cells, 44 weekends, HARD/MEDIUM/SOFT labels, practice design and the 2026-learned scalar factor 0.349 frozen), the calibrated practice forecast has MAE 0.0355 s/lap (95% weekend-bootstrap 0.0286 to 0.0426) against 0.112 for the naive Sunday=Friday forecast (0.092 to 0.132) and 0.0502 for predicting zero. It is indistinguishable from a constant prior (median race rate of the other weekends): constant MAE 0.0311, scalar-minus-constant paired difference +0.0045 s/lap [-0.0044, +0.0136], and from last season's race rate at the same circuit (difference -0.0039 [-0.0149, +0.0072]). In 2026 the tuned figure is 0.0561 against naive 0.0871 (n = 21 cells, tuned on the evaluation cells); nested selection inside each training fold gives 0.0595 (optimism of the shipped number 0.0035).

**Source.** `paper/results/transfer.json -> holdout_frozen, s2026_loeo, nested`

**Script.** scripts/paper/p2_transfer.py (`tyredeg.paper.transfer_eval`)

**Caveat.** The claim is 'much better than naive, no better than a constant'. The practice signal itself is weak (rank correlation ~0.3). The 2026 numbers were used to design the method; 2022-2025 were not. Out-of-season cells use label compounds. Two blended estimators sit in the same table and are NOT shipped: a 50% shrink toward the constant (MAE 0.0295) and a stack of the baselines (MAE 0.0317 after a leakage fix; before it, 0.0348). Stack minus scalar is +0.0039 [-0.0053, +0.0130], so it is not better than the shipped scalar either. Cite the shipped scalar and say the blends were tried.

```claims
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "n_cells"], "value": 70, "places": 0}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "n_weekends"], "value": 44, "places": 0}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "factor_frozen_from_2026"], "value": 0.349, "places": 3}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "scalar", "mae"], "value": 0.0355, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "scalar", "mae_ci_lo"], "value": 0.0286, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "scalar", "mae_ci_hi"], "value": 0.0426, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "naive", "mae"], "value": 0.112, "places": 3}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "naive", "mae_ci_lo"], "value": 0.092, "places": 3}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "naive", "mae_ci_hi"], "value": 0.132, "places": 3}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "zero", "mae"], "value": 0.0502, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "const", "mae"], "value": 0.0311, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "const", "mean_diff"], "value": 0.0045, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "const", "ci_lo"], "value": -0.0044, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "const", "ci_hi"], "value": 0.0136, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "history", "mean_diff"], "value": -0.0039, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "history", "ci_lo"], "value": -0.0149, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "history", "ci_hi"], "value": 0.0072, "places": 4}
{"file": "paper/results/transfer.json", "path": ["s2026_loeo", "scoreboard", "scalar", "mae"], "value": 0.0561, "places": 4}
{"file": "paper/results/transfer.json", "path": ["s2026_loeo", "scoreboard", "naive", "mae"], "value": 0.0871, "places": 4}
{"file": "paper/results/transfer.json", "path": ["nested", "mae_nested"], "value": 0.0595, "places": 4}
{"file": "paper/results/transfer.json", "path": ["nested", "optimism_of_shipped"], "value": 0.0035, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "stack", "mae"], "value": 0.0317, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "shrink50", "mae"], "value": 0.0295, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "stack", "mean_diff"], "value": 0.0039, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "stack", "ci_lo"], "value": -0.0053, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scalar_vs", "stack", "ci_hi"], "value": 0.013, "places": 4}
```

### A11. Pit loss under neutralisation

**[READS REGENERATED FILES: pit_loss.json]**

**Claim.** A stop costs a median 22.50 s of green-flag time (95% cluster bootstrap over races [22.06, 22.97]; 2426 stops, 103 races, 2022-2026). Under a virtual safety car it costs 0.930x that [0.81, 1.09] and under a full safety car 0.883x [0.67, 1.12] (348 stops, 40 races; VSC 235 stops); yellow-flag stops 1.159x. These use a wide sanity filter (loss between -20 and 60 s); the earlier narrow filter (0 to 45 s) gave 0.897 and 0.870. The safety-car ratio is filter-sensitive: with no filter it is 1.11 (95% 0.79 to 1.99) because 119 safety-car stops measure outside the filter, so read it as 'somewhat cheaper than green, not well determined'. The console default fraction is 0.883. The 2026 season alone differs (VSC 1.068, safety car 1.597).

**Source.** `paper/results/pit_loss.json -> by_category, by_season`

**Script.** scripts/10_pit_loss_by_status.py (`tyredeg.paper.pit_loss_eval`)

**Caveat.** Neutralised-stop loss is measured against the median of non-stopping cars on those laps, which a neutralisation itself sets (pit_loss.json -> limitations). Intervals reach 1.0. The optimiser has no concept of track position, which is why teams really pit under a safety car. The non-pitting reference is itself erratic under a safety car, which is why the filter matters.

```claims
{"file": "paper/results/pit_loss.json", "path": ["by_category", "green", "median_s"], "value": 22.5, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "green", "median_ci95", 0], "value": 22.06, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "green", "median_ci95", 1], "value": 22.97, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "green", "n_stops"], "value": 2426, "places": 0}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "green", "n_races"], "value": 103, "places": 0}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_to_green"], "value": 0.93, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_ci95", 0], "value": 0.81, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_ci95", 1], "value": 1.09, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_to_green"], "value": 0.883, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_ci95", 0], "value": 0.67, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_ci95", 1], "value": 1.12, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "n_stops"], "value": 348, "places": 0}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "n_races"], "value": 40, "places": 0}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "yellow", "ratio_to_green"], "value": 1.159, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_season", "2026", "vsc", "ratio"], "value": 1.068, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_season", "2026", "safety_car", "ratio"], "value": 1.597, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["console_default", "filter_sensitivity", "safety_car_ratio_no_filter"], "value": 1.11, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["console_default", "filter_sensitivity", "safety_car_stops_outside_sanity_filter"], "value": 119, "places": 0}
{"file": "paper/results/pit_loss.json", "path": ["console_default", "fraction"], "value": 0.883, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["truncation_effect", "safety_car", "unfiltered", "ratio_ci95", 0], "value": 0.79, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["truncation_effect", "safety_car", "unfiltered", "ratio_ci95", 1], "value": 1.99, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "n_stops"], "value": 235, "places": 0}
```

### A12. Strategy output vs baselines (a negative)

**[READS REGENERATED FILES: strategy.json]**

**Claim.** The strategy layer's stop-count call is only marginally better than simple baselines, out of time. Paired over 38 dry, un-red-flagged races (rates learned without the scored race; pit loss, lap-rule threshold and stint bounds fitted only on EARLIER races, so the first races, with fewer than 30 earlier stints of a label, are not planned), practice-scalar plan minus a race-length rule: stop-count agreement difference +0.158 [0.000, +0.316] (the interval touches zero); minus 'always one stop': +0.289 [+0.053, +0.526]. (Out of time is the headline; `leave_one_race_out_sensitivity` keeps the older form, whose race-length-rule contrast is +0.073 [-0.073, +0.220] over 41 races. The same comparison before the change, on 41 races with later races feeding the baselines, read +0.024 [-0.098, +0.146] and +0.122 [-0.073, +0.317]; the baselines got weaker out of time, the plan did not get better.) On the 2026 headline set the product's own (in-sample, not out of sample) call matched the field's median stop count at 8 of 13 races, always-one-stop at 5 of 13, and a race-length rule (threshold 58 laps, fitted on these events, so an upper bound) at 10 of 13.

**Source.** `paper/results/strategy.json -> paired_vs_baselines, shipped_claim_rechecked`

**Script.** scripts/paper/p5_strategy.py (`tyredeg.paper.strategy_eval`)

**Caveat.** Stop count is the coarsest property of a plan; the stint-length and compound comparisons are in strategy.json (w1_median_car etc.). The assumed 0.6 s fresh-tyre pace step is not estimated well from public data and the race estimator disagrees in sign across seasons (see C-section). The playbook's `actual_median_stops` (data/artifacts/playbook.json) disagrees with strategy.json's field_stop_counts at several races (British, Dutch, Monaco; the last two are red-flag races); the two use different definitions/sources and are NOT reconciled. Stint floors and ceilings are now rebuilt from genuine stints only (`data/artifacts/stint_pool.json`), and plans can be flagged `on_support_bound` or `at_stop_limit`; a plan sitting on a bound is a statement about the bound, not about tyres. The degradation-rate sources are still leave-one-race-out (`out_of_time.still_leave_one_race_out`).

```claims
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "n_races"], "value": 38, "places": 0}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "mean_a_minus_b"], "value": 0.158, "places": 3}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "ci95", 0], "value": 0.0, "places": 3}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "ci95", 1], "value": 0.316, "places": 3}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "always_1", "stop_match_mode", "mean_a_minus_b"], "value": 0.289, "places": 3}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "always_1", "stop_match_mode", "ci95", 0], "value": 0.053, "places": 3}
{"file": "paper/results/strategy.json", "path": ["paired_vs_baselines", "practice_scalar/plugin+support", "always_1", "stop_match_mode", "ci95", 1], "value": 0.526, "places": 3}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "product_in_sample_with_bounds", "matches"], "value": 8, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "product_in_sample_with_bounds", "of"], "value": 13, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "always_one_stop", "matches"], "value": 5, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "always_one_stop", "of"], "value": 13, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "race_length_rule_best_threshold_fitted_on_these_events", "threshold_laps"], "value": 58, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "race_length_rule_best_threshold_fitted_on_these_events", "matches"], "value": 10, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "race_length_rule_best_threshold_fitted_on_these_events", "of"], "value": 13, "places": 0}
{"file": "paper/results/strategy.json", "path": ["leave_one_race_out_sensitivity", "paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "mean_a_minus_b"], "value": 0.073, "places": 3}
{"file": "paper/results/strategy.json", "path": ["leave_one_race_out_sensitivity", "paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "ci95", 0], "value": -0.073, "places": 3}
{"file": "paper/results/strategy.json", "path": ["leave_one_race_out_sensitivity", "paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "ci95", 1], "value": 0.22, "places": 3}
{"file": "paper/results/strategy.json", "path": ["leave_one_race_out_sensitivity", "paired_vs_baselines", "practice_scalar/plugin+support", "lap_rule", "stop_match_mode", "n_races"], "value": 41, "places": 0}
```

### A13. Reproduction of, and head-to-head with, Cappello & Hoegh (arXiv v1 and journal Table 4)

**Claim.** We ran the published model. Table 3 reproduces: Hard 0.0546, Medium 0.0552 s/lap (published 0.054 and 0.060), P(Medium > Hard) 0.528. Their fuel ramp is indexed to retained laps and is misplaced by up to 4.74 kg at Austria; correcting it moves P(Medium > Hard) to 0.539 and the skew-t CRPS by -0.0013: no effect, the compounds still overlap. We do NOT beat their forecaster. On their exact test laps, 15 races: we win 2; mean CRPS 0.326 (ours) against 0.220 (theirs), paired difference +0.106 [+0.067, +0.146], sign test p = 0.0074. At Austria, our CRPS on their laps is 0.229 against 0.2022 (paired lap-level difference +0.024 [-0.004, +0.049]). Scoring their skew-t draws as if normal changes the 12-race mean CRPS by -0.0066 (matched better in 10 of 12), but only -0.0014 at Austria. **Against the published journal Table 4 (Cappello & Hoegh 2026, p. 10), same 15 races:** their SSM column is the repo CSV to three decimals, so the SSM result is unchanged (we win 2; 0.326 vs 0.220). Against their journal AR(1) baseline we win 5 of 15 races: mean CRPS 0.326 against 0.281, paired difference +0.045 [-0.007, +0.095], sign test p = 0.30, so we cannot be told apart from their AR(1) on these races, and we lose to their SSM in 13 of 15. RMSPE tells the same story (their SSM 0.375, their AR(1) 0.456, ours 0.524). Their journal SSM beats its own AR(1) in 14 of these 15 races on CRPS.

**Source.** `paper/results/benchmark.json -> table3, fuel_defect, head_to_head, skewt_cv_season_matched_vs_mismatched, journal_table4`; `benchmark/published_table4.json`

**Script.** scripts/paper/p4_benchmark.py (`tyredeg.paper.benchmark_repro`, `tyredeg.paper.published_benchmark`; transcription in `benchmark/published_table4.json`); scripts/08_benchmark.py (artifact)

**Caveat.** Their model is a one-step single-driver forecaster; ours is a field-pooled degradation estimator, so the comparison is about the forecasting task they chose, not the estimation task we chose. 11 of 12 non-Austria races were re-run at reduced iterations (higher R-hat); the reproduction correlates 0.9997 with their per-race CSV. The journal's Table 4 has 19 races; our cached 2025 feed lacks China, United States, Sao Paulo and Qatar, so the comparison is over the 15 both cover (Abu Dhabi has 2 stints in our reconstruction against their 3, which affects its RMSPE). Their per-race fits are single-driver and independent. Their printed 'Mean Across Races' row equals the 19-race mean including Singapore although the caption says it excludes it, so we use only the per-race rows.

```claims
{"file": "paper/results/benchmark.json", "path": ["table3", "their_ramp", "v_hard", "mean"], "value": 0.0546, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["table3", "their_ramp", "v_medium", "mean"], "value": 0.0552, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["table3", "their_ramp", "p_medium_gt_hard"], "value": 0.528, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["fuel_defect", "austria", "max_abs_kg"], "value": 4.74, "places": 2}
{"file": "paper/results/benchmark.json", "path": ["table3", "corrected_fuel", "p_medium_gt_hard"], "value": 0.539, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["fuel_defect", "skewt_cv_effect", "delta_crps"], "value": -0.0013, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "ours_wins"], "value": 2, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "ours_mean"], "value": 0.326, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "theirs_mean"], "value": 0.22, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "mean_diff_ours_minus_theirs"], "value": 0.106, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "mean_diff_ci95", 0], "value": 0.067, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "mean_diff_ci95", 1], "value": 0.146, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "sign_test_p_two_sided"], "value": 0.0074, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "austria", "ours_their_folds"], "value": 0.229, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "austria", "theirs_repo_csv"], "value": 0.2022, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "austria_paired_vs_reproduced_theirs", "mean_paired_diff"], "value": 0.024, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "austria_paired_vs_reproduced_theirs", "ci95", 0], "value": -0.004, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "austria_paired_vs_reproduced_theirs", "ci95", 1], "value": 0.049, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["skewt_cv_season_matched_vs_mismatched", "mean_diff_matched_minus_normal"], "value": -0.0066, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["skewt_cv_season_matched_vs_mismatched", "races_matched_better"], "value": 10, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["skewt_cv_season_matched_vs_mismatched", "n_races"], "value": 12, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["skewt_cv_austria", "iter10000_theirs", "matched_minus_mismatched_crps"], "value": -0.0014, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "csv_vs_journal", "all_equal_at_3dp"], "equals": true}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "coverage", "ours_scored"], "value": 15, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ssm", "ours_wins"], "value": 2, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ssm", "theirs_mean"], "value": 0.2203, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "ours_wins"], "value": 5, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "theirs_mean"], "value": 0.2809, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "mean_diff_ours_minus_theirs"], "value": 0.045, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "mean_diff_ci95", 0], "value": -0.007, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "mean_diff_ci95", 1], "value": 0.095, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "crps_vs_ar1", "sign_test_p_two_sided"], "value": 0.302, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "rmspe_vs_ssm", "theirs_mean"], "value": 0.375, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "rmspe_vs_ssm", "ours_mean"], "value": 0.524, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "rmspe_vs_ar1", "theirs_mean"], "value": 0.456, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "common_races", "journal_ssm_beats_ar1_on_these_races", "crps"], "value": 14, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "table4_internal_checks", "ssm_beats_ar1", "crps"], "value": 16, "places": 0}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "table4_internal_checks", "crps_ar1", "printed"], "value": 0.3, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["journal_table4", "table4_internal_checks", "crps_ar1", "mean_all_19"], "value": 0.2996, "places": 4}
{"file": "benchmark/published_table4.json", "path": ["races", "Austrian Grand Prix", "crps_ar1"], "value": 0.289, "places": 3}
{"file": "benchmark/published_table4.json", "path": ["other_published_tables_austria", "table2_rmspe_total", "ar1"], "value": 1.478, "places": 3}
```

### A14. Practice design choices barely matter

**Claim.** Out of season (2022-2025), the keep-runs practice frame (runs bridged across short gaps) differs from the shipped frame by -0.0048 s/lap (variant minus shipped, so negative = variant better) [-0.0076, -0.0021] on 69 common cells. In 2026 the warm-up trim changes MAE by -0.0018 [-0.0056, +0.0006] (sign-flip p 0.50), i.e. nothing detectable. Shipped practice frame: 2815 laps against 2339 for the retired design.

**Source.** `paper/results/design_sweep.json -> 2022-2025.keep-runs, 2026.warmup1, parity`

**Script.** scripts/13_design_sweep.py (`tyredeg.paper.transfer_eval`)

**Caveat.** Treat as a robustness statement (out of season, variants change MAE by at most 0.007 s/lap; in 2026 designs keeping every short run score 0.011-0.018 s/lap worse than the shipped one) and not as a claim that any design is best. `diff_vs_shipped` is variant minus shipped (negative = variant better). If this reads oddly, check the sign convention in transfer_eval before quoting.

```claims
{"file": "paper/results/design_sweep.json", "path": ["2022-2025", "keep-runs", "diff_vs_shipped"], "value": -0.0048, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2022-2025", "keep-runs", "diff_ci", 0], "value": -0.0076, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2022-2025", "keep-runs", "diff_ci", 1], "value": -0.0021, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2022-2025", "keep-runs", "n_common_with_shipped"], "value": 69, "places": 0}
{"file": "paper/results/design_sweep.json", "path": ["2026", "warmup1", "diff_vs_shipped"], "value": -0.0018, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "warmup1", "diff_ci", 0], "value": -0.0056, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "warmup1", "diff_ci", 1], "value": 0.0006, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "warmup1", "signflip_p"], "value": 0.5, "places": 2}
{"file": "paper/results/design_sweep.json", "path": ["parity", "shipped_laps"], "value": 2815, "places": 0}
{"file": "paper/results/design_sweep.json", "path": ["parity", "legacy_laps"], "value": 2339, "places": 0}
```

### A15. Dataset size

**Claim.** Dataset: 23856 clean 2026 laps over 14 events (13 with race laps), of which 11039 race laps in 820 long runs by 23 drivers enter the race fit. 2022-2025 hold 13,637 / 13,536 / 15,942 / 15,279 race-fit laps (13637 etc. in model.json -> season_robustness). The 142,737 figure in earlier README versions (all five seasons, every clean lap) is a parquet row count and is not in any JSON.

**Source.** `paper/results/model.json -> dataset, season_robustness; data/artifacts/meta.json`

**Script.** scripts/01_cache_sessions.py; scripts/03_fit_model.py

**Caveat.** 2026 is a partial season (13 races with laps + Spain practice only).

```claims
{"file": "paper/results/model.json", "path": ["dataset", "n_laps_total"], "value": 23856, "places": 0}
{"file": "paper/results/model.json", "path": ["dataset", "n_events_in_file"], "value": 14, "places": 0}
{"file": "paper/results/model.json", "path": ["dataset", "race_fit", "n_events"], "value": 13, "places": 0}
{"file": "paper/results/model.json", "path": ["dataset", "race_fit", "n_laps"], "value": 11039, "places": 0}
{"file": "paper/results/model.json", "path": ["dataset", "race_fit", "n_runs"], "value": 820, "places": 0}
{"file": "paper/results/model.json", "path": ["dataset", "race_fit", "n_drivers"], "value": 23, "places": 0}
{"file": "paper/results/model.json", "path": ["season_robustness", "2022", "n_laps"], "value": 13637, "places": 0}
```


## B. Claims withdrawn

Each entry: the old value, what the re-analysis found, and where. The repository's culture is to say so out loud; the paper should not quietly drop these, and a referee who finds an old deck or README figure should find the retraction here.

### B1. 'Drivers nurse the soft tyre' (rho -0.288, p = 0.0035, 101 cells)

OLD: drivers nurse softer tyres harder in races; Spearman rho = -0.288, p = 0.0035 (two-sided), 101 cells, 5 seasons. RE-ANALYSIS: the legacy number reproduces mechanically (rho -0.288, p 0.0035, 101 cells) but rested on a race-slope estimator that leaked each compound's pace offset into its slope and that differs from the practice estimator. On the SAME estimator in both contexts the ratio test reads rho +0.033, p two-sided 0.774 (data/artifacts/management.json, 81 cells with a usable ratio); the race-slope estimators agree only weakly (Spearman 0.49). What survives is A7; the mechanism is A8/A9.

**Source.** `paper/results/management.json -> legacy_ratio; data/artifacts/management.json`

```claims
{"file": "paper/results/management.json", "path": ["legacy_ratio", "as_shipped_race_slope_field_demeaned", "rho"], "value": -0.288, "places": 3}
{"file": "paper/results/management.json", "path": ["legacy_ratio", "as_shipped_race_slope_field_demeaned", "p_two_sided"], "value": 0.0035, "places": 4}
{"file": "paper/results/management.json", "path": ["legacy_ratio", "as_shipped_race_slope_field_demeaned", "n_cells"], "value": 101, "places": 0}
{"file": "data/artifacts/management.json", "path": ["rho"], "value": 0.033, "places": 3}
{"file": "data/artifacts/management.json", "path": ["p_two_sided"], "value": 0.774, "places": 3}
{"file": "data/artifacts/management.json", "path": ["stability", 4, "n_cells"], "value": 81, "places": 0}
{"file": "paper/results/management.json", "path": ["legacy_ratio", "race_slope_estimator_agreement", "spearman_rho"], "value": 0.49, "places": 2}
```

### B2. 'Practice -> race 35.6% better than naive' (0.056 vs 0.087)

**[READS REGENERATED FILES: transfer.json]**

OLD: practice-to-race MAE 0.056 s/lap against naive 0.087, '35.6% error reduction', 21 cells, 2026. RE-ANALYSIS: the design and session weights were chosen on those cells; nested selection gives 0.0595; the naive baseline is the weakest available; out of season the forecast (0.0355) beats naive (0.112) but not a constant (0.0311). The 'compound-specific factor' scored worse than the scalar (0.0634 against 0.0355). See A10.

**Source.** `paper/results/transfer.json -> nested, holdout_frozen`

```claims
{"file": "paper/results/transfer.json", "path": ["nested", "mae_nested"], "value": 0.0595, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "scalar", "mae"], "value": 0.0355, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "naive", "mae"], "value": 0.112, "places": 3}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "const", "mae"], "value": 0.0311, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "compound", "mae"], "value": 0.0634, "places": 4}
{"file": "paper/results/transfer.json", "path": ["holdout_frozen", "scoreboard", "scalar", "mae"], "value": 0.0355, "places": 4}
```

### B3. 'FP2 predicts the race best, so it is weighted heaviest'

OLD: FP2 is the best predictor, so it is weighted heaviest (FP1 correlation 0.05, then 0.78, then 0.29 across three data states). RE-ANALYSIS: over 2022-2025 (25 shared cells) the correlation difference FP1 minus FP2 is -0.113 [-0.395, +0.186]; MAE difference +0.0019 [-0.009, +0.013]. In 2026 the shipped weighting wins by at most 0.0041 s/lap, on the cells it was chosen on. The weights are a prior, not a finding.

**Source.** `paper/results/session_skill.json -> 2022-2025.contrasts, 2026.weights_2026`

```claims
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "n_common"], "value": 25, "places": 0}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "rho_diff"], "value": -0.113, "places": 3}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "rho_diff_ci", 0], "value": -0.395, "places": 3}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "rho_diff_ci", 1], "value": 0.186, "places": 3}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "mae_diff_a_minus_b"], "value": 0.0019, "places": 4}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "mae_diff_ci", 0], "value": -0.009, "places": 3}
{"file": "paper/results/session_skill.json", "path": ["2022-2025", "contrasts", "FP1-FP2", "mae_diff_ci", 1], "value": 0.013, "places": 3}
{"file": "paper/results/session_skill.json", "path": ["weights_2026", "sprint only", "diff_vs_shipped"], "value": 0.0041, "places": 4}
```

### B4. 'Warm-up fix: 0.0819 -> 0.0491 s/lap'

OLD: dropping the first lap of each run and bridging gaps cut practice-to-race error from 0.0819 to 0.0491. RE-ANALYSIS: not reproducible; the old script's 'current' variant was a retired pipeline (2339 laps against the shipped 2815). On common 2026 cells legacy minus shipped is +0.0126 [-0.0017, +0.0263] (sign-flip p 0.146).

**Source.** `paper/results/design_sweep.json -> parity, 2026.legacy`

```claims
{"file": "paper/results/design_sweep.json", "path": ["parity", "legacy_laps"], "value": 2339, "places": 0}
{"file": "paper/results/design_sweep.json", "path": ["parity", "shipped_laps"], "value": 2815, "places": 0}
{"file": "paper/results/design_sweep.json", "path": ["2026", "legacy", "diff_vs_shipped"], "value": 0.0126, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "legacy", "diff_ci", 0], "value": -0.0017, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "legacy", "diff_ci", 1], "value": 0.0263, "places": 4}
{"file": "paper/results/design_sweep.json", "path": ["2026", "legacy", "signflip_p"], "value": 0.146, "places": 3}
```

### B5. 'Circuit spread 0.146 beats compound spread 0.112; LR 385, p = 4e-86'

OLD: circuit spread 0.146 s/lap against compound spread 0.112; LR = 385, p = 4e-86. RE-ANALYSIS: the 0.146 was a range of shrunken predictions (now 0.191), not an effect size; the converged fit gives circuit SD 0.0498 [0.035, 0.081], LR 220.8, bootstrap p 0.002 (0.0437 [0.031, 0.073], LR 172.8 before the exact demeaning of 2026-11-01). The earlier fit had not converged (unscaled L-BFGS SD 0.099, converged False; rescaled 0.036). See A1.

**Source.** `paper/results/model.json -> circuit_effect, old_fit_reproduction`

```claims
{"file": "paper/results/model.json", "path": ["circuit_effect", "shrunken_predictions_not_an_effect_size", "range_s_per_lap"], "value": 0.191, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "estimate"], "value": 0.0498, "places": 4}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "profile_likelihood_95", 0], "value": 0.035, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_effect", "circuit_slope_sd_s_per_lap", "profile_likelihood_95", 1], "value": 0.081, "places": 3}
{"file": "paper/results/model.json", "path": ["circuit_effect", "lr_test", "reml_lr_statistic"], "value": 220.8, "places": 1}
{"file": "paper/results/model.json", "path": ["circuit_effect", "lr_test", "p_parametric_bootstrap"], "value": 0.002, "places": 3}
{"file": "paper/results/model.json", "path": ["old_fit_reproduction", "unscaled_lbfgs", "circuit_slope_sd_s_per_lap"], "value": 0.099, "places": 3}
{"file": "paper/results/model.json", "path": ["old_fit_reproduction", "unscaled_lbfgs", "converged"], "equals": false}
{"file": "paper/results/model.json", "path": ["old_fit_reproduction", "rescaled_lbfgs", "circuit_slope_sd_s_per_lap"], "value": 0.036, "places": 3}
```

### B6. 'Five distinct rates, in strict order, no crossings' (and the old C1-C5 values)

OLD: C1-C5 rates 0.084 / 0.050 / 0.044 / 0.029 / 0.011 'in strict order, no crossings, five distinct rates'. RE-ANALYSIS: converged rates 0.082 / 0.053 / 0.050 / 0.042 / 0.022 (approximate column construction); with exact within-(event, lap) demeaning (2026-11-01) 0.041 / 0.037 / 0.054 / 0.063 / 0.039; the 95% intervals overlap pairwise (`separated_pairs_nonoverlap` has 0 entries); `monotone_softer_degrades_faster` = False.

**Source.** `paper/results/model.json -> shipped_fit`

```claims
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "mean"], "value": 0.041, "places": 3}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C2", "mean"], "value": 0.037, "places": 3}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C3", "mean"], "value": 0.054, "places": 3}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C4", "mean"], "value": 0.063, "places": 3}
{"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C5", "mean"], "value": 0.039, "places": 3}
{"file": "paper/results/model.json", "path": ["shipped_fit", "monotone_softer_degrades_faster"], "equals": false}
```

### B7. '11 of 25 pairs separated' and '4.4x tighter intervals'

OLD: '11 of 25 adjacent compound pairs separate race by race', '4.4x tighter than the published model, up to 9.1x'. RE-ANALYSIS: lap-level, 11 of 25 pairs separate but only 2 in the physical direction (9 the wrong way); with exact demeaning (2026-11-01) 7 of 25 separate, 2 in the physical direction (5 the wrong way). With run-level terms: 4 of 25, 1 in the physical direction; now 3 of 25, 1. The tightening ratio is 4.66x median (max 10.3x) at lap level but 2.47x (max 5.43x) once runs are the unit (4.41x, max 9.08x, and 2.09x, max 6.15x, before 2026-11-01); and the comparison is not like for like (one driver's three stints against a field's).

**Source.** `paper/results/model.json -> per_race_separation`

```claims
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "pairs_separated_non_overlap_any_direction"], "value": 7.0, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "n_adjacent_pairs"], "value": 25, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "pairs_separated_non_overlap_correct_direction"], "value": 2, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "pairs_separated_non_overlap_wrong_direction"], "value": 5.0, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_run_level", "pairs_separated_non_overlap_any_direction"], "value": 3.0, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_run_level", "pairs_separated_non_overlap_correct_direction"], "value": 1, "places": 0}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "median_tightening_ratio"], "value": 4.66, "places": 2}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_lap_level", "max_tightening_ratio"], "value": 10.3, "places": 2}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_run_level", "median_tightening_ratio"], "value": 2.47, "places": 2}
{"file": "paper/results/model.json", "path": ["per_race_separation", "protocols", "min_runs_1_run_level", "max_tightening_ratio"], "value": 5.43, "places": 2}
```

### B8. 'Power: 512 needed, we have 820'

OLD: separating two compounds 0.006 s/lap apart needs 512 driver-stints; we have 820. RE-ANALYSIS: the 512 was a grid point from a Welch-t calculation that ignores circuit and run variance (interpolated crossing 310.8). The mixed-model analysis needs 1702 stints balanced (1212 before the exact demeaning of 2026-11-01), and a split design never reaches 80% (ceiling 0.055; 0.057 before). See A6.

**Source.** `paper/results/model.json -> power`

```claims
{"file": "paper/results/model.json", "path": ["power", "original_welch_calculation", "interpolated_crossing"], "value": 310.8, "places": 1}
{"file": "paper/results/model.json", "path": ["power", "curves", "balanced_effect_0.006", "crossing_80pct_wald_z", "crossing"], "value": 1702.0, "places": 0}
{"file": "paper/results/model.json", "path": ["power", "ceiling_split_design_power", "0.006"], "value": 0.055, "places": 3}
```

### B9. '80.6% coverage at nominal 80%' as the calibration headline

OLD: '80% intervals cover 80.6% empirically' presented as honest uncertainty. RE-ANALYSIS: that is held-out LAP coverage (0.8059), dominated by lap scatter. For the rates themselves, leave-one-circuit-out coverage is 0.667 (n = 36; 0.778 before the exact demeaning of 2026-11-01). See A5.

**Source.** `paper/results/model.json -> lap_level_calibration, rate_calibration`

```claims
{"file": "paper/results/model.json", "path": ["lap_level_calibration", "coverage_80"], "value": 0.8059, "places": 4}
{"file": "paper/results/model.json", "path": ["rate_calibration", "leave_one_event_out", "variants", "full_predictive", "coverage", "0.80", "coverage"], "value": 0.667, "places": 3}
```

### B10. Naive vs deconfounded table (label-keyed)

OLD table (label-keyed): HARD naive +0.075 / deconfounded +0.072; MEDIUM -0.122 / +0.029; SOFT -0.082 / +0.003 ('two of four compounds the naive fit says the tyre gets faster'). RE-ANALYSIS: that compared a label-pooled naive fit with C-number deconfounded fits, which is not like for like. The three-way table on physical compounds (A3) has naive -0.035 / -0.037 / -0.070 negative at C1/C3/C5, but a fuel-corrected practitioner fit lands within 0.001-0.030 s/lap of ours (0.002-0.049 before the exact demeaning of 2026-11-01).

**Source.** `paper/results/model.json -> ablation`

```claims
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C1", "naive_run_clustered", "mean"], "value": -0.035, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C3", "naive_run_clustered", "mean"], "value": -0.037, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "rows", "C5", "naive_run_clustered", "mean"], "value": -0.07, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C3"], "value": 0.006, "places": 3}
{"file": "paper/results/model.json", "path": ["ablation", "abs_gap_deconfounded_vs_practitioner_s_per_lap", "C1"], "value": 0.007, "places": 3}
```

### B11. 'Strategy call matches teams at 10 of 11 races'

**[READS REGENERATED FILES: strategy.json]**

OLD: 'the stop count agrees with what real teams ran at 10 of 11 races'; 'refuses 2 races'. RE-ANALYSIS: on the 2026 races scored (13), product in-sample 8 (7 without the stint bounds), always one stop 5, race-length rule (fitted on these events) 10; the product's rates were fitted on the scored race. Out of time the plan beats the baselines only marginally on stop-count agreement (the interval touches zero against the race-length rule; see A12).

**Source.** `paper/results/strategy.json -> shipped_claim_rechecked`

```claims
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "events_scored"], "value": 13, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "product_in_sample_with_bounds", "matches"], "value": 8, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "always_one_stop", "matches"], "value": 5, "places": 0}
{"file": "paper/results/strategy.json", "path": ["shipped_claim_rechecked", "race_length_rule_best_threshold_fitted_on_these_events", "matches"], "value": 10, "places": 0}
```

### B12. 'VSC stop costs 0.84x'; 'safety car not measurable'

**[READS REGENERATED FILES: pit_loss.json]**

OLD: a VSC stop costs 0.84x a green one (48 stops, 2026 only); a full safety car is 'not measurable' (23 stops, two races; an earlier version used 0.45). RE-ANALYSIS (2022-2026): VSC 0.930 [0.81, 1.09], safety car 0.883 [0.67, 1.12] on 348 stops, under a wide outlier filter (the earlier 0.897 and 0.870 used a narrow one); the safety-car ratio is 1.11 with no filter. See A11.

**Source.** `paper/results/pit_loss.json -> by_category`

```claims
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_to_green"], "value": 0.93, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_ci95", 0], "value": 0.81, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "vsc", "ratio_ci95", 1], "value": 1.09, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_to_green"], "value": 0.883, "places": 3}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_ci95", 0], "value": 0.67, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "ratio_ci95", 1], "value": 1.12, "places": 2}
{"file": "paper/results/pit_loss.json", "path": ["by_category", "safety_car", "n_stops"], "value": 348, "places": 0}
```

### B13. Benchmark: old season means, Singapore 3.83, '0.522 -> 0.515', skew-t premise

OLD: season mean CRPS 0.5471 (ours, our feed) against 0.220, with Singapore scoring 3.83 'because of a 32-second lap'; fuel-ramp bug '0.522 -> 0.515'. RE-ANALYSIS: the 3.83 was our scope error (their pipeline drops Singapore laps after 58; our feed kept them). On their retained test laps the season mean is 0.326 against 0.220 (2 wins of 15 either way). The fuel correction moves P(Medium > Hard) 0.528 -> 0.539 (no effect). The premise that their skew-t draws were scored as a normal was only half true: the mismatch costs -0.0066 mean CRPS over 12 races, 0.0014 at Austria.

**Source.** `paper/results/benchmark.json -> head_to_head, table3, skewt_cv_season_matched_vs_mismatched`

```claims
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "A_own_folds_vs_published", "ours_mean"], "value": 0.5471, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "A_own_folds_vs_published", "theirs_mean"], "value": 0.22, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "ours_mean"], "value": 0.326, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["head_to_head", "season_vs_published", "B_their_folds_vs_published", "theirs_mean"], "value": 0.22, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["table3", "their_ramp", "p_medium_gt_hard"], "value": 0.528, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["table3", "corrected_fuel", "p_medium_gt_hard"], "value": 0.539, "places": 3}
{"file": "paper/results/benchmark.json", "path": ["skewt_cv_season_matched_vs_mismatched", "mean_diff_matched_minus_normal"], "value": -0.0066, "places": 4}
```

### B14. Fuel coefficient 0.043 ('30% high') and 0.0501

OLD: our hierarchical state-space model reads a fuel coefficient of 0.043 s/kg ('about 30% high'), and 0.0501 for the 70-kg-start bug. RE-ANALYSIS: neither reproduces. Whole-race gamma at the correct 110 kg start is 0.0201 (Austria 2025) and 0.0356 (Bahrain 2024); the 70 kg bug gives 0.0557, which rescales to 0.0355.

**Source.** `paper/results/benchmark.json -> gamma.ours_hierarchical`

```claims
{"file": "paper/results/benchmark.json", "path": ["gamma", "ours_hierarchical", "Austrian Grand Prix 2025 | shipped_prior_sd0.02", "gamma_mean"], "value": 0.0201, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["gamma", "ours_hierarchical", "Bahrain Grand Prix 2024, 110 kg start (correct) | shipped_prior_sd0.02", "gamma_mean"], "value": 0.0356, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["gamma", "ours_hierarchical", "Bahrain Grand Prix 2024, 70 kg start (the bug) | shipped_prior_sd0.02", "gamma_mean"], "value": 0.0557, "places": 4}
{"file": "paper/results/benchmark.json", "path": ["gamma", "ours_hierarchical", "Bahrain Grand Prix 2024, 70 kg start (the bug) | shipped_prior_sd0.02", "mean_if_rescaled_to_110kg"], "value": 0.0355, "places": 4}
```

### B15. Dataset size (19,708 laps) and test count (293)

OLD: 19,708 clean laps. NOW: 23856 clean 2026 laps (14 events in the file); 142,737 across 2022-2026 (parquet row count; not in a JSON). The count of tests in the repository is 497 (was 293); that is not in a JSON either.

**Source.** `paper/results/model.json -> dataset`

```claims
{"file": "paper/results/model.json", "path": ["dataset", "n_laps_total"], "value": 23856, "places": 0}
```


## C. Open limitations a referee will raise, and how the data answers them

| Objection | What the data says | Where |
| --- | --- | --- |
| "Circuit and compound are confounded: Pirelli nominates by track." | True, and it caps what a pooled fit can say. Within-event contrasts (compounds co-running at one race) are free of the circuit confound; power for compounds run at disjoint circuits is capped near 0.06 (A6). Per-race separation counts are within-event (B7). | `model.json` -> `power`, `per_race_separation` |
| "One season, 13 circuits for the circuit SD." | Correct for the 2026-only fit, whose SD (0.050) mixes persistent and race-specific variation because each circuit is raced once; single seasons give 0.023 to 0.046 (`season_robustness`). The pooled 2022-2026 nested model (23 circuits, 20 with repeat visits) separates them: persistent 0.028 [0.020, 0.041], race-specific 0.024 [0.021, 0.029] (A1). | `model.json` -> `circuit_vs_event`, `season_robustness` |
| "The softer-degrades-slower reversal is an artefact." | It was, in the earlier fit: the approximate column construction leaked slopes and offsets between compounds mixed on a lap (2026 C1 0.082 -> 0.041; B6). With exact within-(event, lap) demeaning the 2026 within-event trend is about zero (-0.0010 [-0.0047, +0.0028]) and pooled rates rise with softness from C2 to C5; C5 below C4 in 2026 is unexplained. Within-event seasonal trends are positive in 2022, 2023 and 2025, about zero in 2024 and 2026 (A2). Nursing is not established (A8, A9). | `model.json` -> `compound_within_event`, `season_robustness` |
| "Race and practice slopes differ in estimator, set state and fuel." | The headline gap (A7) uses one estimator and used sets in both contexts (start-age matching and the intersection design are reported as robustness rows); a practice-vs-practice placebo shows what the design can fake. The cross-sectional race estimator is not identified and is not cited. | `management.json` -> `referee_additions`, `post_review_additions`, `analysis_tiers` |
| "Forking paths: the management analysis was changed after review." | Disclosed in `post_review_additions.provenance` and `referee_additions.provenance`: the matching specs were added after an independent review, fixed before being run, and are not in the original pre-registration. The 74-spec multiverse is published; 70 of 74 betas are negative, 37 have CIs excluding zero. | `management.json` -> `multiverse_summary` |
| "The strategy layer assumes a 0.6 s fresh-tyre pace step." | It does, and the data do not pin it down. The 2026 race-within-driver estimate has the softer tyre faster by only about -0.17 s/step (the wrong sign) and the practice estimate has an interval from -0.55 to 1.71; 2022-2025 race gives 0.18. The estimates disagree in sign across seasons, so the assumed value cannot be validated from public data. See the checks below. | `strategy.json` -> `pace_step_estimation`, `pace_step_sensitivity` |
| "Neutralised-stop loss depends on your outlier filter." | For safety cars it does: 0.87 (narrow filter), 0.883 (wide), 1.11 (none, interval 0.79 to 1.99). We report the wide one with that caveat. VSC is stable (0.90, 0.93, 0.94). | `pit_loss.json` -> `console_default.filter_sensitivity`, `truncation_effect` |
| "The strategy evaluation leaks future races." | It did, in the baselines' pit loss, lap-rule threshold and stint bounds. The headline is now out of time: only earlier races feed those. The degradation-rate sources are still leave-one-race-out (`out_of_time.still_leave_one_race_out`). The older form is kept as `leave_one_race_out_sensitivity`. | `strategy.json` -> `out_of_time`, `leave_one_race_out_sensitivity` |
| "The optimiser ignores track position, traffic and the undercut." | Yes, and the console says so. A stop-count call is the coarsest thing a plan says (A12). | `pit_loss.json` -> `limitations` |
| "Benchmark comparison is unfair to them (or to you)." | It is a different task. We lose their task (A13). Their reproduction has correlation 0.9997 with their per-race CSV over 12 races, so the loss is not a replication artefact. | `benchmark.json` -> `head_to_head` |
| "Residual autocorrelation within runs." | 0.22 in the shipped model (down from 0.36 lap-level), handled with run intercepts and slopes; cluster-robust OLS comparators are reported in A4. | `model.json` -> `inference_comparison` |
| "Telemetry came from a different source than the one you planned." | Yes: F1 live-timing returned HTTP 401, so OpenF1 was used, switched after proxies and tests were frozen. The cross-check against the one FastF1-cached event is in `cross_check_sources`. | `telemetry.json` -> `meta.data_source`, `access.probe` |
| "Practice-to-race forecast: you tuned on 2026." | The out-of-season frozen holdout is the evidence (A10); the 2026 optimism of the tuned number is 0.0035 s/lap. | `transfer.json` -> `nested`, `holdout_frozen` |

```claims
{"file": "paper/results/strategy.json", "path": ["pace_step_estimation", "2026", "race_within_driver_lap", "softer_faster_by_s_per_step"], "value": -0.174, "places": 3}
{"file": "paper/results/strategy.json", "path": ["pace_step_estimation", "2022-25", "race_within_driver_lap", "softer_faster_by_s_per_step"], "value": 0.178, "places": 3}
{"file": "paper/results/strategy.json", "path": ["pace_step_estimation", "2026", "practice", "ci95_softer_faster_by", 0], "value": -0.55, "places": 2}
{"file": "paper/results/strategy.json", "path": ["pace_step_estimation", "2026", "practice", "ci95_softer_faster_by", 1], "value": 1.71, "places": 2}
{"file": "paper/results/model.json", "path": ["season_robustness", "2025", "circuit_slope_sd_s_per_lap"], "value": 0.0227, "places": 4}
{"file": "paper/results/model.json", "path": ["season_robustness", "2026", "circuit_slope_sd_s_per_lap"], "value": 0.046, "places": 4}
{"file": "paper/results/management.json", "path": ["multiverse_summary", "n_specs"], "value": 74, "places": 0}
{"file": "paper/results/management.json", "path": ["multiverse_summary", "n_beta_negative"], "value": 70, "places": 0}
{"file": "paper/results/management.json", "path": ["multiverse_summary", "n_ci_excludes_zero_negative"], "value": 37, "places": 0}
{"file": "paper/results/telemetry.json", "path": ["access", "probe", "hosts", "livetiming.formula1.com (FastF1 primary)", "http_status"], "value": 401, "places": 0}
{"file": "paper/results/telemetry.json", "path": ["access", "probe", "hosts", "api.openf1.org", "http_status"], "value": 200, "places": 0}
{"file": "paper/results/model.json", "path": ["inference_comparison", "residual_lag1_autocorrelation_within_runs", "D_run_intercept_and_slope_SHIPPED"], "value": 0.22, "places": 2}
```

Literature context for the objections above (all opened; see `paper/refs/NOTES.md`): compound x race-phase confounding of
per-race base pace is documented by Santillana (arXiv:2607.06495, p. 7; unreviewed); a published FastF1 multi-season attempt
that gets negative TyreLife slopes for SOFT and MEDIUM from fuel/age collinearity is Joshi (arXiv:2608.03192, Table 1, p. 6;
unreviewed); Cappello & Hoegh (J. Sports Analytics 2026, read in full) evaluate forecasts over 19 races but fit each race on its own, "rather than pooled across events" (p. 7), list multiple drivers and hierarchical track effects as future work (p. 12), use practice only as a hand-set prior for the post-pit pace level (pp. 6-7), and hypothesise tyre management as the reason compounds look alike (p. 10) without measuring it. Race-level
inference because laps within a race are not independent: Shakila & Baalaji (Research Square preprint, 2026; not peer
reviewed). None of these pre-empts a claim in section A as far as the notes could establish (NOTES section A4 is the claim-by-claim table for the journal version: multi-race pooled estimation, per-circuit random slopes, race-vs-practice, practice-to-race forecast and power are all not pre-empted; state-space modelling of tyre degradation is theirs); the search was targeted, not a
systematic review (NOTES section B).

## D. Pending items

1. **Cappello & Hoegh journal version: RESOLVED (read in full).** J. Sports Analytics 12 (2026), doi 10.1177/22150218261446170, CC BY-NC 4.0 as printed on p. 1; PDF supplied by the user (the SAGE bot check blocks scripted download; see `paper/refs/fetch.sh`). Findings: 19 races but each fitted independently, one driver, AR(1) benchmark, practice used only as a level prior, hierarchy and multi-driver left as future work. No claim in section A is pre-empted (NOTES A4); A13 now also reports the journal's per-race AR(1) numbers. Table 4's printed 'Mean Across Races' row contradicts its own caption (see A13 caveat); we do not quote it.
2. **Formula 1 live-timing now requires F1-account authentication** (HTTP 401 from `livetiming.formula1.com`, 404 from the
   FastF1 mirror, probed 2026-10-30; see the check in section C). New sessions cannot be pulled with FastF1's default path;
   `scripts/01_cache_sessions.py` will fail until that is resolved (an authenticated FastF1 session, or OpenF1 as the
   telemetry run used). The cached 2022-2026 data and all `paper/results` are unaffected.
3. **Regenerated results landed (30 Oct 2026).** `pit_loss.json`, `strategy.json`, `transfer.json`, the `strategy` artifact and the new `stint_pool.json` (the `playbook` and `circuits` artifacts were regenerated too and have since been removed from the repository with the product code) were regenerated after the strategy-layer, pit-loss and transfer-stacking fixes; A10-A12, B2, B11 and B12 were re-verified. If any are regenerated again, re-run `python scripts/paper/check_claims.py -v`.
4. **Playbook vs strategy.json stop counts: moot.** An earlier `playbook.json` `actual_median_stops` and
   `strategy.json` `per_race_2026[*].field_stop_counts` differed at several races (British, Dutch, Monaco; the last two are red-flag races). The playbook
   artifact and the product code that displayed it have been removed; only the `strategy.json` counts are quoted.
5. **Not obtained, not cited** (NOTES section C): Heilmeier 2022 dissertation, Tulabandhula & Rudin 2014, Choo 2015 (HTTP 429);
   Quiroga Garcia 2024 (bot challenge); Tremlett & Limebeer, Bekker & Lotz, Steegen et al. multiverse, Diebold & Mariano 1995, Self & Liang
   1987 (closed). Re-run `paper/refs/fetch.sh` later for the rate-limited ones.
6. **Numbers with no JSON source**: the 142,737 lap total (parquet row counts across `data/processed/laps*.parquet`) is not machine-checked here, and the README no longer quotes it.
7. **Re-run timing**: the approximate runtimes in the README were measured by hand and are not in a JSON
   except the fit time (`model.json` -> `engine_agreement.engine_seconds`).
