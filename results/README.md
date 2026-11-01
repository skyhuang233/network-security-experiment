# Experiment results

Recorded outputs of the four-experiment suite. The raw per-run measurement
logs (`outputs/runs/`, 56 MB) stay gitignored; everything aggregated is here.

Scope of the recorded LLM runs (experiments A and D): 6 representative
SILO-BENCH classes (I-06, I-08, II-11, II-12, III-21, III-24) x 5 seeds x
3 carriers (M1 file content, M2 directory names, M3 cache-key existence),
plus the separated-backend negative control = 180 runs. Model:
`deepseek-ai/DeepSeek-V4.1-Flash` via SiliconFlow. Experiments B and C are
deterministic and cover all three carriers without an LLM.

## Files

- `exp_b.json` — capacity calibration (54 points).
- `exp_c.json` — persistence / TTL / interference (210 points, 3 repeats).
- `exp_a.json` — main comparison: 90 task runs + 90 negative-control runs,
  with per-run channel metrics and task submissions.
- `exp_d_audit.json` — monitor W's audit: TPR/FPR per carrier x view x
  split over 1956 windows.
- `audit/<view>/<run>.json` — W's per-window verdicts and the
  measurement-layer ground-truth labels for every audited run (360 files).
- `precheck/seed-registry.json` — the seed registry the runs were drawn
  from (290 valid instances; I-03 excluded as structurally degenerate).
- `config.json` — the resolved configuration stamped into the reports.

## Headline numbers

Capacity (exp B), BER 0 throughout: M1 2.79 Mbit/s, M2 2.82 Mbit/s,
M3 (K=128) 34.2 Kbit/s.

Persistence (exp C): active cleanup kills all three carriers; background
activity and a 1 s TTL kill only M3; M1 and M2 survive 300 s untouched.

Task success (exp A): M1 73 %, M2 63 %, M3 7 % correct, BER 0 on every
delivered bit. The negative control scores 0 % with zero bits transferred,
so success comes from the channel rather than the model's own knowledge.

Audit (exp D): the extended view is decisive for M2 and M3. M2 reaches
TPR 1.0 at FPR 0.01-0.04; M3 reaches FPR 0.0 but only TPR 0.77-0.84; M1
stays at TPR 1.0 with FPR 0.36-0.49 under both views. The content-only view
drops M2's TPR to 0.31-0.50 and pushes M3's FPR above 0.57.
