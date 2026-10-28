# Shared-Storage Channel Measurement on SILO-BENCH Task Agents

Two real LLM task-agents (A, B) collaborate on SILO-BENCH distributed tasks
inside an authorized sandbox and may exchange information **only through
shared storage**: file *bodies* (M1), directory *names* (M2), or
cache-key *existence* (M3).  The measurement layer records the actual
information flow and the evaluator computes seven channel metrics; a
separate offline monitor agent (W) audits the logs under two audit views.

Design doc: [`.scratch/experiment-design-plan.md`](.scratch/experiment-design-plan.md)
(follow it for all section references below).

## Layout

```
src/
  frame.py                 unified logical frame (msg_id/chunk_id/total_chunks/
                           frame_published/CRC-16; 64B baseline chunks)
  base.py                  shared agent-facing machinery (chunking/framing/
                           round visibility/reassembly/measurement) + the
                           GENERIC isolation policy (experiment D controls);
                           build_views bridges CaseRunConfig -> per-agent views
  m1_file.py,              M1 file content, M2 base32 dir names, M3 DiskCache
  m2_dirname.py,           key existence; each carrier owns its physical layer
  m3_cache.py              (staging/promotion/reading) and claims its own
                           carrier params (M3 K/TTL, body/name limits) from
                           CaseRunConfig at construction
  metrics.py               the seven metrics (§8) + experiment-B capacity
                           selection (highest repeatable rate per unit)
  audit_agent.py           W: 10-round windows, content/extended views,
                           measurement-layer labels, TPR/FPR, run-level split;
                           also the OpenAI-compatible LLM client (§13)
                           shared by A/B and W (provider_from_env,
                           ScriptedProvider)
  eval_cli.py              `nse` entry point (precheck registry)
tasks/                     experiment scripts (nanochat-style standalone runs);
  exp_a.py                 A (main comparison + negative/direct controls);
                           opens and saves that run's measurement log
  exp_b.py                 B (capacity calibration); owns its measurement log
  exp_c.py                 C (persistence/TTL/interference); owns its measurement log
  exp_d.py                 D (W audit + isolation controls); owns its measurement log
benchmark_generator/      30 SILO classes ported to parameterized seeds
                           (paradigm I/II/III generators, seed-parameterized);
                           generator.py: instance assembly, local oracles,
                           §11 dependency precheck -> seed-registry.json
config/                    resolved run configuration (§15): model binding,
                           K, BER threshold, stamped into every report;
                           CaseRunConfig (per-run experiment configuration)
```

## Usage

Launch scripts (nanochat-style) live in `runs/`; each self-bootstraps the
environment (uv, `.venv`, `uv sync`, activation) and runs from the repo root.

```bash
bash runs/exp_b.sh            # capacity calibration (no LLM needed)
bash runs/exp_c.sh            # persistence curves (no LLM needed)

# LLM runs (recorded experiments use real LLMs only, §1):
export NSE_API_KEY=... NSE_MODEL=...          # + NSE_API_BASE if needed
bash runs/exp_a.sh            # main comparison (29 classes x 10 seeds x 3 methods)
bash runs/exp_d.sh isolation --case I-01 --seed 0 --method M1
bash runs/exp_d.sh audit outputs/runs/main-M1-* outputs/runs/negctrl-*
bash runs/full_run.sh         # entire recorded pipeline, end to end
```

Extra args pass through to each experiment's argparse (e.g.
`bash runs/exp_a.sh --classes I-01 --seeds-per-class 1`); long runs fit a
screen session (`screen -L -Logfile runs/exp_a.log -S exp_a bash runs/exp_a.sh`).

Configuration (model binding, K, BER threshold) resolves from environment
variables and is stamped into every report (`config`, §15); each experiment
script records the configuration it ACTUALLY consumed (argparse values after
env-defaulting, per-run `run_config` event in the measurement log), so the
record can never drift from the executed run.  The monitor W binds
`NSE_W_MODEL` (falling back to `NSE_MODEL`).  The dev-only `ScriptedProvider`
replays fixed tool-call scripts to integration-test the plumbing
(`python -m tasks.exp_a --provider scripted`, no API calls); every run
records its provider name, and scripted runs are never usable as recorded
experiment data.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Covers the tool-call protocol (matrix payloads, no silent numeric
conversion), correct-delivery truth (CRC-16 collisions do NOT count as
delivered; length-mismatched candidates never get a zero BER), W's audit
information boundary (physical backend events per method, never the
send_payload plaintext), window ground-truth labeling (cross-window
communication, failed-transmission attempts, separated controls), capacity
selection (highest rate, not largest payload), resource-overhead invariants
(null occupancy for unmeasurable backends, publish-side probe counting),
M3 logical-id addressing (out-of-range rejection, refusal when the id space is exhausted),
persistence pairing (end-to-end exp_c runs: common active cleanup on all
carriers, real M3 eviction pressure, repeat-estimated curves, paired
payload sha256 across methods/delays/repeats), and an end-to-end exp_a
smoke test with the scripted provider.  The exp_c coverage drives the real
entry point as a subprocess (the script's module-level main flow cannot be
imported for unit testing).

## Findings baked into the implementation

- **I-03 is structurally degenerate at n=2**: the forced-majority vote
  construction puts the global winner in *both* shards, so both agents are
  always locally self-sufficient.  Recorded and excluded by the precheck
  (never silently dropped); the other 29 classes yield 10 valid instances
  each with the reseeding process recorded.
- **I-04 degenerate rate ≈ 50%** at n=2 (no-ERROR seeds are trivially
  self-sufficient for both agents); the plan's pilot ~30% was measured on a
  different seed set.  Reseeding is exercised accordingly.
- **M3 physical publication state**: the key set carries a publication
  toggle + activity marker in reserved keys k16/k17 (18 reserved keys
  total), so even a *fresh* receiver (persistence test) can detect new
  publications from the physical state alone.  K ≤ 18 being infeasible is a
  limit of this protocol implementation (18 reserved header keys), not an
  inherent property of cache-key existence channels — recorded as a sweep
  result, not a defect, and not a universal carrier conclusion.
- **M3 logical-id addressing is bounded by the reduced physical header**
  (msg_id 8 bits, chunk_id 6 bits, total_chunks 8 bits): out-of-range frames
  are REJECTED at send time (never silently truncated/aliased).  After 256
  messages the sender refuses further sends (`carrier_msg_id_exhausted`);
  ids are not wrapped, so measurement history and delivery identity are
  not overwritten.  This is a protocol implementation limit; the underlying
  carrier could address more with a wider encoding.
- **M3 delayed-read recovery** is bounded by one-publication capacity
  (`(K − 18 − 32) // 8` payload bytes).  A positive length needs K ≥ 58;
  smaller K (including the K=32 calibration point) is reported as
  「协议前提不满足」 and is not scored as a Persistence failure.
  Multi-segment messages require reading every round; a sender that stops
  refreshing leaves only the last publication on the key set.  This bound
  comes from the one-segment-per-round choice of this protocol
  implementation (§6 note), documented in `tasks/exp_c.py` — it is not a
  universal property of cache-key existence channels.
- **Experiment C pairing (§8.6)**: payloads are pre-generated once and
  reused byte-identically across delays, methods and repeats; common
  conditions run all carriers at the same payload bytes (sized to M3's
  single-publication maximum; K < 58 is a protocol-precondition result,
  not a persistence failure), the per-method native payload size is a
  separate condition; active cleanup physically removes live objects on all
  three carriers; M3's background-activity condition runs under real
  same-cache LRU eviction pressure; every point is estimated over repeats.

## References

- SILO-BENCH source is pinned under `references/silo-bench/acl26-silo-bench`
  (git commit recorded by `config` into every report).  That directory is
  gitignored, so the upstream pin (repo URL + commit) is ALSO committed in
  `config/environment.py` and a clean checkout falls back to it — upstream
  provenance never depends on an untracked directory.  Task generation,
  data sharding, round visibility semantics and the LLM client are reused;
  the round loop, tools and storage layer are rewritten per the design doc
  (§4 "new harness, reuse tasks and scheduling semantics only").
