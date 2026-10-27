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
src/network_security_experiment/
  protocol/frame.py        unified logical frame (msg_id/chunk_id/total_chunks/
                           frame_published/CRC-16; 64B baseline chunks)
  channels/                M1 file content, M2 base32 dir names, M3 DiskCache
                           key existence; unified send_payload/receive;
                           ChannelPolicy for experiment D isolation controls
  measurement/recorder.py  dual interception: send-side truth + decode
                           candidates (pre-integrity-drop) + storage ops
  harness/job.py           deterministic coordinator: round loop (publish this
                           round, read next round), XML tool calls, submit /
                           max-rounds termination
  evaluation/metrics.py    the seven metrics (§8)
  monitor/audit.py         W: 10-round windows, content/extended views,
                           measurement-layer labels, TPR/FPR, run-level split
  experiments/             A (main comparison + negative/direct controls),
                           B (capacity calibration), C (persistence/TTL/
                           interference), D (W audit + isolation controls)
  cli.py                   `nse` entry point
tasks/                     30 SILO classes ported to parameterized seeds,
                           local answer oracles, §11 dependency precheck
```

## Usage

```bash
uv sync
uv run pytest                     # deterministic-core test suite (55 tests)

uv run nse precheck               # §11 dependency precheck -> seed registry
uv run nse exp-b                  # capacity calibration (no LLM needed)
uv run nse exp-c                  # persistence curves (no LLM needed)

# LLM runs (recorded experiments use real LLMs only, §1):
export NSE_API_KEY=... NSE_MODEL=...          # + NSE_API_BASE if needed
uv run nse exp-a                   # main comparison (29 classes x 10 seeds x 3 methods)
uv run nse exp-d-isolation --case I-01 --seed 0 --method M1
uv run nse exp-d-audit runs/main-M1-* runs/negctrl-*
```

Configuration (model binding, K, BER threshold) resolves from environment
variables and is stamped into every report (`config.py`, §15).  The
dev-only `ScriptedProvider` replays fixed tool-call scripts to integration-
test the plumbing; every run records its provider name, and scripted runs
are never usable as recorded experiment data.

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
  publications from the physical state alone.  K ≤ 18 is structurally
  infeasible — recorded as a sweep result, not a defect.
- **M3 delayed-read recovery** is bounded by one-publication capacity
  (`(K − 18 − 32)/8` payload bytes at K): multi-segment messages require
  reading every round; a sender that stops refreshing leaves only the last
  publication on the key set.  This is the carrier's structural property
  (§6 note), documented in `experiments/exp_c.py`.

## References

- SILO-BENCH source is pinned under `references/silo-bench/acl26-silo-bench`
  (git commit recorded by `config.py` into each report).  Task generation,
  data sharding, round visibility semantics and the LLM client are reused;
  the round loop, tools and storage layer are rewritten per the design doc
  (§4 "new harness, reuse tasks and scheduling semantics only").
