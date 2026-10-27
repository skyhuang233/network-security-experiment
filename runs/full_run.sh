#!/bin/bash

# Reference full pipeline for the recorded experiment suite, end to end:
#   precheck -> exp_b (capacity) -> exp_c (persistence) -> exp_a (main comparison
#   + negative control) -> exp_d audit over the run dirs exp_a just produced.
# Requires NSE_API_KEY / NSE_MODEL in the environment (exp_a / exp_d use real LLMs).
# Takes no arguments; for a partial run or custom args, use runs/exp_<x>.sh instead
# (exp_d isolation stays a manual per-case spot-check via runs/exp_d.sh).
#
# Run from the repo root (the .venv path is relative to it):
#   bash runs/full_run.sh
# The whole suite is long; consider a screen session:
#   screen -L -Logfile runs/full_run.log -S full_run bash runs/full_run.sh

set -euo pipefail
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # first-time uv install does not refresh current shell PATH
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

if [ -z "${NSE_API_KEY:-}" ] || [ -z "${NSE_MODEL:-}" ]; then
    echo "error: NSE_API_KEY and NSE_MODEL are required for LLM experiments" >&2
    exit 1
fi

# §11 dependency precheck -> seed registry (outputs/precheck/seed-registry.json)
nse precheck

# no-LLM experiments
python -m tasks.exp_b
python -m tasks.exp_c

# LLM experiments
python -m tasks.exp_a
python -m tasks.exp_d audit outputs/runs/main-* outputs/runs/negctrl-*
