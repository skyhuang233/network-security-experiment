#!/bin/bash

# Reference launch for Experiment D: monitoring and isolation (§9, §10).
# Requires NSE_API_KEY / NSE_MODEL in the environment (W judge / task agents).
# The subcommand (audit / isolation) and its args pass through, e.g.:
#   bash runs/exp_d.sh isolation --case I-01 --seed 0 --method M1
#   bash runs/exp_d.sh audit outputs/runs/main-M1-* outputs/runs/negctrl-*
#
# Run from the repo root (the .venv path is relative to it):
#   bash runs/exp_d.sh <subcommand> ...
# Or in a screen session:
#   screen -L -Logfile runs/exp_d.log -S exp_d bash runs/exp_d.sh audit outputs/runs/main-*

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

python -m tasks.exp_d "$@"
