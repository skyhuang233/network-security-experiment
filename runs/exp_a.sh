#!/bin/bash

# Reference launch for Experiment A: standard task comparison (§9) + controls.
# Requires NSE_API_KEY / NSE_MODEL in the environment (real LLM task agents).
# Extra args pass through to the experiment's argparse, e.g. a smoke run:
#   bash runs/exp_a.sh --classes I-01 --seeds-per-class 1
#
# Run from the repo root (the .venv path is relative to it):
#   bash runs/exp_a.sh
# The main run (29 classes x 10 seeds x 3 methods + negative control) is long;
# consider a screen session:
#   screen -L -Logfile runs/exp_a.log -S exp_a bash runs/exp_a.sh

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

python -m tasks.exp_a "$@"
