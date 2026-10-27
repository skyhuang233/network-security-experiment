#!/bin/bash

# Reference launch for Experiment B: capacity calibration (§9, §8.1). No LLM needed.
# Extra args pass through to the experiment's argparse, e.g.:
#   bash runs/exp_b.sh --ber-threshold 0.05
#
# Run from the repo root (the .venv path is relative to it):
#   bash runs/exp_b.sh
# Or in a screen session:
#   screen -L -Logfile runs/exp_b.log -S exp_b bash runs/exp_b.sh

set -euo pipefail
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # first-time uv install does not refresh current shell PATH
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

python -m tasks.exp_b "$@"
