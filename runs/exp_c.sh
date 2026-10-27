#!/bin/bash

# Reference launch for Experiment C: lifecycle, persistence and interference (§9, §8.6). No LLM needed.
# Extra args pass through to the experiment's argparse, e.g.:
#   bash runs/exp_c.sh --delays 0 10 60 --m3-k 64
#
# Run from the repo root (the .venv path is relative to it):
#   bash runs/exp_c.sh
# Or in a screen session:
#   screen -L -Logfile runs/exp_c.log -S exp_c bash runs/exp_c.sh

set -euo pipefail
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # first-time uv install does not refresh current shell PATH
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

python -m tasks.exp_c "$@"
