#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    # Ensure requirements are up to date
    pip install -r requirements.txt > /dev/null 2>&1
fi

# Set display for GUI if not set (assuming user is logged in on :0)
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
fi

# Run the agent using python3 (Use new Hailo optimized agent)
exec python3 agent_hailo.py "$@"

