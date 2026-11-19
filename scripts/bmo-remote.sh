#!/usr/bin/env bash
# Run scripts/bmoctl.py on the Pi over SSH from a development machine.
#
#   scripts/bmo-remote.sh status
#   scripts/bmo-remote.sh say "what time is it" --quiet
#   scripts/bmo-remote.sh turns -n 5
#   scripts/bmo-remote.sh screenshot            # copies the image back, prints its local path
#   scripts/bmo-remote.sh logs [-n 80]          # recent service log
#
# BMO_HOST selects the device (default selop@raspi); BMO_DIR its checkout.
set -euo pipefail

HOST="${BMO_HOST:-selop@raspi}"
DIR="${BMO_DIR:-Git/be-more-hailo}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST")

if [ "${1:-}" = "logs" ]; then
    shift
    n=80
    if [ "${1:-}" = "-n" ]; then n="$2"; fi
    exec "${SSH[@]}" "sudo journalctl -u bmo-gui -n $n --no-pager -o cat"
fi

if [ "${1:-}" = "screenshot" ]; then
    local_file="${2:-${TMPDIR:-/tmp}/bmo-screenshot-$(date +%H%M%S).png}"
    "${SSH[@]}" "cd $DIR && venv/bin/python scripts/bmoctl.py screenshot logs/screenshot.png" > /dev/null
    scp -q -o BatchMode=yes "$HOST:$DIR/logs/screenshot.png" "$local_file"
    echo "$local_file"
    exit 0
fi

# Quote every argument so text with spaces survives the remote shell
args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
exec "${SSH[@]}" "cd $DIR && venv/bin/python scripts/bmoctl.py$args"
