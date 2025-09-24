#!/usr/bin/env bash
set -Eeuo pipefail

# ---- Resolve repo dir (where this script lives) ----
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
cd "$SCRIPT_DIR"

# ---- Pick Python: prefer venv if available ----
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "ERROR: No Python found. Install Python 3 and/or create .venv first." >&2
  exit 1
fi

# ---- Fixed command ----
CMD=( "$PY" -m src.daemon \
  --interval-sec 300 \
  --use-binance true \
  --timeframe 1h \
  --universe-size 80 \
  --testnet true \
  --live false )

# ---- Optional tmux mode ----
RUN_IN_TMUX="${RUN_IN_TMUX:-0}"
TMUX_SESSION="${TMUX_SESSION:-trader}"

wrap_sleep_inhibitor() {
  if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -dimsu -- "$@"
  elif command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --why="llm-crypto-daemon" --what="sleep" --mode=block "$@"
  else
    exec "$@"
  fi
}

if [[ "$RUN_IN_TMUX" == "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: RUN_IN_TMUX=1 but tmux not found. Install tmux or unset RUN_IN_TMUX." >&2
    exit 1
  fi
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TMUX_SESSION"
  fi
  if command -v caffeinate >/dev/null 2>&1; then
    tmux new-session -d -s "$TMUX_SESSION" "caffeinate -dimsu -- ${CMD[*]}"
  elif command -v systemd-inhibit >/dev/null 2>&1; then
    tmux new-session -d -s "$TMUX_SESSION" "systemd-inhibit --why='llm-crypto-daemon' --what='sleep' --mode=block ${CMD[*]}"
  else
    tmux new-session -d -s "$TMUX_SESSION" "${CMD[*]}"
  fi
  echo "Started tmux session: $TMUX_SESSION"
  echo "Attach:  tmux attach -t $TMUX_SESSION"
  echo "Detach:  Ctrl-b then d"
  echo "Kill:    tmux kill-session -t $TMUX_SESSION"
  exit 0
else
  wrap_sleep_inhibitor "${CMD[@]}"
fi