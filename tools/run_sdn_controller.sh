#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/backend/src:${PYTHONPATH:-}"
export MLDDOS_MODEL="${MLDDOS_MODEL:-selected_model}"
export MLDDOS_OBSERVE_ONLY="${MLDDOS_OBSERVE_ONLY:-1}"
export MLDDOS_THRESHOLD="${MLDDOS_THRESHOLD:-0.95}"
export MLDDOS_POLL_INTERVAL="${MLDDOS_POLL_INTERVAL:-3}"
export MLDDOS_PPS_THRESHOLD="${MLDDOS_PPS_THRESHOLD:-800}"
export MLDDOS_EVENTS_CSV="${MLDDOS_EVENTS_CSV:-$PROJECT_ROOT/results/live_events.csv}"
export MLDDOS_RESET_EVENTS="${MLDDOS_RESET_EVENTS:-0}"

exec ryu-manager "$PROJECT_ROOT/sdn_ryu_detector.py"
