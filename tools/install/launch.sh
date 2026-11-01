#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/install/bootstrap.sh"
ludwig_julia no
app="${1:-workbench}"
if [ "$#" -gt 0 ]; then shift; fi
case "$app" in
  workbench) entry=apps/ludwig_web/run.jl; project=third_party/JXfoil;;
  airfoil) entry=apps/airfoil_web/run.jl; project=third_party/JXfoil;;
  solver) entry=src/main.jl; project=src;;
  *) echo "Unknown application: $app" >&2; exit 2;;
esac
if [ "$(uname -s)" = Darwin ] && [ "$app" = solver ]; then
  echo 'The current CUDA solver is supported on Windows and Linux. Use the web workbench for JXfoil and results on Mac.' >&2
  exit 1
fi
mkdir -p "$LUDWIG_PRIVATE_ROOT/TEMP_DELETEABLE/launchers"
log="$LUDWIG_PRIVATE_ROOT/TEMP_DELETEABLE/launchers/${app}_$(date -u +%Y%m%d_%H%M%S)_$$.log"
echo "Starting OpenAeroLab ($app). Log: $log"
cd "$ROOT"
"$LUDWIG_JULIA_BIN" --startup-file=no --project="$ROOT/$project" --threads=auto "$ROOT/$entry" "$@" 2>&1 | tee "$log"
