#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/tools/install/bootstrap.sh"
install=yes
for arg in "$@"; do
  case "$arg" in
    --offline|--dry-run|--help|-h) install=no;;
    --no-precompile|--no-verify) ;;
    *) echo "Unknown setup option: $arg. Use --help." >&2; exit 2;;
  esac
done
ludwig_julia "$install"
exec "$LUDWIG_JULIA_BIN" --startup-file=no "$ROOT/tools/install/setup.jl" "$@"
