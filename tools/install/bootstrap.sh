#!/usr/bin/env bash
# Source from setup/launch. The caller defines ROOT; all messages go to stderr.
# Any Julia from the release in julia_min_version.txt on is accepted (1.13 means 1.13.0, 1.13.5, 1.14, 2.0 ...);
# julia_version.txt names the release downloaded when no acceptable Julia is found.

# True when the text is "julia version X.Y..." with X.Y at or above the minimum release.
ludwig_julia_acceptable() {
  local text="$1" minimum major minor rest min_major min_minor
  minimum="$(tr -d '\r\n' < "$ROOT/tools/install/julia_min_version.txt")"
  case "$text" in "julia version "*) ;; *) return 1;; esac
  IFS='.-+' read -r major minor rest <<< "${text#julia version }"
  IFS='.' read -r min_major min_minor rest <<< "$minimum"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$min_major" =~ ^[0-9]+$ && "$min_minor" =~ ^[0-9]+$ ]] || return 1
  (( 10#$major > 10#$min_major || (10#$major == 10#$min_major && 10#$minor >= 10#$min_minor) ))
}

ludwig_julia() {
  local install="${1:-no}" version minimum runtime candidate detected system arch record url digest stage archive actual
  version="$(tr -d '\r\n' < "$ROOT/tools/install/julia_version.txt")"
  minimum="$(tr -d '\r\n' < "$ROOT/tools/install/julia_min_version.txt")"
  export LUDWIG_PRIVATE_ROOT="${LUDWIG_PRIVATE_ROOT:-$ROOT/../02_PRIVATE_PROJECT_DEVELOPMENT}"
  case "$LUDWIG_PRIVATE_ROOT" in /*) ;; *) LUDWIG_PRIVATE_ROOT="$PWD/$LUDWIG_PRIVATE_ROOT";; esac
  runtime="$LUDWIG_PRIVATE_ROOT/runtime/julia-$version"
  local explicit="${LUDWIG_JULIA:-${AIRFOIL_JULIA:-}}"
  if [ -n "$explicit" ]; then
    detected="$("$explicit" --startup-file=no --version 2>/dev/null || true)"
    ludwig_julia_acceptable "$detected" || { echo "Julia $minimum or later is required; check LUDWIG_JULIA / AIRFOIL_JULIA." >&2; return 1; }
    LUDWIG_JULIA_BIN="$explicit"; return
  fi
  for candidate in "$runtime/julia-$version/bin/julia" "$runtime/julia-$version.app/Contents/Resources/julia/bin/julia" "$(command -v julia || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    detected="$("$candidate" --startup-file=no --version 2>/dev/null || true)"
    if ludwig_julia_acceptable "$detected"; then LUDWIG_JULIA_BIN="$candidate"; return; fi
  done
  [ "$install" = yes ] || { echo "No Julia $minimum or later was found. Run bash setup.sh (Linux) or bash setup.command (Mac)." >&2; return 1; }
  case "$(uname -s)" in Linux) system=linux;; Darwin) system=mac;; *) echo 'Use setup.cmd on Windows.' >&2; return 1;; esac
  case "$(uname -m)" in x86_64|amd64) arch=x86_64;; arm64|aarch64) arch=aarch64;; *) echo 'A 64-bit x86 or ARM system is required.' >&2; return 1;; esac
  record="$(awk -F '|' -v key="$system-$arch" '$1==key {print $0}' "$ROOT/tools/install/runtimes.tsv")"
  IFS='|' read -r candidate url digest <<< "$record"
  digest="${digest//$'\r'/}"
  [ -n "$url" ] || { echo 'No Julia download is configured for this system.' >&2; return 1; }
  command -v tar >/dev/null || { echo 'Install tar using your system package manager.' >&2; return 1; }
  mkdir -p "$LUDWIG_PRIVATE_ROOT/TEMP_DELETEABLE/setup" "$LUDWIG_PRIVATE_ROOT/runtime"
  stage="$(mktemp -d "$LUDWIG_PRIVATE_ROOT/TEMP_DELETEABLE/setup/julia.XXXXXXXX")"
  archive="$stage/julia.tar.gz"
  echo "Downloading Julia $version ($system / $arch)..." >&2
  if command -v curl >/dev/null; then curl --fail --location --retry 3 "$url" -o "$archive"
  elif command -v wget >/dev/null; then wget -O "$archive" "$url"
  else echo 'Install curl or wget using your system package manager.' >&2; return 1; fi
  if command -v sha256sum >/dev/null; then actual="$(sha256sum "$archive" | awk '{print $1}')"
  elif command -v shasum >/dev/null; then actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
  else echo 'Install sha256sum or shasum to verify the Julia download.' >&2; return 1; fi
  [ "$actual" = "$digest" ] || { echo "Julia checksum mismatch. No runtime installed. Download: $archive" >&2; return 1; }
  echo 'Checksum verified. Extracting the private Julia runtime...' >&2
  mkdir "$stage/unpacked"
  tar -xzf "$archive" -C "$stage/unpacked"
  [ ! -e "$runtime" ] || { echo "Incomplete runtime exists at $runtime; rename it, then rerun setup." >&2; return 1; }
  mv "$stage/unpacked" "$runtime"
  rm -- "$archive"
  ludwig_julia no
}
