#!/usr/bin/env bash
# Builds lq.rs with stable rustc, compiles every probe against it, and
# compares each probe's first error with its expectation.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="$here/target/probes"
mkdir -p "$out"
rustc --edition 2024 --crate-type=lib --crate-name lq "$here/lq.rs" -o "$out/liblq.rlib"

for probe in "$here"/probes/q*.rs; do
  expect=$(sed -n 's|^// expect: ||p' "$probe" | head -1)
  start=$(date +%s%N)
  diagnostics=$(rustc --edition 2024 --crate-type=lib --emit=metadata -o "$out/probe.rmeta" \
    --extern lq="$out/liblq.rlib" -L "$out" "$probe" 2>&1 || true)
  ms=$(( ($(date +%s%N) - start) / 1000000 ))
  first=$(printf '%s\n' "$diagnostics" | grep -m1 -E '^error' | sed -E 's/^error(\[(E[0-9]+)\])?: ?(.*)$/\2|\3/' || true)
  if [ -z "$first" ]; then
    got=pass
  elif [ -n "${first%%|*}" ]; then
    got=${first%%|*}
  else
    got=${first#*|}
  fi
  [ "$got" = "$expect" ] && verdict=as-expected || verdict=UNEXPECTED
  printf '%-32s expect=%-36s got=%-36s %5sms %s\n' "$(basename "$probe")" "$expect" "$got" "$ms" "$verdict"
done
