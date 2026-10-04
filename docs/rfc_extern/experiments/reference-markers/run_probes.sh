#!/usr/bin/env bash
# Builds the model with stable rustc, compiles every probe against it, and
# compares each probe's first error with the expectation on its second line.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="$here/target/probes"
mkdir -p "$out"
rustc --edition 2024 --crate-type=lib --crate-name model "$here/model.rs" -o "$out/libmodel.rlib"

for probe in "$here"/probes/p*.rs; do
  expect=$(sed -n 's|^// expect: ||p' "$probe" | head -1)
  diagnostics=$(rustc --edition 2024 --crate-type=lib --emit=metadata -o "$out/probe.rmeta" \
    --extern model="$out/libmodel.rlib" -L "$out" "$probe" 2>&1 || true)
  first=$(printf '%s\n' "$diagnostics" | grep -m1 -E '^error' | sed -E 's/^error(\[(E[0-9]+)\])?: ?(.*)$/\2|\3/' || true)
  if [ -z "$first" ]; then
    got=pass
  elif [ -n "${first%%|*}" ]; then
    got=${first%%|*}
  else
    got=${first#*|}
  fi
  [ "$got" = "$expect" ] && verdict=as-expected || verdict=UNEXPECTED
  printf '%-36s expect=%-38s got=%-38s %s\n' "$(basename "$probe")" "$expect" "$got" "$verdict"
done
