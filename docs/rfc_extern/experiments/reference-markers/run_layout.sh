#!/usr/bin/env bash
# Builds layout.rs with nightly rustc, once without layout randomization and
# once at each seed from 1 to 32, and counts how often each view's layout
# matched the generic view's.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="$here/target/layout"
mkdir -p "$out"

run() {
  rustc +nightly --edition 2024 --crate-type=lib --crate-name model "$@" "$here/model.rs" -o "$out/libmodel.rlib"
  rustc +nightly --edition 2024 "$@" --extern model="$out/libmodel.rlib" -L "$out" "$here/layout.rs" -o "$out/layout"
  "$out/layout"
}

echo "without randomization:"
run

echo "-Zrandomize-layout, seeds 1 to 32 (count, view, verdict):"
for seed in $(seq 1 32); do
  run -Zrandomize-layout "-Zlayout-seed=$seed"
done | sort | uniq -c
