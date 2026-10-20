#!/usr/bin/env bash
# Builds the probes against the repository's compiler and prints each
# outcome beside its expectation.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CARGO_TARGET_DIR="$here/target" cargo run --quiet --manifest-path "$here/Cargo.toml"
