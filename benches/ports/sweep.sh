#!/usr/bin/env bash
#
# One sweep over the six runtimes for the rows docs/performance.md carries.
#
# Per row: one warm-up execution per runtime, discarded, then three reps that
# alternate across the six runtimes one row at a time. `uptime` is written
# before every rep and a rep waits while the 1-minute load average is 16 or
# above, which is the protocol acvus-interpreter-test/benches/README.md states.
# acvus and Rust are one execution of the bench binary, so the sweep's
# `execute/us` and `rust/us` are drawn in the same rep as the ports beside them.
#
# Every execution runs under `setarch -R taskset -c $CORE`: `-R` fixes the load
# base the kernel gives a position-independent binary, `taskset` pins the one
# thread these rows run on.
#
#   BENCH_BINS=<dir of accum/shapes/programs/mandelbrot/attention> \
#     CORE=15 benches/ports/sweep.sh <out-dir>
#
# The internal rep count each port takes is the bench binary's own for that
# row, so both sides of a row report the median of the same number of samples.

set -u -o pipefail

CORE=${CORE:-15}
REPS=${REPS:-3}
N=${N:-1000000}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_BINS=${BENCH_BINS:?BENCH_BINS names the directory holding the bench binaries}
ROWS=${ROWS:-$HERE/rows.txt}
OUT=${1:?the output directory}
mkdir -p "$OUT"

PIN=(setarch -R taskset -c "$CORE")

below16() { awk -v l="$1" 'BEGIN{exit !(l < 16)}'; }

gate() {
  while :; do
    local load
    load=$(awk '{print $1}' /proc/loadavg)
    below16 "$load" && return 0
    sleep 5
  done
}

slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '-' | tr -s '-' | sed 's/^-//; s/-$//'; }

# Prints "<us> <us>" — acvus's execute and its Rust twin — for one row, from
# one execution of the bench binary.
acvus_and_rust() {
  local bench=$1 row=$2 exec_at=$3 rust_at=$4 pattern=$5
  local -a env_pairs=("${@:6}")
  env "${env_pairs[@]}" "${PIN[@]}" "$BENCH_BINS/$bench" |
    awk -v p="$pattern" -v e="$exec_at" -v r="$rust_at" \
      '$0 ~ p && !seen { print $(NF-e), $(NF-r); seen = 1 }'
}

# Prints the median microseconds one port reported, and checks its value.
port() {
  local want=$1
  shift
  local line
  line=$("${PIN[@]}" "$@")
  local us value
  us=$(printf '%s' "$line" | cut -f2)
  value=$(printf '%s' "$line" | cut -f3)
  if [ "$value" != "$want" ]; then
    echo "VALUE MISMATCH: $* produced $value, the Rust twin produces $want" >&2
    printf '%s VALUE-MISMATCH(%s)' "$us" "$value"
    return 0
  fi
  printf '%s' "$us"
}

# The row table: bench, row, the port's argument list, the bench binary's
# selector, and where the two columns sit from the end of the printed line.
row_spec() {
  case "$1" in
  accum:*)
    ROW=${1#accum:}
    BENCH=accum
    PORT_ARGS=("$ROW" "$N" 4)
    BENCH_ENV=("ACCUM_CASE=$ROW")
    PATTERN="^ *${ROW//|/[|]} +$N "
    EXEC_AT=3
    RUST_AT=2
    ;;
  shapes:*)
    ROW=${1#shapes:}
    BENCH=shapes
    PORT_ARGS=("$ROW" "$N" 4)
    BENCH_ENV=("SHAPES_CASE=$ROW")
    PATTERN="^ *$ROW +$N "
    EXEC_AT=3
    RUST_AT=2
    ;;
  programs:*)
    ROW=${1#programs:}
    BENCH=programs
    PORT_ARGS=("$ROW" "$N" 3)
    BENCH_ENV=("PROGRAMS_CASE=$ROW")
    PATTERN="^ *$ROW +$N "
    EXEC_AT=3
    RUST_AT=2
    ;;
  mandelbrot:*)
    ROW=${1#mandelbrot:}
    BENCH=mandelbrot
    PORT_ARGS=("$ROW" 4)
    BENCH_ENV=()
    PATTERN="^ *$ROW "
    EXEC_AT=3
    RUST_AT=2
    ;;
  attention:*)
    ROW=${1#attention:}
    BENCH=attention
    PORT_ARGS=("$ROW" 4)
    BENCH_ENV=()
    PATTERN="^ *vec +$ROW "
    EXEC_AT=4
    RUST_AT=2
    ;;
  *)
    echo "no row spec for $1" >&2
    exit 1
    ;;
  esac
}

sweep_row() {
  local spec=$1 want=$2
  row_spec "$spec"
  local file="$OUT/$(slug "$BENCH-$ROW").txt"
  {
    echo "== $BENCH $ROW"
    echo "== Rust twin's value: $want"
    echo "== core $CORE, setarch -R, $REPS reps, one warm-up per runtime discarded"
  } >"$file"

  local -a py=(python3 "$HERE/python/$BENCH.py" "${PORT_ARGS[@]}")
  local -a js=(node "$HERE/js/$BENCH.js" "${PORT_ARGS[@]}")
  local -a jsj=(node --jitless "$HERE/js/$BENCH.js" "${PORT_ARGS[@]}")
  local -a l54=(lua5.4 "$HERE/lua/$BENCH.lua" "${PORT_ARGS[@]}")
  local -a ljt=(luajit "$HERE/lua/$BENCH.lua" "${PORT_ARGS[@]}")

  echo "== warm-up (discarded)" >>"$file"
  acvus_and_rust "$BENCH" "$ROW" "$EXEC_AT" "$RUST_AT" "$PATTERN" "${BENCH_ENV[@]}" >/dev/null
  port "$want" "${py[@]}" >/dev/null
  port "$want" "${js[@]}" >/dev/null
  port "$want" "${jsj[@]}" >/dev/null
  port "$want" "${l54[@]}" >/dev/null
  port "$want" "${ljt[@]}" >/dev/null

  local rep
  for rep in $(seq 1 "$REPS"); do
    gate
    {
      echo "== rep $rep"
      uptime
    } >>"$file"
    local pair
    pair=$(acvus_and_rust "$BENCH" "$ROW" "$EXEC_AT" "$RUST_AT" "$PATTERN" "${BENCH_ENV[@]}")
    echo "acvus $(printf '%s' "$pair" | cut -d' ' -f1)" >>"$file"
    echo "rust $(printf '%s' "$pair" | cut -d' ' -f2)" >>"$file"
    echo "cpython $(port "$want" "${py[@]}")" >>"$file"
    echo "node $(port "$want" "${js[@]}")" >>"$file"
    echo "node-jitless $(port "$want" "${jsj[@]}")" >>"$file"
    echo "lua5.4 $(port "$want" "${l54[@]}")" >>"$file"
    echo "luajit $(port "$want" "${ljt[@]}")" >>"$file"
  done
  echo "done $BENCH $ROW"
}

while IFS=$'\t' read -r spec want; do
  [ -z "$spec" ] && continue
  case "$spec" in \#*) continue ;; esac
  sweep_row "$spec" "$want"
done <"$ROWS"
