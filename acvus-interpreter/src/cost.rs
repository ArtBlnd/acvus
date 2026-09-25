//! The interpreter's cost table (RFC-0066 rule 8): what each operation
//! family weighs here, in ticks, fixed in the backend.
//!
//! A tick is one interpreter dispatch: 0.5527 ns, the `accum` bench's
//! `for range` row at n = 1 000 000 (median of five reps, 1105.4 µs)
//! divided by the two dispatches one of its iterations runs, the `For` op
//! and one `Add`. Every row below was derived once, from the benches in
//! `acvus-interpreter-test/benches` measured on 2026-09-25 at master
//! 83a90e8b under that directory's README protocol, core 15. A row is the
//! difference between two bench rows over the dispatches that separate them,
//! rounded to a whole tick:
//!
//! - arithmetic is one dispatch, by the tick's definition.
//! - compare: `for range break` (1480.1 µs) runs an `Eq` and its branch
//!   beside `for range`'s two dispatches, 0.37 ns more per iteration,
//!   which rounds up to one tick. A branch weighs nothing of its own and
//!   the compare before it carries it.
//! - load: `for slice` (4297.5 µs) reads the element in its step and
//!   through the reference in its body, two loads over `for range`,
//!   1.60 ns each.
//! - extern call: `extern while` (1662.0 µs) is taken as `for range`'s
//!   two dispatches and one call of a trivial extern, 0.56 ns more.
//! - allocation: `trim owned` (18478.0 µs) allocates and releases a
//!   `String` where `trim view` (6789.1 µs) does not, 11.7 ns more.
//! - spawn: `spawn`'s `straight-8` at spin 45 000, `heavy/opaque`, runs
//!   eight ordered calls through the blocking pool where `sync/pure` runs
//!   them in place, 10.1 µs more per call in the median rep.
//!
//! No bench isolates the other rows, so each is stated from the measured
//! rows: a store is a load's write, a local call is a call as an extern's
//! is, a merge is one dispatch, a chunk handed to the executor is one
//! blocking-pool round trip as a spawn is, and a buffered element is one
//! store and one load. A heavy extern is work its author declared worth a
//! blocking-pool round trip, so its least weight is that round trip.

use acvus_mir::analysis::cost::CostTable;

pub const INTERPRETER_COSTS: CostTable = CostTable {
    arithmetic: 1,
    compare: 1,
    load: 3,
    store: 3,
    allocation: 21,
    local_call: 1,
    extern_call: 1,
    heavy: 18_274,
    spawn: 18_274,
    merge: 1,
    chunk_dispatch: 18_274,
    buffered_element: 6,
    k: SPLIT_MARGIN,
};

/// A loop splits only when its work in place is 32 times the cost of one
/// chunk run apart, so a split that finds no second worker costs at most
/// 1/32 of the loop, 3.1 %. The benches' README puts the least difference
/// between two builds a sweep can certify at 7 %, so a wrong split costs
/// less than half of what the benches can see. The margin also covers the
/// table itself, measured on one machine: the spawn row alone ranged from
/// 7.7 to 14.3 µs across three reps.
const SPLIT_MARGIN: u64 = 32;
