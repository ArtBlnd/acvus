/// The operations that hold no successor — `code.rs`'s terminators, the node
/// that ends a region's part, and the calls and waited storage accesses that
/// leave through the driver.
/// A return ends one of these; every other `Op::run` ends in the tail call to
/// its successor, a `jmp` natively and a `return_call_indirect` on `wasm32`.
/// `control::ForAhead` is `ForAt` of a loop lowered ahead (RFC-0103): it
/// returns the block to enter as `ForAt` does, and the prefix chain it runs
/// is a part it calls, not a successor it could jump to.
pub const NO_SUCCESSOR: &[&str] = &[
    "control::Goto",
    "control::JumpIf",
    "control::ForAt",
    "control::ForAhead",
    "switch::Switch",
    "switch::SwitchOption",
    "switch::SwitchWord",
    "string::SwitchStr",
    "run::SwitchRun",
    "control::Return",
    "control::Diverge",
    "control::Poison",
    "control::Yield",
    "control::Fall",
    "control::Break",
    "control::Continue",
    "call::CallExternAsync",
    "call::CallStateAsync",
    "call::CallHeavy",
    "call::CallDirectAsync",
    "call::CallIndirectAsync",
    "call::Eval",
    "storage::FetchWaited",
    "storage::CommitWaited",
];
