use rustc_hash::FxHashMap;
use serde::Serialize;
use tsify::Tsify;

use crate::schema::JsConcreteValue;

/// Read-only snapshot of storage state at a specific entry.
#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct StorageView {
    /// UUID of the entry this view was taken from.
    pub cursor: String,
    /// Depth of the entry.
    pub depth: usize,
    /// All visible key-value pairs (accumulated + turn_diff merged).
    pub entries: FxHashMap<String, JsConcreteValue>,
}
