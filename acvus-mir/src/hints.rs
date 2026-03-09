use rustc_hash::FxHashMap;

pub type InstIdx = usize;

#[derive(Debug, Clone)]
pub struct HintTable {
    pub hints: FxHashMap<InstIdx, Vec<Hint>>,
}

impl Default for HintTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HintTable {
    pub fn new() -> Self {
        Self {
            hints: FxHashMap::default(),
        }
    }

    pub fn add(&mut self, idx: InstIdx, hint: Hint) {
        self.hints.entry(idx).or_default().push(hint);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Hint {
    Pure,
    Effectful,
    Batchable,
    AutoParallelizable,
    KnownIterCount(usize),
    Unused,
}
