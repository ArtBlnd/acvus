use std::collections::HashMap;

pub type InstIdx = usize;

#[derive(Debug, Clone)]
pub struct HintTable {
    pub hints: HashMap<InstIdx, Vec<Hint>>,
}

impl HintTable {
    pub fn new() -> Self {
        Self {
            hints: HashMap::new(),
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
