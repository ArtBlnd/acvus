//! What an edit changed, for the analysis manager's invalidation
//! (passes-structure §2).

/// One aspect of a body an analysis may read and an edit may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Aspect {
    CfgShape,
    Instructions,
    Types,
    Storage,
    OrderTokens,
}

/// The aspects a sequence of edits changed.
#[derive(Debug, Clone, Default)]
pub struct Changes {
    changed: Vec<Aspect>,
}

impl Changes {
    pub fn record(&mut self, aspect: Aspect) {
        if !self.changed.contains(&aspect) {
            self.changed.push(aspect);
        }
    }
    pub fn touched(&self, aspect: Aspect) -> bool {
        self.changed.contains(&aspect)
    }

    /// Every aspect recorded, each once. Cost: O(aspects).
    pub fn aspects(&self) -> impl Iterator<Item = Aspect> + '_ {
        self.changed.iter().copied()
    }
}
