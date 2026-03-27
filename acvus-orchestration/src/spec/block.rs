/// Execution mode for a block.
pub enum BlockMode {
    Script,
    Template,
}

/// A named acvus source block — referenced by other specs.
pub struct Block {
    pub name: String,
    pub source: String,
    pub mode: BlockMode,
}
