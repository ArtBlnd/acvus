use super::{Block, DisplaySpec, LlmSpec};

/// A top-level container. Not an Item itself — contains Items.
/// Order matters: items defined above can be referenced by items below.
pub struct Namespace {
    pub name: String,
    pub items: Vec<Item>,
}

/// A definition within a namespace.
pub enum Item {
    Block(Block),
    Llm(LlmSpec),
    Display(DisplaySpec),
}
