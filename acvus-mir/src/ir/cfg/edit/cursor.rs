//! A cursor: traversal and editing through one object. It stays valid
//! across every insert and removal made through it, so no pass collects
//! positions to edit later.

use super::{Editor, InstRef, Scope};

pub struct Cursor<'e, 'm, S>
where
    S: Scope,
{
    editor: &'e mut Editor<'m, S>,
    at: Option<InstRef>,
}

impl<'e, 'm, S> Cursor<'e, 'm, S>
where
    S: Scope,
{
    /// The next instruction in block order, or `None` at the body's end.
    pub fn next(&mut self) -> Option<InstRef> {
        todo!("ir::cfg::edit::Cursor::next")
    }

    /// The editor, for an edit at or around the cursor's instruction.
    pub fn editor(&mut self) -> &mut Editor<'m, S> {
        self.editor
    }
}
