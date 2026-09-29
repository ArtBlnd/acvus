//! A cursor: traversal and editing through one object. It stays valid
//! across every insert and removal made through it, so no pass collects
//! positions to edit later.

use super::{Editor, InstRef, Scope};
use crate::ir::cfg::BlockId;

/// The editor keeps the cursor's place, and every edit moves that place
/// with the instructions it shifts: an instruction put in before the place
/// is not yielded, one put in at or after it is, and a removed one is not.
/// An instruction is yielded at most once, wherever an edit moves it.
pub struct Cursor<'e, 'm, S>
where
    S: Scope,
{
    editor: &'e mut Editor<'m, S>,
}

impl<'e, 'm, S> Cursor<'e, 'm, S>
where
    S: Scope,
{
    pub(super) fn new(editor: &'e mut Editor<'m, S>) -> Self {
        Cursor { editor }
    }

    /// The next instruction in block order, or `None` at the body's end.
    pub fn next(&mut self) -> Option<InstRef> {
        loop {
            let walk = self.editor.walk;
            let body = self.editor.body_ref();
            if walk.block.index() >= body.block_count() {
                return None;
            }
            let next = match walk.after {
                Some(after) => body.next_in_block(after),
                None => body.first_inst(walk.block),
            };
            let Some(id) = next else {
                self.editor.walk.block = BlockId::of_index(walk.block.index() + 1);
                self.editor.walk.after = None;
                continue;
            };
            self.editor.walk.after = Some(id);
            if self.editor.yielded.insert(id) {
                return Some(InstRef { id });
            }
        }
    }

    /// The editor, for an edit at or around the cursor's instruction.
    pub fn editor(&mut self) -> &mut Editor<'m, S> {
        self.editor
    }
}
