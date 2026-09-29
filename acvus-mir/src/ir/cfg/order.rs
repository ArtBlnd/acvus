//! Where instructions stand. Each block holds its instructions as a doubly
//! linked list over the body's arena of instructions, indexed by `InstId`,
//! so an insertion, a removal and a move cost O(1) and no instruction's
//! name changes when another moves. Which of two instructions of one block
//! comes first is answered by order numbers that rise along the block, as
//! LLVM's instruction ordering is.
//!
//! An instruction put between two neighbours takes the number halfway
//! between theirs; one put at a block's end takes its predecessor's plus a
//! spacing of 2^32. Where two neighbours leave no number between them, the
//! block is numbered afresh at that spacing, O(the block's instructions).
//! So a renumbering follows at least 32 insertions into one gap, and an
//! insertion costs O(1) amortized, and O(the block's instructions / 32) in
//! the worst case where every insertion lands in the same gap.

use super::body::{BlockId, Body, Def, Inst, InstId, InstNode, Site};

/// The distance between the order numbers of neighbours after a
/// renumbering, and between a block's last instruction and one appended.
const SPACING: u64 = 1 << 32;

/// Where an instruction is put: before another, or at a block's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum At {
    Before(InstId),
    End(BlockId),
}

impl Body {
    pub(super) fn node(&self, id: InstId) -> &InstNode {
        self.insts
            .get(id.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{id:?} is not an instruction of this body"))
    }

    fn node_mut(&mut self, id: InstId) -> &mut InstNode {
        self.insts
            .get_mut(id.index())
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("{id:?} is not an instruction of this body"))
    }

    /// The instruction `id` names. Cost: O(1).
    ///
    /// # Panics
    /// `id` names no instruction of this body.
    pub fn inst(&self, id: InstId) -> &Inst {
        &self.node(id).inst
    }

    pub(super) fn inst_mut(&mut self, id: InstId) -> &mut Inst {
        &mut self.node_mut(id).inst
    }

    /// Whether `id` names an instruction of this body now. Cost: O(1).
    pub fn contains_inst(&self, id: InstId) -> bool {
        self.insts.get(id.index()).is_some_and(Option::is_some)
    }

    /// The block `id` stands in. Cost: O(1).
    ///
    /// # Panics
    /// `id` names no instruction of this body.
    pub fn block_of(&self, id: InstId) -> BlockId {
        self.node(id).block
    }

    /// The instruction after `id` in its block. Cost: O(1).
    pub fn next_in_block(&self, id: InstId) -> Option<InstId> {
        self.node(id).next
    }

    /// The instruction before `id` in its block. Cost: O(1).
    pub fn prev_in_block(&self, id: InstId) -> Option<InstId> {
        self.node(id).prev
    }

    /// The first instruction of `block`. Cost: O(1).
    pub fn first_inst(&self, block: BlockId) -> Option<InstId> {
        self.blocks[block.index()].list.head
    }

    /// The last instruction of `block`. Cost: O(1).
    pub fn last_inst(&self, block: BlockId) -> Option<InstId> {
        self.blocks[block.index()].list.tail
    }

    /// How many instructions `block` holds. Cost: O(1).
    pub fn inst_count(&self, block: BlockId) -> usize {
        self.blocks[block.index()].list.len
    }

    /// `id`'s order number, which rises along its block. Cost: O(1).
    pub(super) fn order_number(&self, id: InstId) -> u64 {
        self.node(id).order
    }

    /// Whether `a` stands before `b` in their one block. Cost: O(1).
    ///
    /// # Panics
    /// The two stand in different blocks.
    pub fn comes_before(&self, a: InstId, b: InstId) -> bool {
        let (a, b) = (self.node(a), self.node(b));
        assert_eq!(a.block, b.block, "instructions are ordered within one block");
        a.order < b.order
    }

    /// How `a` and `b` stand in block order: by their blocks, and within
    /// one block by place, an instruction before the terminator. Cost: O(1).
    pub fn site_order(&self, a: Site, b: Site) -> std::cmp::Ordering {
        let key = |site: Site| match site {
            Site::Inst(inst) => (self.block_of(inst), self.order_number(inst)),
            Site::Term(block) => (block, u64::MAX),
        };
        key(a).cmp(&key(b))
    }

    /// The place of `id` among its block's instructions, counted from 0.
    /// Cost: O(the instructions before it), for display and tests.
    pub fn index_in_block(&self, id: InstId) -> usize {
        std::iter::successors(self.prev_in_block(id), |at| self.prev_in_block(*at)).count()
    }

    /// The instruction at place `index` of `block`, counted from 0. Cost:
    /// O(index), for tests.
    pub fn inst_at(&self, block: BlockId, index: usize) -> Option<InstId> {
        std::iter::successors(self.first_inst(block), |at| self.next_in_block(*at)).nth(index)
    }

    /// Records `inst` as defining its values and using its operands at its
    /// own site. Cost: O(its operands).
    ///
    /// # Panics
    /// A value it defines or reads is not a value of this body.
    pub(super) fn record_inst(&mut self, inst: &Inst) {
        for value in inst.defs() {
            self.value_def_mut(value).def = Def::Inst(inst.id());
        }
        let site = Site::Inst(inst.id());
        for value in inst.operands() {
            self.use_sites_mut(value).add(site, 1);
        }
    }

    /// Puts `inst` at `at` and records its definitions and uses. Cost:
    /// O(its operands), and the order numbers' amortized O(1).
    pub(super) fn link(&mut self, at: At, inst: Inst) {
        self.record_inst(&inst);
        self.place(at, inst);
    }

    /// Puts `inst`, whose definitions and uses are recorded already, at
    /// `at`. Cost: amortized O(1).
    pub(super) fn place(&mut self, at: At, inst: Inst) {
        let id = inst.id();
        if self.insts.len() <= id.index() {
            self.insts.resize_with(id.index() + 1, || None);
        }
        assert!(self.insts[id.index()].is_none(), "{id:?} is placed once");
        let block = self.block_at(at);
        self.insts[id.index()] = Some(InstNode { inst, block, prev: None, next: None, order: 0 });
        self.splice(id, at);
    }

    /// Takes `id` out of its block and the arena, with its uses; the values
    /// it defined keep a stale definition, which the caller replaces or
    /// drops. Cost: O(its operands).
    pub(super) fn unlink(&mut self, id: InstId) -> Inst {
        self.detach(id);
        let node = self.insts[id.index()].take().expect("a detached instruction is in the arena");
        let site = Site::Inst(id);
        for value in node.inst.operands() {
            self.use_sites_mut(value).remove_one(site);
        }
        node.inst
    }

    /// Moves `id` to `at`, with its definitions and uses, which name it and
    /// not its place. Cost: amortized O(1).
    ///
    /// # Panics
    /// `at` is before `id` itself.
    pub(super) fn relink(&mut self, id: InstId, at: At) {
        assert!(at != At::Before(id), "{id:?} is not moved before itself");
        self.detach(id);
        let block = self.block_at(at);
        self.node_mut(id).block = block;
        self.splice(id, at);
    }

    fn block_at(&self, at: At) -> BlockId {
        match at {
            At::Before(next) => self.node(next).block,
            At::End(block) => block,
        }
    }

    /// Unlinks `id` from its block's list; it stays in the arena.
    fn detach(&mut self, id: InstId) {
        let InstNode { block, prev, next, .. } = *self.node(id);
        match prev {
            Some(prev) => self.node_mut(prev).next = next,
            None => self.blocks[block.index()].list.head = next,
        }
        match next {
            Some(next) => self.node_mut(next).prev = prev,
            None => self.blocks[block.index()].list.tail = prev,
        }
        self.blocks[block.index()].list.len -= 1;
        let node = self.node_mut(id);
        node.prev = None;
        node.next = None;
    }

    /// Links `id`, in the arena and detached, at `at` in its node's block,
    /// and gives it an order number.
    fn splice(&mut self, id: InstId, at: At) {
        let block = self.node(id).block;
        let (prev, next) = match at {
            At::Before(next) => (self.node(next).prev, Some(next)),
            At::End(_) => (self.blocks[block.index()].list.tail, None),
        };
        {
            let node = self.node_mut(id);
            node.prev = prev;
            node.next = next;
        }
        match prev {
            Some(prev) => self.node_mut(prev).next = Some(id),
            None => self.blocks[block.index()].list.head = Some(id),
        }
        match next {
            Some(next) => self.node_mut(next).prev = Some(id),
            None => self.blocks[block.index()].list.tail = Some(id),
        }
        self.blocks[block.index()].list.len += 1;

        let low = prev.map_or(0, |prev| self.node(prev).order);
        let order = match next.map(|next| self.node(next).order) {
            Some(high) if high - low >= 2 => Some(low + (high - low) / 2),
            Some(_) => None,
            None => low.checked_add(SPACING),
        };
        match order {
            Some(order) => self.node_mut(id).order = order,
            None => self.renumber(block),
        }
    }

    /// Numbers `block`'s instructions afresh, SPACING apart. Cost: O(its
    /// instructions).
    fn renumber(&mut self, block: BlockId) {
        let mut at = self.blocks[block.index()].list.head;
        let mut order = 0;
        while let Some(id) = at {
            order += SPACING;
            let node = self.node_mut(id);
            node.order = order;
            at = node.next;
        }
    }

    /// Every instruction of `block`, in order, by identity. Cost: O(its
    /// instructions).
    pub(super) fn inst_ids(&self, block: BlockId) -> Vec<InstId> {
        std::iter::successors(self.first_inst(block), |at| self.next_in_block(*at)).collect()
    }

    /// Moves every instruction of `from`, in order, to the end of `to`.
    /// Cost: O(the instructions moved).
    pub(super) fn append_all(&mut self, from: BlockId, to: BlockId) {
        for id in self.inst_ids(from) {
            self.relink(id, At::End(to));
        }
    }

    /// Renames the block every instruction stands in, through `rename`,
    /// after blocks were removed and renumbered. Cost: O(the arena).
    pub(super) fn rename_inst_blocks(&mut self, rename: impl Fn(BlockId) -> BlockId) {
        for node in self.insts.iter_mut().flatten() {
            node.block = rename(node.block);
        }
    }
}
