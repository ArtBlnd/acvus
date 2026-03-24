//! SSA construction for context variables.
//!
//! Cranelift-style SSABuilder: tracks define/use of context variables across
//! basic blocks, automatically inserts PHI (block params) at merge points.
//!
//! Usage:
//! 1. Create SSABuilder
//! 2. As lowerer emits code:
//!    - `define(block, ctx, val)` when writing a context
//!    - `use_var(block, ctx)` when reading a context → returns ValueId
//! 3. When all predecessors of a block are known, `seal_block(block)`
//! 4. `finish()` returns the PHI insertions to apply

use crate::graph::ContextId;
use crate::ir::{Label, ValueId};
use rustc_hash::{FxHashMap, FxHashSet};

/// A pending PHI that needs to be resolved when the block is sealed.
#[derive(Debug, Clone)]
struct PendingPhi {
    context: ContextId,
    /// The ValueId allocated for this PHI result (block param).
    result: ValueId,
}

/// Tracks the SSA state for context variables.
#[derive(Debug)]
pub struct SSABuilder {
    /// Current definition of each context variable per block.
    /// (block, context) → ValueId
    current_defs: FxHashMap<(Label, ContextId), ValueId>,

    /// Predecessors of each block.
    predecessors: FxHashMap<Label, Vec<Label>>,

    /// Blocks that have been sealed (all predecessors known).
    sealed: FxHashSet<Label>,

    /// Pending PHIs for unsealed blocks.
    /// When a use_var is requested in an unsealed block and no local def exists,
    /// a PHI is tentatively placed and resolved when the block is sealed.
    pending_phis: FxHashMap<Label, Vec<PendingPhi>>,

    /// Completed PHI insertions: block → list of (context, phi_result, incoming values).
    /// The lowerer uses this to patch block params and jump args.
    phi_results: Vec<PhiInsertion>,
}

/// A completed PHI insertion.
#[derive(Debug, Clone)]
pub struct PhiInsertion {
    /// The merge block where this PHI lives.
    pub block: Label,
    /// Which context variable this PHI is for.
    pub context: ContextId,
    /// The ValueId that represents the PHI result (block param).
    pub result: ValueId,
    /// Incoming values from predecessors: (predecessor_block, ValueId).
    pub incoming: Vec<(Label, ValueId)>,
}

/// Entry block label constant — used by Lowerer to identify the implicit entry block.
pub const ENTRY_BLOCK: Label = Label(u32::MAX);

impl Default for SSABuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SSABuilder {
    pub fn new() -> Self {
        Self {
            current_defs: FxHashMap::default(),
            predecessors: FxHashMap::default(),
            sealed: FxHashSet::default(),
            pending_phis: FxHashMap::default(),
            phi_results: Vec::new(),
        }
    }

    /// Declare a predecessor relationship: `from` jumps to `to`.
    pub fn add_predecessor(&mut self, to: Label, from: Label) {
        self.predecessors.entry(to).or_default().push(from);
    }

    /// Define a context variable in a block.
    pub fn define(&mut self, block: Label, context: ContextId, value: ValueId) {
        self.current_defs.insert((block, context), value);
    }

    /// Use a context variable in a block. Returns the ValueId holding the value.
    ///
    /// If the variable was defined in this block, returns that definition.
    /// If not, looks up predecessors (recursively). If the block is sealed
    /// and has multiple predecessors with different definitions, a PHI is inserted.
    /// If the block is not yet sealed, a pending PHI is created.
    pub fn use_var(
        &mut self,
        block: Label,
        context: ContextId,
        alloc_val: &mut impl FnMut() -> ValueId,
    ) -> ValueId {
        // Local definition in this block?
        if let Some(&val) = self.current_defs.get(&(block, context)) {
            return val;
        }

        // Block sealed?
        if self.sealed.contains(&block) {
            return self.use_var_sealed(block, context, alloc_val);
        }

        // Not sealed — place a pending PHI.
        let phi_val = alloc_val();
        self.pending_phis
            .entry(block)
            .or_default()
            .push(PendingPhi {
                context,
                result: phi_val,
            });
        // Tentatively define this as the current value.
        self.current_defs.insert((block, context), phi_val);
        phi_val
    }

    /// Seal a block: all predecessors are now known.
    /// Resolves any pending PHIs by looking up predecessor definitions.
    pub fn seal_block(&mut self, block: Label, alloc_val: &mut impl FnMut() -> ValueId) {
        assert!(
            !self.sealed.contains(&block),
            "block {:?} already sealed",
            block
        );
        self.sealed.insert(block);

        // Resolve pending PHIs.
        let pending = self.pending_phis.remove(&block).unwrap_or_default();
        for phi in pending {
            self.resolve_phi(block, phi.context, phi.result, alloc_val);
        }
    }

    /// Finish SSA construction. Returns all PHI insertions.
    pub fn finish(self) -> Vec<PhiInsertion> {
        self.phi_results
    }

    // ── Internal ────────────────────────────────────────────────────

    fn use_var_sealed(
        &mut self,
        block: Label,
        context: ContextId,
        alloc_val: &mut impl FnMut() -> ValueId,
    ) -> ValueId {
        let preds = self.predecessors.get(&block).cloned().unwrap_or_default();

        if preds.is_empty() {
            panic!(
                "use_var: context {:?} has no definition and no predecessors in block {:?}",
                context, block
            );
        }

        if preds.len() == 1 {
            // Single predecessor — just look up recursively.
            let val = self.use_var(preds[0], context, alloc_val);
            self.current_defs.insert((block, context), val);
            return val;
        }

        // Multiple predecessors — need PHI.
        let phi_val = alloc_val();
        // Define before resolving to break cycles (loop back edges).
        self.current_defs.insert((block, context), phi_val);
        self.resolve_phi(block, context, phi_val, alloc_val);
        // resolve_phi may have replaced with a trivial value.
        *self.current_defs.get(&(block, context)).unwrap()
    }

    fn resolve_phi(
        &mut self,
        block: Label,
        context: ContextId,
        phi_val: ValueId,
        alloc_val: &mut impl FnMut() -> ValueId,
    ) {
        let preds = self.predecessors.get(&block).cloned().unwrap_or_default();
        let mut incoming = Vec::new();

        for pred in &preds {
            let val = self.use_var(*pred, context, alloc_val);
            incoming.push((*pred, val));
        }

        // Trivial PHI elimination: if all incoming values are the same (or the phi itself),
        // replace with that value.
        let unique: FxHashSet<ValueId> = incoming
            .iter()
            .map(|(_, v)| *v)
            .filter(|&v| v != phi_val)
            .collect();

        if unique.len() == 1 {
            // Trivial — all predecessors provide the same value.
            let single = *unique.iter().next().unwrap();
            self.current_defs.insert((block, context), single);
            // No PHI insertion needed.
        } else {
            // Non-trivial PHI.
            self.current_defs.insert((block, context), phi_val);
            self.phi_results.push(PhiInsertion {
                block,
                context,
                result: phi_val,
                incoming,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(n: u32) -> Label {
        Label(n)
    }

    fn make_val_alloc() -> impl FnMut() -> ValueId {
        use acvus_utils::LocalIdOps;
        let mut counter = 0usize;
        move || {
            let id = ValueId::from_raw(counter);
            counter += 1;
            id
        }
    }

    // ── Completeness: correct PHI insertion ──

    /// Single block, define then use — no PHI needed.
    #[test]
    fn single_block_no_phi() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        let v0 = alloc();
        ssa.define(label(0), ctx, v0);
        let result = ssa.use_var(label(0), ctx, &mut alloc);

        assert_eq!(result, v0);
        assert!(ssa.finish().is_empty(), "no PHI needed");
    }

    /// Linear blocks: define in block 0, use in block 1 — no PHI.
    #[test]
    fn linear_blocks_no_phi() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        let v0 = alloc();
        ssa.define(label(0), ctx, v0);
        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);

        let result = ssa.use_var(label(1), ctx, &mut alloc);
        assert_eq!(result, v0, "should propagate from predecessor");
        assert!(ssa.finish().is_empty(), "single predecessor, no PHI");
    }

    /// Diamond: block 0 → block 1 (write), block 0 → block 2 (no write), merge at block 3.
    #[test]
    fn diamond_phi_inserted() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        // block 0: initial definition
        let v_init = alloc();
        ssa.define(label(0), ctx, v_init);

        // block 1: write
        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);
        let v_write = alloc();
        ssa.define(label(1), ctx, v_write);

        // block 2: no write (uses v_init from block 0)
        ssa.add_predecessor(label(2), label(0));
        ssa.seal_block(label(2), &mut alloc);

        // block 3: merge — two predecessors with different definitions
        ssa.add_predecessor(label(3), label(1));
        ssa.add_predecessor(label(3), label(2));
        ssa.seal_block(label(3), &mut alloc);

        let result = ssa.use_var(label(3), ctx, &mut alloc);
        assert_ne!(result, v_init, "should not be the initial value");
        assert_ne!(result, v_write, "should not be the written value");

        let phis = ssa.finish();
        assert_eq!(phis.len(), 1, "one PHI for the merge");
        assert_eq!(phis[0].block, label(3));
        assert_eq!(phis[0].context, ctx);
        assert_eq!(phis[0].incoming.len(), 2);
    }

    /// Diamond where both sides write — PHI with two different values.
    #[test]
    fn diamond_both_write_phi() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        let v_init = alloc();
        ssa.define(label(0), ctx, v_init);

        // block 1: write v1
        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);
        let v1 = alloc();
        ssa.define(label(1), ctx, v1);

        // block 2: write v2
        ssa.add_predecessor(label(2), label(0));
        ssa.seal_block(label(2), &mut alloc);
        let v2 = alloc();
        ssa.define(label(2), ctx, v2);

        // merge
        ssa.add_predecessor(label(3), label(1));
        ssa.add_predecessor(label(3), label(2));
        ssa.seal_block(label(3), &mut alloc);

        let _result = ssa.use_var(label(3), ctx, &mut alloc);
        let phis = ssa.finish();
        assert_eq!(phis.len(), 1);

        // Incoming should be v1 from block 1, v2 from block 2
        let incoming_vals: FxHashSet<ValueId> = phis[0].incoming.iter().map(|(_, v)| *v).collect();
        assert!(incoming_vals.contains(&v1));
        assert!(incoming_vals.contains(&v2));
    }

    /// Nested diamond: outer branch, inner branch in one arm.
    #[test]
    fn nested_diamond() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        // block 0: init
        let v0 = alloc();
        ssa.define(label(0), ctx, v0);

        // block 1: outer then → inner branch
        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);

        // block 2: inner then — write
        ssa.add_predecessor(label(2), label(1));
        ssa.seal_block(label(2), &mut alloc);
        let v_inner = alloc();
        ssa.define(label(2), ctx, v_inner);

        // block 3: inner else — no write
        ssa.add_predecessor(label(3), label(1));
        ssa.seal_block(label(3), &mut alloc);

        // block 4: inner merge
        ssa.add_predecessor(label(4), label(2));
        ssa.add_predecessor(label(4), label(3));
        ssa.seal_block(label(4), &mut alloc);

        // block 5: outer else — write
        ssa.add_predecessor(label(5), label(0));
        ssa.seal_block(label(5), &mut alloc);
        let v_outer = alloc();
        ssa.define(label(5), ctx, v_outer);

        // block 6: outer merge
        ssa.add_predecessor(label(6), label(4));
        ssa.add_predecessor(label(6), label(5));
        ssa.seal_block(label(6), &mut alloc);

        let result = ssa.use_var(label(6), ctx, &mut alloc);
        let phis = ssa.finish();

        // Should have PHIs at inner merge (block 4) and outer merge (block 6)
        assert!(
            phis.len() >= 2,
            "need PHIs at both merge points, got {}",
            phis.len()
        );

        let phi_blocks: FxHashSet<Label> = phis.iter().map(|p| p.block).collect();
        assert!(
            phi_blocks.contains(&label(4)),
            "inner merge should have PHI"
        );
        assert!(
            phi_blocks.contains(&label(6)),
            "outer merge should have PHI"
        );

        // result should be the outer merge PHI
        let outer_phi = phis.iter().find(|p| p.block == label(6)).unwrap();
        assert_eq!(result, outer_phi.result);
    }

    /// Loop: block 0 → block 1 (loop header) → block 2 (body, writes) → back to block 1.
    #[test]
    fn loop_back_edge_phi() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        // block 0: init
        let v0 = alloc();
        ssa.define(label(0), ctx, v0);

        // block 1: loop header — NOT sealed yet (back edge pending)
        ssa.add_predecessor(label(1), label(0));

        // Use in unsealed block → pending PHI
        let v_header = ssa.use_var(label(1), ctx, &mut alloc);

        // block 2: loop body — writes
        ssa.add_predecessor(label(2), label(1));
        ssa.seal_block(label(2), &mut alloc);
        let v_body = alloc();
        ssa.define(label(2), ctx, v_body);

        // Back edge: block 2 → block 1
        ssa.add_predecessor(label(1), label(2));
        // NOW seal block 1 — both predecessors known
        ssa.seal_block(label(1), &mut alloc);

        let phis = ssa.finish();
        // Loop header should have a PHI (v0 from entry, v_body from back edge)
        assert!(!phis.is_empty(), "loop header should have PHI");
        let header_phi = phis.iter().find(|p| p.block == label(1)).unwrap();
        let incoming_vals: FxHashSet<ValueId> =
            header_phi.incoming.iter().map(|(_, v)| *v).collect();
        assert!(
            incoming_vals.contains(&v0),
            "should have initial value from entry"
        );
        assert!(
            incoming_vals.contains(&v_body),
            "should have loop body value"
        );
    }

    // ── Soundness: trivial PHI elimination ──

    /// Diamond where neither side writes — no PHI needed (both use same value).
    #[test]
    fn diamond_no_write_no_phi() {
        let ctx = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        let v0 = alloc();
        ssa.define(label(0), ctx, v0);

        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);

        ssa.add_predecessor(label(2), label(0));
        ssa.seal_block(label(2), &mut alloc);

        ssa.add_predecessor(label(3), label(1));
        ssa.add_predecessor(label(3), label(2));
        ssa.seal_block(label(3), &mut alloc);

        let result = ssa.use_var(label(3), ctx, &mut alloc);
        assert_eq!(
            result, v0,
            "should be the same value — trivial PHI eliminated"
        );
        assert!(ssa.finish().is_empty(), "trivial PHI should be eliminated");
    }

    /// Multiple contexts — independent PHIs.
    #[test]
    fn multiple_contexts_independent() {
        let ctx_a = ContextId::alloc();
        let ctx_b = ContextId::alloc();
        let mut ssa = SSABuilder::new();
        let mut alloc = make_val_alloc();

        let va0 = alloc();
        let vb0 = alloc();
        ssa.define(label(0), ctx_a, va0);
        ssa.define(label(0), ctx_b, vb0);

        // block 1: write ctx_a only
        ssa.add_predecessor(label(1), label(0));
        ssa.seal_block(label(1), &mut alloc);
        let va1 = alloc();
        ssa.define(label(1), ctx_a, va1);

        // block 2: no writes
        ssa.add_predecessor(label(2), label(0));
        ssa.seal_block(label(2), &mut alloc);

        // merge
        ssa.add_predecessor(label(3), label(1));
        ssa.add_predecessor(label(3), label(2));
        ssa.seal_block(label(3), &mut alloc);

        let _ra = ssa.use_var(label(3), ctx_a, &mut alloc);
        let rb = ssa.use_var(label(3), ctx_b, &mut alloc);

        let phis = ssa.finish();
        // ctx_a should have PHI (different values from two sides)
        // ctx_b should NOT have PHI (same value from both sides — trivial)
        let phi_contexts: Vec<ContextId> = phis.iter().map(|p| p.context).collect();
        assert!(phi_contexts.contains(&ctx_a), "ctx_a needs PHI");
        assert!(
            !phi_contexts.contains(&ctx_b),
            "ctx_b should be trivially eliminated"
        );
        assert_eq!(rb, vb0, "ctx_b should resolve to original value");
    }
}
