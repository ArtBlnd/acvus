//! Move checking pass.
//!
//! Effectful values (`Iterator<T, Effectful>`, `Sequence<T, O, Effectful>`,
//! `Opaque`) are move-only: once consumed, they cannot be used again.
//! This pass performs forward dataflow analysis over the CFG to detect
//! use-after-move violations.
//!
//! Design:
//! - `Ty::Error` → skip (analysis mode).
//! - `Fn` with move-only captures → FnOnce (transitive).
//! - Join at merge points: `Alive ⊔ Moved = Moved` (conservative).
//! - $variables: tracked by name. `Store` (via Ref) revives, `Load` of move-only consumes.

use std::collections::VecDeque;

use acvus_ast::Span;
use acvus_utils::LocalIdOps;
use rustc_hash::FxHashMap;

use crate::cfg::{BlockIdx, Terminator, promote};
use crate::ir::{Callee, Inst, InstKind, MirBody, MirModule, ValueId};
use crate::ty::Ty;

use super::type_check::{ValidationError, ValidationErrorKind};

// ---------------------------------------------------------------------------
// is_move_only
// ---------------------------------------------------------------------------

/// Determine whether a type requires move semantics.
///
/// Returns `Some(true)` for move-only, `Some(false)` for copyable,
/// `None` for unknown (skip — analysis mode).
pub fn is_move_only(ty: &Ty) -> Option<bool> {
    match ty {
        // Primitives — always Copy
        Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => {
            Some(false)
        }

        // Handle — always move-only (deferred computation, must be consumed exactly once)
        Ty::Handle(..) => Some(true),

        // UserDefined — always move-only
        Ty::UserDefined { .. } => Some(true),

        // Containers — transitive
        Ty::List(inner) | Ty::Deque(inner, _) | Ty::Option(inner) => is_move_only(inner),
        Ty::Tuple(elems) => {
            let mut any_move = false;
            for e in elems {
                match is_move_only(e) {
                    Some(true) => any_move = true,
                    None => return None,
                    Some(false) => {}
                }
            }
            Some(any_move)
        }
        Ty::Object(fields) => {
            let mut any_move = false;
            for v in fields.values() {
                match is_move_only(v) {
                    Some(true) => any_move = true,
                    None => return None,
                    Some(false) => {}
                }
            }
            Some(any_move)
        }
        Ty::Enum { variants, .. } => {
            let mut any_move = false;
            for payload in variants.values().flatten() {
                match is_move_only(payload) {
                    Some(true) => any_move = true,
                    None => return None,
                    Some(false) => {}
                }
            }
            Some(any_move)
        }

        // Fn — move-only if any capture is move-only (FnOnce)
        Ty::Fn { captures, .. } => {
            let mut any_move = false;
            for c in captures {
                match is_move_only(c) {
                    Some(true) => any_move = true,
                    None => return None,
                    Some(false) => {}
                }
            }
            Some(any_move)
        }

        // Identity — always copyable (just an id)
        Ty::Identity(_) => Some(false),

        // Ref — ephemeral, always immediately consumed. Skip (not subject to move analysis).
        Ty::Ref(..) => None,

        // Unknown — skip
        Ty::Error(_) => None,

        // Post-inference: no type variables remain.
        Ty::Var(v) => match *v {},
    }
}

// ---------------------------------------------------------------------------
// Move state
// ---------------------------------------------------------------------------

/// Liveness of a single value or variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Liveness {
    Alive,
    /// Moved at instruction index `at`.
    Moved {
        at: usize,
    },
}

impl Liveness {
    /// Conservative join: if either side is Moved, result is Moved.
    fn join(self, other: Liveness) -> Liveness {
        match (self, other) {
            (Liveness::Alive, Liveness::Alive) => Liveness::Alive,
            (Liveness::Moved { at }, _) | (_, Liveness::Moved { at }) => Liveness::Moved { at },
        }
    }
}

/// Tracks move state for both ValueIds and $variables.
#[derive(Debug, Clone)]
struct MoveState {
    values: FxHashMap<ValueId, Liveness>,
    /// Variable/param liveness, keyed by storage slot ValueId.
    vars: FxHashMap<ValueId, Liveness>,
}

impl MoveState {
    fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            vars: FxHashMap::default(),
        }
    }

    fn get_value(&self, id: ValueId) -> Option<Liveness> {
        self.values.get(&id).copied()
    }

    fn set_value(&mut self, id: ValueId, liveness: Liveness) {
        self.values.insert(id, liveness);
    }

    fn get_var(&self, slot: ValueId) -> Option<Liveness> {
        self.vars.get(&slot).copied()
    }

    fn set_var(&mut self, slot: ValueId, liveness: Liveness) {
        self.vars.insert(slot, liveness);
    }

    /// Join another state into this one. Returns true if anything changed.
    fn join_from(&mut self, other: &MoveState) -> bool {
        let mut changed = false;
        for (&id, &liveness) in &other.values {
            use std::collections::hash_map::Entry;
            match self.values.entry(id) {
                Entry::Vacant(e) => {
                    e.insert(liveness);
                    changed = true; // New entry = change.
                }
                Entry::Occupied(mut e) => {
                    let joined = e.get().join(liveness);
                    if *e.get() != joined {
                        e.insert(joined);
                        changed = true;
                    }
                }
            }
        }
        for (&name, &liveness) in &other.vars {
            use std::collections::hash_map::Entry;
            match self.vars.entry(name) {
                Entry::Vacant(e) => {
                    e.insert(liveness);
                    changed = true;
                }
                Entry::Occupied(mut e) => {
                    let joined = e.get().join(liveness);
                    if *e.get() != joined {
                        e.insert(joined);
                        changed = true;
                    }
                }
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Check move semantics for the entire module.
pub fn check_moves(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    check_body("main", &module.main, &mut errors);
    for (label, closure) in &module.closures {
        let scope = format!("closure({:?})", label);
        check_body(&scope, closure, &mut errors);
    }
    errors
}

fn check_body(scope: &str, body: &MirBody, errors: &mut Vec<ValidationError>) {
    let cfg = promote(body.clone());
    if cfg.blocks.is_empty() {
        return;
    }

    // Build ref_target map: Ref dst → RefTarget.
    let mut ref_target: FxHashMap<ValueId, crate::ir::RefTarget> = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref { dst, target, .. } = &inst.kind {
                ref_target.insert(*dst, target.clone());
            }
        }
    }

    let n = cfg.blocks.len();
    let mut block_entry: Vec<MoveState> = (0..n).map(|_| MoveState::new()).collect();
    let mut block_exit: Vec<MoveState> = (0..n).map(|_| MoveState::new()).collect();

    let mut worklist = VecDeque::new();
    worklist.push_back(BlockIdx(0));
    let mut visited = vec![false; n];

    while let Some(idx) = worklist.pop_front() {
        visited[idx.0] = true;
        let block = &cfg.blocks[idx.0];
        let mut state = block_entry[idx.0].clone();

        // Process instructions in this block
        for (i, inst) in block.insts.iter().enumerate() {
            process_inst(
                scope,
                i,
                inst,
                &cfg.val_types,
                &ref_target,
                &mut state,
                errors,
            );
        }

        block_exit[idx.0] = state;

        // Propagate to successors
        match &block.terminator {
            Terminator::Jump { label, args } => {
                if let Some(&target_idx) = cfg.label_to_block.get(label) {
                    propagate_args(
                        scope,
                        &block_exit[idx.0],
                        args,
                        &cfg.blocks[target_idx.0].params,
                        &cfg.val_types,
                        errors,
                        &mut block_entry[target_idx.0],
                    );
                    if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::JumpIf {
                cond: _,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                for (label, args) in [(then_label, then_args), (else_label, else_args)] {
                    if let Some(&target_idx) = cfg.label_to_block.get(label) {
                        propagate_args(
                            scope,
                            &block_exit[idx.0],
                            args,
                            &cfg.blocks[target_idx.0].params,
                            &cfg.val_types,
                            errors,
                            &mut block_entry[target_idx.0],
                        );
                        if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            Terminator::ListStep {
                dst,
                index_dst,
                done,
                done_args,
                ..
            } => {
                // ListStep defines dst (element) and index_dst (new index).
                // list is borrowed (not consumed), index_src is Int (copyable).
                block_exit[idx.0].set_value(*dst, Liveness::Alive);
                block_exit[idx.0].set_value(*index_dst, Liveness::Alive);

                // Fallthrough.
                let next = idx.0 + 1;
                if next < n && propagate_state(&block_exit[idx.0], &mut block_entry[next]) {
                    worklist.push_back(BlockIdx(next));
                }
                // Done branch.
                if let Some(&target_idx) = cfg.label_to_block.get(done) {
                    propagate_args(
                        scope,
                        &block_exit[idx.0],
                        done_args,
                        &cfg.blocks[target_idx.0].params,
                        &cfg.val_types,
                        errors,
                        &mut block_entry[target_idx.0],
                    );
                    if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n && propagate_state(&block_exit[idx.0], &mut block_entry[next]) {
                    worklist.push_back(BlockIdx(next));
                }
            }
            Terminator::Return(_) => {}
        }
    }
}

fn propagate_state(source: &MoveState, target: &mut MoveState) -> bool {
    target.join_from(source)
}

fn propagate_args(
    _scope: &str,
    source: &MoveState,
    args: &[ValueId],
    params: &[ValueId],
    val_types: &FxHashMap<ValueId, Ty>,
    _errors: &mut Vec<ValidationError>,
    target_entry: &mut MoveState,
) {
    // Map arg liveness → param liveness.
    // If an arg is move-only and moved, the param inherits Moved.
    for (arg, param) in args.iter().zip(params.iter()) {
        let arg_liveness = source.get_value(*arg).unwrap_or(Liveness::Alive);
        let is_move = val_types.get(param).and_then(is_move_only) == Some(true);
        if is_move {
            let entry = target_entry.values.entry(*param).or_insert(Liveness::Alive);
            let joined = entry.join(arg_liveness);
            *entry = joined;
        } else {
            target_entry.values.entry(*param).or_insert(Liveness::Alive);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-instruction processing
// ---------------------------------------------------------------------------

/// Try to consume (move) a ValueId. If it's move-only and already moved, emit error.
/// Returns true if the value was consumed (is move-only).
fn try_consume_value(
    scope: &str,
    inst_idx: usize,
    span: Span,
    id: ValueId,
    val_types: &FxHashMap<ValueId, Ty>,
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) -> bool {
    let Some(ty) = val_types.get(&id) else {
        return false;
    };
    let move_only = is_move_only(ty);
    let Some(true) = move_only else {
        return false;
    };

    // Check if already moved
    if let Some(Liveness::Moved { at }) = state.get_value(id) {
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span,
            kind: ValidationErrorKind::UseAfterMove {
                value_id: id.to_raw() as u32,
                moved_at: at,
                ty: ty.clone(),
            },
        });
        return true;
    }

    // Mark as moved
    state.set_value(id, Liveness::Moved { at: inst_idx });
    true
}

/// Process a single instruction: check uses and update move state.
fn process_inst(
    scope: &str,
    inst_idx: usize,
    inst: &Inst,
    val_types: &FxHashMap<ValueId, Ty>,
    ref_target: &FxHashMap<ValueId, crate::ir::RefTarget>,
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let span = inst.span;

    match &inst.kind {
        // === No operands / define only ===
        InstKind::Const { dst, .. } | InstKind::Poison { dst } | InstKind::Undef { dst } => {
            state.set_value(*dst, Liveness::Alive);
        }
        // Ref: register target for subsequent Load/Store lookup, define dst as alive.
        InstKind::Ref { dst, target, .. } => {
            // Note: ref_target is pre-built; just mark dst alive.
            let _ = target;
            state.set_value(*dst, Liveness::Alive);
        }
        // Load: src is a Ref (not consumed as move-only — Refs are ephemeral).
        // Define dst. Track variable liveness by name.
        InstKind::Load { dst, src, .. } => {
            // Note: do NOT try_consume_value on src — it's a Ref type which
            // is ephemeral and not subject to move checking.
            // Variable name tracking: if src points to a Var/Param, check var liveness.
            if let Some(target) = ref_target.get(src) {
                let name = match target {
                    crate::ir::RefTarget::Var(n) | crate::ir::RefTarget::Param(n) => Some(*n),
                    crate::ir::RefTarget::Context(_) => None,
                };
                if let Some(name) = name {
                    // Check if variable has been moved
                    if let Some(Liveness::Moved { at }) = state.get_var(name)
                        && let Some(ty) = val_types.get(dst)
                        && is_move_only(ty) == Some(true)
                    {
                        errors.push(ValidationError {
                            scope: scope.to_string(),
                            inst_index: inst_idx,
                            span,
                            kind: ValidationErrorKind::UseAfterMove {
                                value_id: dst.to_raw() as u32,
                                moved_at: at,
                                ty: ty.clone(),
                            },
                        });
                    }
                    // Loading move-only → var is now moved
                    if let Some(ty) = val_types.get(dst)
                        && is_move_only(ty) == Some(true)
                    {
                        state.set_var(name, Liveness::Moved { at: inst_idx });
                    }
                }
            }
            state.set_value(*dst, Liveness::Alive);
        }
        // Store: dst is a Ref (ephemeral, not move-checked). Consume value. Revive variable.
        InstKind::Store { dst, value, .. } => {
            // Note: do NOT try_consume_value on dst — it's a Ref type.
            try_consume_value(scope, inst_idx, span, *value, val_types, state, errors);
            // Variable name tracking: Store revives the variable.
            if let Some(target) = ref_target.get(dst)
                && let crate::ir::RefTarget::Var(name) = target
            {
                state.set_var(*name, Liveness::Alive);
            }
        }
        InstKind::BlockLabel { params, .. } => {
            for p in params {
                state.values.entry(*p).or_insert(Liveness::Alive);
            }
        }
        InstKind::Nop => {}

        // === Consuming operations (move operands) ===
        InstKind::Return(v) => {
            try_consume_value(scope, inst_idx, span, *v, val_types, state, errors);
        }
        InstKind::Cast { dst, src, .. } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::Clone { dst, src, .. } => {
            // Clone reads src (not consumed — it's being cloned)
            // src remains alive after clone
            state.set_value(*dst, Liveness::Alive);
            let _ = src;
        }
        InstKind::Drop { src } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
        }
        InstKind::ListStep { dst, index_dst, .. } => {
            // list is borrowed (not consumed), index_src is Int (copyable).
            state.set_value(*dst, Liveness::Alive);
            state.set_value(*index_dst, Liveness::Alive);
        }

        // Functions
        InstKind::LoadFunction { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        // Calls — all args are consumed; indirect callee is also consumed
        InstKind::FunctionCall {
            dst,
            callee,
            args,
            context_uses,
            context_defs,
        } => {
            if let Callee::Indirect(closure) = callee {
                try_consume_value(scope, inst_idx, span, *closure, val_types, state, errors);
            }
            for arg in args {
                try_consume_value(scope, inst_idx, span, *arg, val_types, state, errors);
            }
            for (_, vid) in context_uses {
                try_consume_value(scope, inst_idx, span, *vid, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
            for (_, vid) in context_defs {
                state.set_value(*vid, Liveness::Alive);
            }
        }

        // Constructors — elements are consumed
        InstKind::MakeDeque { dst, elements } => {
            for e in elements {
                try_consume_value(scope, inst_idx, span, *e, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeObject { dst, fields } => {
            for (_, v) in fields {
                try_consume_value(scope, inst_idx, span, *v, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeTuple { dst, elements } => {
            for e in elements {
                try_consume_value(scope, inst_idx, span, *e, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeRange {
            dst, start, end, ..
        } => {
            // start and end are always Int (Pure) — no move check needed
            state.set_value(*dst, Liveness::Alive);
            let _ = (start, end);
        }
        InstKind::MakeClosure { dst, captures, .. } => {
            for cap in captures {
                try_consume_value(scope, inst_idx, span, *cap, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeVariant { dst, payload, .. } => {
            if let Some(p) = payload {
                try_consume_value(scope, inst_idx, span, *p, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }

        // === Non-consuming operations (borrow operands) ===
        // These read the value but don't take ownership.
        InstKind::FieldGet { dst, object: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::FieldSet {
            dst,
            object: _,
            value,
            ..
        } => {
            try_consume_value(scope, inst_idx, span, *value, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ObjectGet { dst, object: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TupleIndex { dst, tuple: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ListIndex { dst, list: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ListGet {
            dst,
            list: _,
            index: _,
        } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ListSlice { dst, list: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::UnwrapVariant { dst, src: _ } => {
            state.set_value(*dst, Liveness::Alive);
        }

        // Pattern tests — read-only, always produce Bool
        InstKind::TestLiteral { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestListLen { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestObjectKey { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestRange { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestVariant { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }

        // Arithmetic — operands are always pure scalars, no move
        InstKind::BinOp {
            dst,
            left: _,
            right: _,
            ..
        } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::UnaryOp {
            dst, operand: _, ..
        } => {
            state.set_value(*dst, Liveness::Alive);
        }

        // Spawn — consumes args (and indirect callee), defines dst
        InstKind::Spawn {
            dst, callee, args, ..
        } => {
            if let Callee::Indirect(closure) = callee {
                try_consume_value(scope, inst_idx, span, *closure, val_types, state, errors);
            }
            for arg in args {
                try_consume_value(scope, inst_idx, span, *arg, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        // Eval — consumes Handle (move-only), defines dst
        InstKind::Eval { dst, src, .. } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }

        // Control flow — handled at block level
        InstKind::Jump { .. } | InstKind::JumpIf { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::QualifiedRef;
    use crate::ir::{Callee, DebugInfo, Inst, MirBody, MirModule, RefTarget};
    use crate::ty::{Effect, Param};
    use acvus_utils::{Interner, LocalFactory};

    /// Create a dummy Param for tests where parameter name is irrelevant.
    fn param(ty: Ty) -> Param {
        let interner = Interner::new();
        Param::new(interner.intern("_"), ty)
    }

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst { span: span(), kind }
    }

    fn make_module(insts: Vec<Inst>, val_types: FxHashMap<ValueId, Ty>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types,
                params: Vec::new(),
                captures: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 10,
            },
            closures: FxHashMap::default(),
        }
    }

    // -- is_move_only tests --

    fn test_user_defined() -> Ty {
        let i = Interner::new();
        Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestType")),
            type_args: vec![],
            effect_args: vec![],
        }
    }

    #[test]
    fn pure_types_are_copy() {
        assert_eq!(is_move_only(&Ty::Int), Some(false));
        assert_eq!(is_move_only(&Ty::String), Some(false));
        assert_eq!(is_move_only(&Ty::Bool), Some(false));
        assert_eq!(is_move_only(&Ty::List(Box::new(Ty::Int))), Some(false));
    }

    #[test]
    fn user_defined_is_move() {
        assert_eq!(is_move_only(&test_user_defined()), Some(true));
    }

    #[test]
    fn tuple_with_user_defined_is_move() {
        let ty = Ty::Tuple(vec![Ty::Int, test_user_defined()]);
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_user_defined_capture_is_move() {
        let ty = Ty::Fn {
            params: vec![param(Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![test_user_defined()],
            effect: Effect::pure(),
            hint: None,
        };
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_pure_captures_is_copy() {
        let ty = Ty::Fn {
            params: vec![param(Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![Ty::Int, Ty::String],
            effect: Effect::pure(), // effect of the fn doesn't matter, only captures
            hint: None,
        };
        assert_eq!(is_move_only(&ty), Some(false));
    }

    // -- move check integration tests --

    #[test]
    fn error_for_user_defined_reuse() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        // v0 = UserDefined (move-only), used twice → ERROR
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, test_user_defined());
        val_types.insert(v1, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v2, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::FunctionCall {
                    dst: v1,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                inst(InstKind::FunctionCall {
                    dst: v2,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "UserDefined reuse should be rejected");
        assert!(matches!(
            errors[0].kind,
            ValidationErrorKind::UseAfterMove { .. }
        ));
    }

    #[test]
    fn no_error_for_single_use_user_defined() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        // v0 = UserDefined, used once → OK
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, test_user_defined());
        val_types.insert(v1, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![inst(InstKind::FunctionCall {
                dst: v1,
                callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                args: vec![v0],
                context_uses: vec![],
                context_defs: vec![],
            })],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty());
    }

    #[test]
    fn var_reassign_revives() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let r0 = vf.next(); // ref for first Store
        let r1 = vf.next(); // ref for first Load
        let r2 = vf.next(); // ref for second Store
        let r3 = vf.next(); // ref for second Load
        let a = vf.next(); // storage slot for variable "a"
        // $a = move-only (v0), Load (v1) → moved, $a = new value (v2) → alive, Load (v3) → OK
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
        val_types.insert(v3, move_ty.clone());
        val_types.insert(v4, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v5, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                // $a = v0 (move-only)
                inst(InstKind::Ref {
                    dst: r0,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Store {
                    dst: r0,
                    value: v0,
                    volatile: false,
                }),
                // v1 = $a → moves $a
                inst(InstKind::Ref {
                    dst: r1,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Load {
                    dst: v1,
                    src: r1,
                    volatile: false,
                }),
                // use v1
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v1],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                // $a = v2 (new value) → revives $a
                inst(InstKind::Ref {
                    dst: r2,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Store {
                    dst: r2,
                    value: v2,
                    volatile: false,
                }),
                // v3 = $a → OK (new value)
                inst(InstKind::Ref {
                    dst: r3,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Load {
                    dst: v3,
                    src: r3,
                    volatile: false,
                }),
                // use v3
                inst(InstKind::FunctionCall {
                    dst: v5,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v3],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(
            errors.is_empty(),
            "reassigned variable should be alive: {errors:?}"
        );
    }

    #[test]
    fn var_use_after_move() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let r0 = vf.next(); // ref for Store
        let r1 = vf.next(); // ref for first Load
        let r2 = vf.next(); // ref for second Load
        let a = vf.next(); // storage slot for variable "a"
        // $a = move-only, Load → moved, Load again → ERROR
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
        val_types.insert(v3, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v4, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::Ref {
                    dst: r0,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Store {
                    dst: r0,
                    value: v0,
                    volatile: false,
                }),
                inst(InstKind::Ref {
                    dst: r1,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Load {
                    dst: v1,
                    src: r1,
                    volatile: false,
                }),
                inst(InstKind::FunctionCall {
                    dst: v3,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v1],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                // Second load — $a already moved
                inst(InstKind::Ref {
                    dst: r2,
                    target: RefTarget::Var(a),
                    path: vec![],
                }),
                inst(InstKind::Load {
                    dst: v2,
                    src: r2,
                    volatile: false,
                }),
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v2],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "use after move of $var should be rejected");
    }

    #[test]
    fn ty_param_skipped() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        // v0 = Ty::Error (unresolved), used twice → no error (analysis mode)
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, Ty::error());

        let module = make_module(
            vec![inst(InstKind::Return(v0)), inst(InstKind::Return(v0))],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "Ty::Error should be skipped");
    }

    // E2E compile pipeline tests have been migrated to acvus-mir-test/tests/e2e.rs
    // (they depend on ExternFn registries which are only available there).
}
