//! Move checking pass.
//!
//! Effectful values (`Iterator<T, Effectful>`, `Sequence<T, O, Effectful>`,
//! `Opaque`) are move-only: once consumed, they cannot be used again.
//! This pass performs forward dataflow analysis over the CFG to detect
//! use-after-move violations.
//!
//! Design:
//! - `Ty::Error` / `Ty::Var` / `Effect::Var` → skip (analysis mode).
//! - `Fn` with move-only captures → FnOnce (transitive).
//! - Join at merge points: `Alive ⊔ Moved = Moved` (conservative).
//! - $variables: tracked by name. `VarStore` revives, `VarLoad` of move-only consumes.

use std::collections::VecDeque;

use acvus_ast::Span;
use rustc_hash::{FxHashMap, FxHashSet};
use acvus_utils::Astr;

use crate::analysis::cfg::{BasicBlock, BlockIdx, Cfg, Terminator};
use crate::ir::{Inst, InstKind, Label, MirBody, MirModule, ValueId};
use crate::ty::{Effect, Ty};

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
        Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit
        | Ty::Range | Ty::Byte => Some(false),

        // Containers — depends on effect
        Ty::Iterator(_, Effect::Effectful) => Some(true),
        Ty::Iterator(_, Effect::Pure) => Some(false),
        Ty::Sequence(_, _, Effect::Effectful) => Some(true),
        Ty::Sequence(_, _, Effect::Pure) => Some(false),

        // Unresolved effect — skip
        Ty::Iterator(_, Effect::Var(_)) => None,
        Ty::Sequence(_, _, Effect::Var(_)) => None,

        // Opaque — always move-only (unknown internals)
        Ty::Opaque(_) => Some(true),

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

        // Unknown — skip
        Ty::Var(_) | Ty::Error(_) | Ty::Infer(_) => None,
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
    Moved { at: usize },
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
    vars: FxHashMap<Astr, Liveness>,
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

    fn get_var(&self, name: Astr) -> Option<Liveness> {
        self.vars.get(&name).copied()
    }

    fn set_var(&mut self, name: Astr, liveness: Liveness) {
        self.vars.insert(name, liveness);
    }

    /// Join another state into this one. Returns true if anything changed.
    fn join_from(&mut self, other: &MoveState) -> bool {
        let mut changed = false;
        for (&id, &liveness) in &other.values {
            let entry = self.values.entry(id).or_insert(Liveness::Alive);
            let joined = entry.join(liveness);
            if *entry != joined {
                *entry = joined;
                changed = true;
            }
        }
        for (&name, &liveness) in &other.vars {
            let entry = self.vars.entry(name).or_insert(Liveness::Alive);
            let joined = entry.join(liveness);
            if *entry != joined {
                *entry = joined;
                changed = true;
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
        check_body(&scope, &closure.body, &mut errors);
    }
    errors
}

fn check_body(scope: &str, body: &MirBody, errors: &mut Vec<ValidationError>) {
    let cfg = Cfg::build(&body.insts);
    if cfg.blocks.is_empty() {
        return;
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
        for &inst_idx in &block.inst_indices {
            process_inst(
                scope,
                inst_idx,
                &body.insts[inst_idx],
                &body.val_types,
                &mut state,
                errors,
            );
        }

        block_exit[idx.0] = state;

        // Propagate to successors
        match &block.terminator {
            Terminator::Jump { target, args } => {
                if let Some(&target_idx) = cfg.label_to_block.get(target) {
                    // Check move state of jump args
                    propagate_args(
                        scope,
                        &block_exit[idx.0],
                        args,
                        &cfg.blocks[target_idx.0].params,
                        &body.val_types,
                        &body.insts,
                        errors,
                    );
                    if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::JumpIf { cond: _, then_label, then_args, else_label, else_args } => {
                for (label, args) in [(then_label, then_args), (else_label, else_args)] {
                    if let Some(&target_idx) = cfg.label_to_block.get(label) {
                        propagate_args(
                            scope,
                            &block_exit[idx.0],
                            args,
                            &cfg.blocks[target_idx.0].params,
                            &body.val_types,
                            &body.insts,
                            errors,
                        );
                        if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n {
                    if propagate_state(&block_exit[idx.0], &mut block_entry[next]) {
                        worklist.push_back(BlockIdx(next));
                    }
                }
            }
            Terminator::Return => {}
        }
    }
}

fn propagate_state(source: &MoveState, target: &mut MoveState) -> bool {
    target.join_from(source)
}

fn propagate_args(
    _scope: &str,
    _source: &MoveState,
    _args: &[ValueId],
    _params: &[ValueId],
    _val_types: &FxHashMap<ValueId, Ty>,
    _insts: &[Inst],
    _errors: &mut Vec<ValidationError>,
) {
    // Block params receive values from args. The args are consumed (moved)
    // at the jump site, which is handled by the block's exit state.
    // No additional checking needed here — the args were already checked
    // when they were used in the block's instructions.
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
    let Some(ty) = val_types.get(&id) else { return false };
    let Some(true) = is_move_only(ty) else { return false };

    // Check if already moved
    if let Some(Liveness::Moved { at }) = state.get_value(id) {
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span,
            kind: ValidationErrorKind::UseAfterMove {
                value_id: id.0,
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
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let span = inst.span;

    match &inst.kind {
        // === No operands / define only ===
        InstKind::Const { dst, .. } | InstKind::ContextLoad { dst, .. } | InstKind::Poison { dst } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::BlockLabel { params, .. } => {
            for p in params {
                state.set_value(*p, Liveness::Alive);
            }
        }
        InstKind::Nop => {}

        // === Variable operations ===
        InstKind::VarLoad { dst, name } => {
            // Check if $variable has been moved
            if let Some(Liveness::Moved { at }) = state.get_var(*name) {
                if let Some(ty) = val_types.get(dst) {
                    if is_move_only(ty) == Some(true) {
                        errors.push(ValidationError {
                            scope: scope.to_string(),
                            inst_index: inst_idx,
                            span,
                            kind: ValidationErrorKind::UseAfterMove {
                                value_id: dst.0,
                                moved_at: at,
                                ty: ty.clone(),
                            },
                        });
                    }
                }
            }
            // Loading a move-only value from $var → var is now moved
            if let Some(ty) = val_types.get(dst) {
                if is_move_only(ty) == Some(true) {
                    state.set_var(*name, Liveness::Moved { at: inst_idx });
                }
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::VarStore { name, src } => {
            // Consume the source value
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            // Variable is now alive with new value
            state.set_var(*name, Liveness::Alive);
        }

        // === Consuming operations (move operands) ===
        InstKind::Yield(v) => {
            try_consume_value(scope, inst_idx, span, *v, val_types, state, errors);
        }
        InstKind::Return(v) => {
            try_consume_value(scope, inst_idx, span, *v, val_types, state, errors);
        }
        InstKind::Cast { dst, src, .. } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::IterStep { dst, src } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }

        // Calls — all args are consumed
        InstKind::BuiltinCall { dst, args, .. } => {
            for arg in args {
                try_consume_value(scope, inst_idx, span, *arg, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ExternCall { dst, args, .. } => {
            for arg in args {
                try_consume_value(scope, inst_idx, span, *arg, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ClosureCall { dst, closure, args } => {
            try_consume_value(scope, inst_idx, span, *closure, val_types, state, errors);
            for arg in args {
                try_consume_value(scope, inst_idx, span, *arg, val_types, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
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
        InstKind::MakeRange { dst, start, end, .. } => {
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
        InstKind::ObjectGet { dst, object: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TupleIndex { dst, tuple: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ListIndex { dst, list: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ListGet { dst, list: _, index: _ } => {
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
        InstKind::BinOp { dst, left: _, right: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::UnaryOp { dst, operand: _, .. } => {
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
    use crate::ir::{DebugInfo, Inst, MirBody, MirModule};
    use crate::ty::Effect;
    use acvus_utils::Interner;

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
                debug: DebugInfo::new(),
                val_count: 100,
                label_count: 10,
            },
            closures: FxHashMap::default(),
        }
    }

    // -- is_move_only tests --

    #[test]
    fn pure_iterator_is_copy() {
        assert_eq!(is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::Pure)), Some(false));
    }

    #[test]
    fn effectful_iterator_is_move() {
        assert_eq!(is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::Effectful)), Some(true));
    }

    #[test]
    fn effect_var_is_unknown() {
        assert_eq!(is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::Var(0))), None);
    }

    #[test]
    fn pure_types_are_copy() {
        assert_eq!(is_move_only(&Ty::Int), Some(false));
        assert_eq!(is_move_only(&Ty::String), Some(false));
        assert_eq!(is_move_only(&Ty::Bool), Some(false));
        assert_eq!(is_move_only(&Ty::List(Box::new(Ty::Int))), Some(false));
    }

    #[test]
    fn opaque_is_move() {
        assert_eq!(is_move_only(&Ty::Opaque("handle".into())), Some(true));
    }

    #[test]
    fn tuple_with_effectful_is_move() {
        let ty = Ty::Tuple(vec![
            Ty::Int,
            Ty::Iterator(Box::new(Ty::Int), Effect::Effectful),
        ]);
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_effectful_capture_is_move() {
        let ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            kind: crate::ty::FnKind::Lambda,
            captures: vec![Ty::Iterator(Box::new(Ty::Int), Effect::Effectful)],
            effect: Effect::Pure,
        };
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_pure_captures_is_copy() {
        let ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            kind: crate::ty::FnKind::Lambda,
            captures: vec![Ty::Int, Ty::String],
            effect: Effect::Effectful, // effect of the fn doesn't matter, only captures
        };
        assert_eq!(is_move_only(&ty), Some(false));
    }

    // -- move check integration tests --

    #[test]
    fn no_error_for_pure_iterator_reuse() {
        // v0 = Pure Iterator, used twice → OK
        let mut val_types = FxHashMap::default();
        val_types.insert(ValueId(0), Ty::Iterator(Box::new(Ty::Int), Effect::Pure));
        val_types.insert(ValueId(1), Ty::List(Box::new(Ty::Int)));
        val_types.insert(ValueId(2), Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::BuiltinCall {
                    dst: ValueId(1),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(0)],
                }),
                inst(InstKind::BuiltinCall {
                    dst: ValueId(2),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(0)],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "pure iterator reuse should be allowed");
    }

    #[test]
    fn error_for_effectful_iterator_reuse() {
        // v0 = Effectful Iterator, used twice → ERROR
        let mut val_types = FxHashMap::default();
        val_types.insert(ValueId(0), Ty::Iterator(Box::new(Ty::Int), Effect::Effectful));
        val_types.insert(ValueId(1), Ty::List(Box::new(Ty::Int)));
        val_types.insert(ValueId(2), Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::BuiltinCall {
                    dst: ValueId(1),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(0)],
                }),
                inst(InstKind::BuiltinCall {
                    dst: ValueId(2),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(0)],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "effectful iterator reuse should be rejected");
        assert!(matches!(errors[0].kind, ValidationErrorKind::UseAfterMove { .. }));
    }

    #[test]
    fn no_error_for_single_use_effectful() {
        // v0 = Effectful Iterator, used once → OK
        let mut val_types = FxHashMap::default();
        val_types.insert(ValueId(0), Ty::Iterator(Box::new(Ty::Int), Effect::Effectful));
        val_types.insert(ValueId(1), Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::BuiltinCall {
                    dst: ValueId(1),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(0)],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty());
    }

    #[test]
    fn var_reassign_revives() {
        // $a = effectful iter (v0), VarLoad (v1) → moved, $a = new iter (v2) → alive, VarLoad (v3) → OK
        let mut val_types = FxHashMap::default();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::Effectful);
        val_types.insert(ValueId(0), eff_iter.clone());
        val_types.insert(ValueId(1), eff_iter.clone());
        val_types.insert(ValueId(2), eff_iter.clone());
        val_types.insert(ValueId(3), eff_iter.clone());
        val_types.insert(ValueId(4), Ty::List(Box::new(Ty::Int)));
        val_types.insert(ValueId(5), Ty::List(Box::new(Ty::Int)));

        let interner = Interner::new();
        let a = interner.intern("a");

        let module = make_module(
            vec![
                // $a = v0 (effectful)
                inst(InstKind::VarStore { name: a, src: ValueId(0) }),
                // v1 = $a → moves $a
                inst(InstKind::VarLoad { dst: ValueId(1), name: a }),
                // use v1
                inst(InstKind::BuiltinCall {
                    dst: ValueId(4),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(1)],
                }),
                // $a = v2 (new value) → revives $a
                inst(InstKind::VarStore { name: a, src: ValueId(2) }),
                // v3 = $a → OK (new value)
                inst(InstKind::VarLoad { dst: ValueId(3), name: a }),
                // use v3
                inst(InstKind::BuiltinCall {
                    dst: ValueId(5),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(3)],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "reassigned variable should be alive: {errors:?}");
    }

    #[test]
    fn var_use_after_move() {
        // $a = effectful, VarLoad → moved, VarLoad again → ERROR
        let mut val_types = FxHashMap::default();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::Effectful);
        val_types.insert(ValueId(0), eff_iter.clone());
        val_types.insert(ValueId(1), eff_iter.clone());
        val_types.insert(ValueId(2), eff_iter.clone());
        val_types.insert(ValueId(3), Ty::List(Box::new(Ty::Int)));
        val_types.insert(ValueId(4), Ty::List(Box::new(Ty::Int)));

        let interner = Interner::new();
        let a = interner.intern("a");

        let module = make_module(
            vec![
                inst(InstKind::VarStore { name: a, src: ValueId(0) }),
                inst(InstKind::VarLoad { dst: ValueId(1), name: a }),
                inst(InstKind::BuiltinCall {
                    dst: ValueId(3),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(1)],
                }),
                // Second load — $a already moved
                inst(InstKind::VarLoad { dst: ValueId(2), name: a }),
                inst(InstKind::BuiltinCall {
                    dst: ValueId(4),
                    builtin: crate::builtins::BuiltinId::Collect,
                    args: vec![ValueId(2)],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "use after move of $var should be rejected");
    }

    #[test]
    fn ty_var_skipped() {
        // v0 = Ty::Var, used twice → no error (analysis mode)
        let mut val_types = FxHashMap::default();
        val_types.insert(ValueId(0), Ty::Var(crate::ty::TyVar(0)));
        val_types.insert(ValueId(1), Ty::Unit);
        val_types.insert(ValueId(2), Ty::Unit);

        let module = make_module(
            vec![
                inst(InstKind::Yield(ValueId(0))),
                inst(InstKind::Yield(ValueId(0))),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "Ty::Var should be skipped");
    }

    // =====================================================================
    // E2E compile pipeline tests
    //
    // These test the full pipeline: parse → typeck → lower → validate.
    // Effectful iterators are injected via context types.
    // =====================================================================

    mod e2e {
        use crate::context_registry::ContextTypeRegistry;
        use crate::error::MirErrorKind;
        use crate::ty::{Effect, FnKind, Ty};
        use acvus_utils::Interner;
        use rustc_hash::FxHashMap;

        fn eff_iter_ty() -> Ty {
            Ty::Iterator(Box::new(Ty::Int), Effect::Effectful)
        }

        fn pure_iter_ty() -> Ty {
            Ty::Iterator(Box::new(Ty::Int), Effect::Pure)
        }

        fn compile_script(source: &str, ctx: &[(&str, Ty)]) -> Result<(), Vec<String>> {
            let i = Interner::new();
            let context: FxHashMap<_, _> = ctx
                .iter()
                .map(|(name, ty)| (i.intern(name), ty.clone()))
                .collect();
            let reg = ContextTypeRegistry::all_system(context);
            let script = acvus_ast::parse_script(&i, source)
                .map_err(|e| vec![format!("parse error: {e}")])?;
            crate::compile_script(&i, &script, &reg)
                .map(|_| ())
                .map_err(|errs| {
                    errs.iter()
                        .map(|e| format!("{:?}", e.kind))
                        .collect()
                })
        }

        fn compile_template(source: &str, ctx: &[(&str, Ty)]) -> Result<(), Vec<String>> {
            let i = Interner::new();
            let context: FxHashMap<_, _> = ctx
                .iter()
                .map(|(name, ty)| (i.intern(name), ty.clone()))
                .collect();
            let reg = ContextTypeRegistry::all_system(context);
            let template = acvus_ast::parse(&i, source)
                .map_err(|e| vec![format!("parse error: {e}")])?;
            crate::compile(&i, &template, &reg)
                .map(|_| ())
                .map_err(|errs| {
                    errs.iter()
                        .map(|e| format!("{:?}", e.kind))
                        .collect()
                })
        }

        fn has_use_after_move(errors: &[String]) -> bool {
            errors.iter().any(|e| e.contains("use of move-only value"))
        }

        // -- Soundness: should REJECT --

        /// S1: effectful iterator used twice in sequence
        #[test]
        fn reject_effectful_iter_reuse() {
            let result = compile_script(
                "x = @src; x | collect; x | collect",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_err(), "should reject effectful iter reuse");
            assert!(has_use_after_move(&result.unwrap_err()));
        }

        /// S2: effectful iter assigned to $var, loaded twice
        #[test]
        fn reject_var_double_load() {
            let result = compile_template(
                "{{ $a = @src }}{{ $a | collect | len | to_string }}{{ $a | collect | len | to_string }}",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_err(), "should reject $var double load of effectful");
            assert!(has_use_after_move(&result.unwrap_err()));
        }

        /// S3: effectful iter piped twice
        #[test]
        fn reject_effectful_pipe_reuse() {
            let result = compile_script(
                "x = @src; a = x | collect; b = x | collect; a",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_err());
            assert!(has_use_after_move(&result.unwrap_err()));
        }

        // -- Completeness: should ACCEPT --

        /// C1: pure iterator reused freely
        #[test]
        fn accept_pure_iter_reuse() {
            let result = compile_script(
                "x = @src; a = x | collect; b = x | collect; a",
                &[("src", pure_iter_ty())],
            );
            assert!(result.is_ok(), "pure iterator should be reusable: {result:?}");
        }

        /// C2: effectful iterator used once
        #[test]
        fn accept_effectful_single_use() {
            let result = compile_script(
                "x = @src; x | collect",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_ok(), "single use of effectful should be allowed: {result:?}");
        }

        /// C3: effectful iter collected → list is pure, reusable
        #[test]
        fn accept_collect_then_reuse() {
            let result = compile_script(
                "list = @src | collect; a = list | len; b = list | len; a + b",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_ok(), "collected list should be reusable: {result:?}");
        }

        /// C4: $var reassignment revives
        #[test]
        fn accept_var_reassign() {
            let result = compile_template(
                "{{ $a = @src }}{{ $a | collect | len | to_string }}{{ $a = @src2 }}{{ $a | collect | len | to_string }}",
                &[("src", eff_iter_ty()), ("src2", eff_iter_ty())],
            );
            assert!(result.is_ok(), "reassigned $var should be alive: {result:?}");
        }

        /// C5: pure values are always copyable
        #[test]
        fn accept_pure_values_copy() {
            let result = compile_script(
                "x = @val; a = x + 1; b = x + 2; a + b",
                &[("val", Ty::Int)],
            );
            assert!(result.is_ok());
        }

        /// C6: effectful iter in pipe chain (single linear flow)
        #[test]
        fn accept_effectful_pipe_chain() {
            let result = compile_script(
                "@src | filter(x -> x > 0) | map(x -> x * 2) | collect",
                &[("src", eff_iter_ty())],
            );
            assert!(result.is_ok(), "linear pipe chain should be allowed: {result:?}");
        }

        /// C7: effectful function (no move-only captures) can be called multiple times
        #[test]
        fn accept_effectful_fn_multiple_calls() {
            let fn_ty = Ty::Fn {
                params: vec![Ty::Int],
                ret: Box::new(Ty::Int),
                kind: FnKind::Extern,
                captures: vec![],
                effect: Effect::Effectful,
            };
            let result = compile_script(
                "a = @f(1); b = @f(2); a + b",
                &[("f", fn_ty)],
            );
            assert!(result.is_ok(), "effectful fn without move-only captures should be callable multiple times: {result:?}");
        }

        // -- Edge cases --

        /// E1: Ty::Error context → skip move check (no false positive)
        #[test]
        fn skip_error_type() {
            // compile_analysis_partial allows Error types
            let i = Interner::new();
            let reg = ContextTypeRegistry::all_system(FxHashMap::default());
            let template = acvus_ast::parse(&i, "{{ @unknown }}").unwrap();
            // This should fail with undefined context, not move error
            let result = crate::compile(&i, &template, &reg);
            assert!(result.is_err());
            // Should NOT contain use-after-move
            let errors: Vec<String> = result
                .unwrap_err()
                .iter()
                .map(|e| format!("{:?}", e.kind))
                .collect();
            assert!(!has_use_after_move(&errors));
        }

        /// E2: List containing effectful iterator (transitive move-only)
        #[test]
        fn reject_list_of_effectful_reuse() {
            let ty = Ty::List(Box::new(eff_iter_ty()));
            let result = compile_script(
                "x = @src; a = x | len; b = x | len; a + b",
                &[("src", ty)],
            );
            // List<Iterator<Int, Effectful>> is move-only (transitive)
            assert!(result.is_err(), "List containing effectful should be move-only");
        }

        /// E3: Option containing effectful (transitive)
        #[test]
        fn reject_option_effectful_reuse() {
            let ty = Ty::Option(Box::new(eff_iter_ty()));
            let result = compile_script(
                "x = @src; a = x | unwrap | collect; b = x | unwrap | collect; a",
                &[("src", ty)],
            );
            assert!(result.is_err(), "Option<Effectful> should be move-only");
        }
    }
}
