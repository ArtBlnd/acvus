//! Move checking pass.
//!
//! Effectful values (`Iterator<T, Effectful>`, `Sequence<T, O, Effectful>`,
//! `Opaque`) are move-only: once consumed, they cannot be used again.
//! This pass performs forward dataflow analysis over the CFG to detect
//! use-after-move violations.
//!
//! Design:
//! - `Ty::Error` / `Ty::Param` / `Effect::Var` → skip (analysis mode).
//! - `Fn` with move-only captures → FnOnce (transitive).
//! - Join at merge points: `Alive ⊔ Moved = Moved` (conservative).
//! - $variables: tracked by name. `VarStore` revives, `VarLoad` of move-only consumes.

use std::collections::VecDeque;

use acvus_ast::Span;
use acvus_utils::{Astr, LocalIdOps};
use rustc_hash::FxHashMap;

use crate::analysis::cfg::{BlockIdx, Cfg, Terminator};
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

        // Containers — depends on effect
        Ty::Iterator(_, e) if e.is_effectful() => Some(true),
        Ty::Iterator(_, e) if e.is_pure() => Some(false),
        Ty::Sequence(_, _, e) if e.is_effectful() => Some(true),
        Ty::Sequence(_, _, e) if e.is_pure() => Some(false),

        // Unresolved effect — skip
        Ty::Iterator(_, e) if e.is_var() => None,
        Ty::Sequence(_, _, e) if e.is_var() => None,

        // Catch-all for Iterator/Sequence (should not happen with well-formed types)
        Ty::Iterator(_, _) | Ty::Sequence(_, _, _) => None,

        // Handle — always move-only (deferred computation, must be consumed exactly once)
        Ty::Handle(..) => Some(true),

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
        Ty::Param { .. } | Ty::Error(_) => None,
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
        check_body(&scope, closure, &mut errors);
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
            Terminator::IterStep { done, done_args } => {
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
                        &body.val_types,
                        &body.insts,
                        errors,
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
    let Some(ty) = val_types.get(&id) else {
        return false;
    };
    let Some(true) = is_move_only(ty) else {
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
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let span = inst.span;

    match &inst.kind {
        // === No operands / define only ===
        InstKind::Const { dst, .. }
        | InstKind::ContextProject { dst, .. }
        | InstKind::Poison { dst } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ContextLoad { dst, src } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
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
            if let Some(Liveness::Moved { at }) = state.get_var(*name)
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
            // Loading a move-only value from $var → var is now moved
            if let Some(ty) = val_types.get(dst)
                && is_move_only(ty) == Some(true)
            {
                state.set_var(*name, Liveness::Moved { at: inst_idx });
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::VarStore { name, src } => {
            // Consume the source value
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            // Variable is now alive with new value
            state.set_var(*name, Liveness::Alive);
        }
        InstKind::ContextStore { dst, value, .. } => {
            try_consume_value(scope, inst_idx, span, *dst, val_types, state, errors);
            try_consume_value(scope, inst_idx, span, *value, val_types, state, errors);
        }

        // === Consuming operations (move operands) ===
        InstKind::Return(v) => {
            try_consume_value(scope, inst_idx, span, *v, val_types, state, errors);
        }
        InstKind::Cast { dst, src, .. } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::IterStep {
            dst,
            iter_src,
            iter_dst,
            ..
        } => {
            try_consume_value(scope, inst_idx, span, *iter_src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
            state.set_value(*iter_dst, Liveness::Alive);
        }

        // Functions
        InstKind::LoadFunction { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        // Calls — all args are consumed; indirect callee is also consumed
        InstKind::FunctionCall { dst, callee, args, context_uses, context_defs } => {
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
    use crate::graph::FunctionId;
    use crate::ir::{Callee, DebugInfo, Inst, MirBody, MirModule};
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
                param_regs: Vec::new(),
                capture_regs: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 10,
            },
            closures: FxHashMap::default(),
        }
    }

    // -- is_move_only tests --

    #[test]
    fn pure_iterator_is_copy() {
        assert_eq!(
            is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::pure())),
            Some(false)
        );
    }

    #[test]
    fn effectful_iterator_is_move() {
        assert_eq!(
            is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::io())),
            Some(true)
        );
    }

    #[test]
    fn effect_var_is_unknown() {
        assert_eq!(
            is_move_only(&Ty::Iterator(Box::new(Ty::Int), Effect::Var(0))),
            None
        );
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
        let ty = Ty::Tuple(vec![Ty::Int, Ty::Iterator(Box::new(Ty::Int), Effect::io())]);
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_effectful_capture_is_move() {
        let ty = Ty::Fn {
            params: vec![param(Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![Ty::Iterator(Box::new(Ty::Int), Effect::io())],
            effect: Effect::pure(),
        };
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_pure_captures_is_copy() {
        let ty = Ty::Fn {
            params: vec![param(Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![Ty::Int, Ty::String],
            effect: Effect::io(), // effect of the fn doesn't matter, only captures
        };
        assert_eq!(is_move_only(&ty), Some(false));
    }

    // -- move check integration tests --

    #[test]
    fn no_error_for_pure_iterator_reuse() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        // v0 = Pure Iterator, used twice → OK
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, Ty::Iterator(Box::new(Ty::Int), Effect::pure()));
        val_types.insert(v1, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v2, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::FunctionCall {
                    dst: v1,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                inst(InstKind::FunctionCall {
                    dst: v2,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "pure iterator reuse should be allowed");
    }

    #[test]
    fn error_for_effectful_iterator_reuse() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        // v0 = Effectful Iterator, used twice → ERROR
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, Ty::Iterator(Box::new(Ty::Int), Effect::io()));
        val_types.insert(v1, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v2, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![
                inst(InstKind::FunctionCall {
                    dst: v1,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                inst(InstKind::FunctionCall {
                    dst: v2,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v0],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(
            errors.len(),
            1,
            "effectful iterator reuse should be rejected"
        );
        assert!(matches!(
            errors[0].kind,
            ValidationErrorKind::UseAfterMove { .. }
        ));
    }

    #[test]
    fn no_error_for_single_use_effectful() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        // v0 = Effectful Iterator, used once → OK
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, Ty::Iterator(Box::new(Ty::Int), Effect::io()));
        val_types.insert(v1, Ty::List(Box::new(Ty::Int)));

        let module = make_module(
            vec![inst(InstKind::FunctionCall {
                dst: v1,
                callee: Callee::Direct(FunctionId::alloc()),
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
        // $a = effectful iter (v0), VarLoad (v1) → moved, $a = new iter (v2) → alive, VarLoad (v3) → OK
        let mut val_types = FxHashMap::default();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        val_types.insert(v0, eff_iter.clone());
        val_types.insert(v1, eff_iter.clone());
        val_types.insert(v2, eff_iter.clone());
        val_types.insert(v3, eff_iter.clone());
        val_types.insert(v4, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v5, Ty::List(Box::new(Ty::Int)));

        let interner = Interner::new();
        let a = interner.intern("a");

        let module = make_module(
            vec![
                // $a = v0 (effectful)
                inst(InstKind::VarStore { name: a, src: v0 }),
                // v1 = $a → moves $a
                inst(InstKind::VarLoad { dst: v1, name: a }),
                // use v1
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v1],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                // $a = v2 (new value) → revives $a
                inst(InstKind::VarStore { name: a, src: v2 }),
                // v3 = $a → OK (new value)
                inst(InstKind::VarLoad { dst: v3, name: a }),
                // use v3
                inst(InstKind::FunctionCall {
                    dst: v5,
                    callee: Callee::Direct(FunctionId::alloc()),
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
        // $a = effectful, VarLoad → moved, VarLoad again → ERROR
        let mut val_types = FxHashMap::default();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        val_types.insert(v0, eff_iter.clone());
        val_types.insert(v1, eff_iter.clone());
        val_types.insert(v2, eff_iter.clone());
        val_types.insert(v3, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v4, Ty::List(Box::new(Ty::Int)));

        let interner = Interner::new();
        let a = interner.intern("a");

        let module = make_module(
            vec![
                inst(InstKind::VarStore { name: a, src: v0 }),
                inst(InstKind::VarLoad { dst: v1, name: a }),
                inst(InstKind::FunctionCall {
                    dst: v3,
                    callee: Callee::Direct(FunctionId::alloc()),
                    args: vec![v1],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                // Second load — $a already moved
                inst(InstKind::VarLoad { dst: v2, name: a }),
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(FunctionId::alloc()),
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
        // v0 = Ty::Param, used twice → no error (analysis mode)
        let mut subst = crate::ty::TySubst::new();
        let param_ty = subst.fresh_param();
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, param_ty);

        let module = make_module(
            vec![inst(InstKind::Return(v0)), inst(InstKind::Return(v0))],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "Ty::Param should be skipped");
    }

    // =====================================================================
    // E2E compile pipeline tests
    //
    // These test the full pipeline: parse → typeck → lower → validate.
    // Effectful iterators are injected via context types.
    // =====================================================================

    mod e2e {
        use crate::ty::{Effect, Param, Ty};
        use acvus_utils::Interner;

        fn param(ty: Ty) -> Param {
            let interner = Interner::new();
            Param::new(interner.intern("_"), ty)
        }

        fn eff_iter_ty() -> Ty {
            Ty::Iterator(Box::new(Ty::Int), Effect::io())
        }

        fn pure_iter_ty() -> Ty {
            Ty::Iterator(Box::new(Ty::Int), Effect::pure())
        }

        fn compile_script(source: &str, ctx: &[(&str, Ty)]) -> Result<(), Vec<String>> {
            let interner = Interner::new();
            crate::test::compile_script(&interner, source, ctx)
                .map(|_| ())
                .map_err(|errs| {
                    errs.iter()
                        .map(|e| format!("{}", e.display(&interner)))
                        .collect()
                })
        }

        fn compile_template(source: &str, _ctx: &[(&str, Ty)]) -> Result<(), Vec<String>> {
            let interner = Interner::new();
            crate::test::compile_template(&interner, source, _ctx)
                .map(|_| ())
                .map_err(|errs| {
                    errs.iter()
                        .map(|e| format!("{}", e.display(&interner)))
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
            assert!(
                result.is_err(),
                "should reject $var double load of effectful"
            );
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
            assert!(
                result.is_ok(),
                "pure iterator should be reusable: {result:?}"
            );
        }

        /// C2: effectful iterator used once
        #[test]
        fn accept_effectful_single_use() {
            let result = compile_script("x = @src; x | collect", &[("src", eff_iter_ty())]);
            assert!(
                result.is_ok(),
                "single use of effectful should be allowed: {result:?}"
            );
        }

        /// C3: effectful iter collected → list is pure, reusable
        #[test]
        fn accept_collect_then_reuse() {
            let result = compile_script(
                "list = @src | collect; a = list | len; b = list | len; a + b",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_ok(),
                "collected list should be reusable: {result:?}"
            );
        }

        /// C4: $var reassignment revives
        #[test]
        fn accept_var_reassign() {
            let result = compile_template(
                "{{ $a = @src }}{{ $a | collect | len | to_string }}{{ $a = @src2 }}{{ $a | collect | len | to_string }}",
                &[("src", eff_iter_ty()), ("src2", eff_iter_ty())],
            );
            assert!(
                result.is_ok(),
                "reassigned $var should be alive: {result:?}"
            );
        }

        /// C5: pure values are always copyable
        #[test]
        fn accept_pure_values_copy() {
            let result =
                compile_script("x = @val; a = x + 1; b = x + 2; a + b", &[("val", Ty::Int)]);
            assert!(result.is_ok());
        }

        /// C6: effectful iter in pipe chain (single linear flow)
        #[test]
        fn accept_effectful_pipe_chain() {
            let result = compile_script(
                "@src | filter(|x| -> x > 0) | map(|x| -> x * 2) | collect",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_ok(),
                "linear pipe chain should be allowed: {result:?}"
            );
        }

        /// C7: effectful function (no move-only captures) can be called multiple times
        #[test]
        fn accept_effectful_fn_multiple_calls() {
            let fn_ty = Ty::Fn {
                params: vec![param(Ty::Int)],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::io(),
            };
            let result = compile_script("a = @f(1); b = @f(2); a + b", &[("f", fn_ty)]);
            assert!(
                result.is_ok(),
                "effectful fn without move-only captures should be callable multiple times: {result:?}"
            );
        }

        /// E2: List containing effectful iterator (transitive move-only)
        #[test]
        fn reject_list_of_effectful_reuse() {
            let ty = Ty::List(Box::new(eff_iter_ty()));
            let result =
                compile_script("x = @src; a = x | len; b = x | len; a + b", &[("src", ty)]);
            // List<Iterator<Int, Effectful>> is move-only (transitive)
            assert!(
                result.is_err(),
                "List containing effectful should be move-only"
            );
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

        // -- Branch tests: move in one branch, use after merge --
        // Note: @context loads always create fresh values (ContextLoad → Alive),
        // so branch tests must use $var (mutable variable) which tracks liveness.

        /// B1: $var holding move-only value, consumed in one branch, used after merge → ERROR
        /// (conservative: any branch moves → merged state is Moved)
        #[test]
        fn reject_branch_move_then_use() {
            // $a holds effectful iter, consumed in true branch, used again after merge
            let result = compile_template(
                "{{ $a = @src }}{{ true = @flag }}{{ $a | collect | len | to_string }}{{_}}nothing{{/}}{{ $a | collect | len | to_string }}",
                &[("flag", Ty::Bool), ("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "should reject use after move across branch: {result:?}"
            );
            assert!(has_use_after_move(&result.unwrap_err()));
        }

        /// B2: $var consumed in both branches, then used after merge → ERROR
        #[test]
        fn reject_both_branches_move_then_use() {
            let result = compile_template(
                "{{ $a = @src }}{{ true = @flag }}{{ $a | collect | len | to_string }}{{_}}{{ $a | collect | len | to_string }}{{/}}{{ $a | collect | len | to_string }}",
                &[("flag", Ty::Bool), ("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "should reject use after move in both branches: {result:?}"
            );
        }

        /// B3: $var consumed in one branch only, no use after merge → OK
        #[test]
        fn accept_branch_move_no_use_after() {
            let result = compile_template(
                "{{ $a = @src }}{{ true = @flag }}{{ $a | collect | len | to_string }}{{_}}nothing{{/}}",
                &[("flag", Ty::Bool), ("src", eff_iter_ty())],
            );
            assert!(
                result.is_ok(),
                "move in branch without post-merge use should be OK: {result:?}"
            );
        }

        // -- Nested closure + move-only tests --
        // Note: Lambda body cannot contain pipe expressions (-> binds tighter than |).
        // Use collect(x) call syntax instead of x | collect.

        /// N1: Closure capturing effectful iterator → FnOnce, called twice → ERROR
        /// f captures move-only x via closure, so f itself is move-only.
        /// Second call of f → use after move.
        #[test]
        fn reject_fnonce_double_call() {
            let result = compile_script(
                "x = @src; f = (|z| -> collect(x)); a = f(0); b = f(0); a",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "FnOnce called twice should be rejected: {result:?}"
            );
            assert!(has_use_after_move(&result.unwrap_err()));
        }

        /// N2: Closure capturing pure value → Fn, called multiple times → OK
        #[test]
        fn accept_pure_capture_fn_multi_call() {
            let result = compile_script(
                "x = @val; f = (|a| -> x + a); a = f(1); b = f(2); a + b",
                &[("val", Ty::Int)],
            );
            assert!(
                result.is_ok(),
                "Fn with pure captures should be callable multiple times: {result:?}"
            );
        }

        /// N3: Closure capturing effectful → used once → OK
        #[test]
        fn accept_fnonce_single_call() {
            let result = compile_script(
                "x = @src; f = (|z| -> collect(x)); f(0)",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_ok(),
                "FnOnce called once should be OK: {result:?}"
            );
        }

        // -- Lambda return coercion tests --

        /// L1: Lambda returns Deque where Iterator expected (flat_map) → Cast inserted at return
        #[test]
        fn accept_lambda_return_deque_as_iterator() {
            // flat_map expects Fn(T) → Iterator<U>
            // Lambda body [x, x*2] returns Deque → DequeToIterator Cast inserted
            let result = compile_script(
                "@items | flat_map(|x| -> [x, x * 2]) | collect",
                &[("items", Ty::List(Box::new(Ty::Int)))],
            );
            assert!(
                result.is_ok(),
                "lambda returning Deque where Iterator expected should compile: {result:?}"
            );
        }

        /// L2: Lambda returns Int (no coercion needed for map)
        #[test]
        fn accept_lambda_return_scalar() {
            let result = compile_script(
                "@items | map(|x| -> x + 1) | collect",
                &[("items", Ty::List(Box::new(Ty::Int)))],
            );
            assert!(
                result.is_ok(),
                "lambda returning scalar should compile: {result:?}"
            );
        }

        /// L3: Nested flat_map with Deque return at both levels
        #[test]
        fn accept_nested_flat_map_deque_return() {
            let result = compile_script(
                "@items | flat_map(|x| -> [x, x + 10]) | map(|x| -> x * 2) | collect",
                &[("items", Ty::List(Box::new(Ty::Int)))],
            );
            assert!(
                result.is_ok(),
                "nested flat_map + map with Deque return should compile: {result:?}"
            );
        }

        // -- FunctionCall (Indirect) FnOnce tests --

        /// F1: FnOnce passed to map (HOF calls it multiple times) → should be OK
        /// because at the MIR level, the fn is passed once to FunctionCall.
        #[test]
        fn accept_fnonce_passed_to_map() {
            // f captures effectful, passed to map (single use of f at MIR level)
            let result = compile_script(
                "x = @src; f = (|z| -> collect(x)); @items | map(f) | collect",
                &[
                    ("src", eff_iter_ty()),
                    ("items", Ty::List(Box::new(Ty::Int))),
                ],
            );
            // This should compile — f is passed once to map (single FunctionCall arg)
            assert!(
                result.is_ok(),
                "FnOnce passed once to HOF should be OK: {result:?}"
            );
        }

        /// F2: Lambda with @context in body (not capture) should be Fn, not FnOnce.
        /// @src is a ContextLoad in the lambda body, executed at each call, not captured.
        #[test]
        fn accept_lambda_context_in_body_is_fn() {
            let result = compile_template(
                "{{ $f = (|z| -> collect(@src)) }}{{ $f(0) | len | to_string }}{{ $f(0) | len | to_string }}",
                &[("src", eff_iter_ty())],
            );
            // @src is NOT captured — it's loaded fresh each call via ContextLoad.
            // So the lambda is Fn (not FnOnce), callable multiple times.
            assert!(
                result.is_ok(),
                "Lambda with @context in body (not capture) should be Fn: {result:?}"
            );
        }

        /// F3: Closure explicitly capturing a local move-only value, called twice → ERROR
        #[test]
        fn reject_fnonce_local_capture_double() {
            let result = compile_script(
                "x = @src; f = (|z| -> collect(x)); a = f(0); b = f(0); a",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "FnOnce with local capture double call should be rejected: {result:?}"
            );
        }

        /// Effectful without purify — still rejected (soundness baseline)
        #[test]
        fn reject_effectful_without_purify() {
            let result = compile_script(
                "x = @src; a = x | collect; b = x | collect; a",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "effectful without purify should still be rejected"
            );
        }

        /// Effectful in $var — reuse rejected (soundness)
        #[test]
        fn reject_effectful_var_without_purify() {
            let result = compile_template(
                "{{ $a = @src }}{{ $a | collect | len | to_string }}{{ $a | collect | len | to_string }}",
                &[("src", eff_iter_ty())],
            );
            assert!(
                result.is_err(),
                "effectful $var without purify should be rejected"
            );
        }
    }
}
