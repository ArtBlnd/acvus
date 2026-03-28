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

        // Handle — always move-only (deferred computation, must be consumed exactly once)
        Ty::Handle(..) => Some(true),

        // UserDefined — always move-only (unknown internals)
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
                    propagate_args(
                        scope,
                        &block_exit[idx.0],
                        args,
                        &cfg.blocks[target_idx.0].params,
                        &body.val_types,
                        &body.insts,
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
                            &body.val_types,
                            &body.insts,
                            errors,
                            &mut block_entry[target_idx.0],
                        );
                        if propagate_state(&block_exit[idx.0], &mut block_entry[target_idx.0]) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            Terminator::ListStep { done, done_args } => {
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
            Terminator::Return => {}
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
    _insts: &[Inst],
    _errors: &mut Vec<ValidationError>,
    target_entry: &mut MoveState,
) {
    // Map arg liveness → param liveness.
    // If an arg is move-only and moved, the param inherits Moved.
    for (arg, param) in args.iter().zip(params.iter()) {
        let arg_liveness = source.get_value(*arg).unwrap_or(Liveness::Alive);
        let is_move = val_types.get(param).and_then(|ty| is_move_only(ty)) == Some(true);
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
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let span = inst.span;

    match &inst.kind {
        // === No operands / define only ===
        InstKind::Const { dst, .. }
        | InstKind::ContextProject { dst, .. }
        | InstKind::Poison { dst }
        | InstKind::Undef { dst } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ContextLoad { dst, src } => {
            try_consume_value(scope, inst_idx, span, *src, val_types, state, errors);
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::BlockLabel { params, .. } => {
            for p in params {
                // Only set Alive if not already set by propagate_args.
                // If propagate_args already set it to Moved (from incoming),
                // we must not override.
                state.values.entry(*p).or_insert(Liveness::Alive);
            }
        }
        InstKind::Nop => {}

        // === Variable operations ===
        InstKind::VarLoad { dst, name } | InstKind::ParamLoad { dst, name } => {
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
        InstKind::ListStep {
            dst,
            index_dst,
            ..
        } => {
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
        assert_eq!(
            errors.len(),
            1,
            "UserDefined reuse should be rejected"
        );
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
        // $a = move-only (v0), VarLoad (v1) → moved, $a = new value (v2) → alive, VarLoad (v3) → OK
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
        val_types.insert(v3, move_ty.clone());
        val_types.insert(v4, Ty::List(Box::new(Ty::Int)));
        val_types.insert(v5, Ty::List(Box::new(Ty::Int)));

        let interner = Interner::new();
        let a = interner.intern("a");

        let module = make_module(
            vec![
                // $a = v0 (move-only)
                inst(InstKind::VarStore { name: a, src: v0 }),
                // v1 = $a → moves $a
                inst(InstKind::VarLoad { dst: v1, name: a }),
                // use v1
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
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
        // $a = move-only, VarLoad → moved, VarLoad again → ERROR
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
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
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    args: vec![v1],
                    context_uses: vec![],
                    context_defs: vec![],
                }),
                // Second load — $a already moved
                inst(InstKind::VarLoad { dst: v2, name: a }),
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

    // E2E compile pipeline tests have been migrated to acvus-mir-test/tests/e2e.rs
    // (they depend on ExternFn registries which are only available there).
}
