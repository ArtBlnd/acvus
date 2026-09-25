//! Move checking pass.
//!
//! Move-only values (iterators, sequences, opaque values) cannot be used again
//! once consumed. This pass performs forward dataflow analysis over the CFG to
//! detect use-after-move violations.
//!
//! Design:
//! - `Ty::Error` -> skip (analysis mode).
//! - `Fn` with move-only captures -> FnOnce (transitive).
//! - Join at merge points: `Alive  lub  Moved = Moved` (conservative).
//! - A storage is tracked by its slot and by the parts of it that moved: it
//!   is read -- by a `Take` of a place in it, by a `Ref` to one, by a store
//!   into one -- only while the part read is alive. A take of a place moves
//!   that place; a store into a place revives it and everything under it.
//! - A place a `Take` takes out for a call (RFC-0041) is moved, a word's as
//!   any other's, until the `Assign` that restores it: no store revives it
//!   before then, and every store overlapping it is refused.

use std::collections::VecDeque;

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_utils::{Astr, LocalIdOps};
use rustc_hash::FxHashMap;

use crate::cfg::{BlockIdx, Terminator, promote};
use crate::ir::{
    Callee, DebugInfo, Inst, InstKind, MirBody, MirModule, PathSeg, RefTarget, ValueId,
};
use crate::ty::Ty;

use super::type_check::{ConflictTouch, ValidationError, ValidationErrorKind};

// ---------------------------------------------------------------------------
// is_move_only
// ---------------------------------------------------------------------------

/// Whether a type moves at the IR (RFC-0018): every type that is not a word
/// (`TyTerm::is_word`), `String` included. `None` for a type the analysis
/// cannot classify.
pub fn is_move_only(ty: &Ty) -> Option<bool> {
    ty.is_word().map(|word| !word)
}

// ---------------------------------------------------------------------------
// Move state
// ---------------------------------------------------------------------------

/// Liveness of a single value. The instructions that read a part out of a
/// value name that part in the instruction and not in a place, so a value's
/// parts are not tracked by path -- deliberately, since nothing in the IR
/// hands this pass a path to track them by. A storage's parts are places, and
/// a storage carries [`StorageLiveness`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Liveness {
    Alive,
    PartlyMoved { site: MoveSite },
    Moved { site: MoveSite },
}

impl Liveness {
    /// Conservative join: the more moved side wins.
    fn join(self, other: Liveness) -> Liveness {
        match (self, other) {
            (Liveness::Alive, Liveness::Alive) => Liveness::Alive,
            (Liveness::Moved { site }, _) | (_, Liveness::Moved { site }) => {
                Liveness::Moved { site }
            }
            (Liveness::PartlyMoved { site }, _) | (_, Liveness::PartlyMoved { site }) => {
                Liveness::PartlyMoved { site }
            }
        }
    }
}

/// How a value left, in the words the label at the move uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MovedBy {
    TheProgram,
    ACall,
    /// Taken out for a call, lent to it and not yet restored (RFC-0041).
    LentToACall,
}

impl MovedBy {
    fn words(self) -> &'static str {
        match self {
            MovedBy::TheProgram => "moved here",
            MovedBy::ACall => "moved into this call",
            MovedBy::LentToACall => "lent to the call here",
        }
    }
}

/// Where a place left its storage, and what took it. The span is the second
/// place the refusal points at; a synthesized instruction has `Span::ZERO`
/// and the refusal then names one place only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoveSite {
    span: Span,
    by: MovedBy,
}

impl MoveSite {
    fn label(self) -> Vec<Label> {
        match self.span == Span::ZERO {
            true => Vec::new(),
            false => vec![Label::at(self.span, self.by.words())],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MovedPart {
    path: Vec<PathSeg>,
    site: MoveSite,
}

/// Liveness of a storage slot.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StorageLiveness {
    Alive,
    PartlyMoved { parts: Vec<MovedPart> },
    Moved { site: MoveSite },
}

/// How an instruction touches a place inside a storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Touch {
    Read,
    Store,
    /// The store that puts back a place taken out for a call.
    Restore,
}

/// The place a `Ref`, a `Take` or an `Assign` names: a storage and a path
/// into it.
#[derive(Debug, Clone, Copy)]
struct Place<'a> {
    target: &'a RefTarget,
    path: &'a [PathSeg],
}

fn is_prefix(outer: &[PathSeg], inner: &[PathSeg]) -> bool {
    outer.len() <= inner.len() && outer.iter().zip(inner).all(|(a, b)| a == b)
}

/// A store gives the place a value again, so a moved place at the place
/// stored into -- or under it -- does not refuse the store; only one strictly
/// above it does, since a store cannot reach through a place that is gone.
/// A place taken out for a call is the call's until its restore, so a store
/// overlapping it is refused as a read is, and the restore stores into it
/// as a store into a moved place does.
fn refuses(moved: &[PathSeg], by: MovedBy, touched: &[PathSeg], touch: Touch) -> bool {
    let overlaps = is_prefix(moved, touched) || is_prefix(touched, moved);
    match (touch, by) {
        (Touch::Read, _) | (Touch::Store, MovedBy::LentToACall) => overlaps,
        (Touch::Store | Touch::Restore, _) => {
            moved.len() < touched.len() && is_prefix(moved, touched)
        }
    }
}

impl StorageLiveness {
    fn partly(parts: Vec<MovedPart>) -> StorageLiveness {
        if parts.is_empty() {
            StorageLiveness::Alive
        } else {
            StorageLiveness::PartlyMoved { parts }
        }
    }

    fn refusal(&self, touched: &[PathSeg], touch: Touch) -> Option<MoveSite> {
        match self {
            StorageLiveness::Alive => None,
            StorageLiveness::Moved { site } => {
                refuses(&[], site.by, touched, touch).then_some(*site)
            }
            StorageLiveness::PartlyMoved { parts } => parts
                .iter()
                .find(|part| refuses(&part.path, part.site.by, touched, touch))
                .map(|part| part.site),
        }
    }

    fn moved(self, path: &[PathSeg], site: MoveSite) -> StorageLiveness {
        if path.is_empty() {
            return StorageLiveness::Moved { site };
        }
        let mut parts = match self {
            StorageLiveness::Moved { .. } => return self,
            StorageLiveness::Alive => Vec::new(),
            StorageLiveness::PartlyMoved { parts } => parts,
        };
        if !parts.iter().any(|part| part.path == path) {
            parts.push(MovedPart {
                path: path.to_vec(),
                site,
            });
        }
        StorageLiveness::partly(parts)
    }

    /// A store revives the place it stores into and everything under it,
    /// except a place taken out for a call, which only its restore revives.
    fn stored(self, path: &[PathSeg], touch: Touch) -> StorageLiveness {
        let revives = |site: &MoveSite| touch == Touch::Restore || site.by != MovedBy::LentToACall;
        match self {
            StorageLiveness::Moved { site } if path.is_empty() && revives(&site) => {
                StorageLiveness::Alive
            }
            StorageLiveness::PartlyMoved { parts } => StorageLiveness::partly(
                parts
                    .into_iter()
                    .filter(|part| !(is_prefix(path, &part.path) && revives(&part.site)))
                    .collect(),
            ),
            StorageLiveness::Alive | StorageLiveness::Moved { .. } => self,
        }
    }

    /// Conservative join: the more moved side wins, and two partly moved
    /// sides meet in the union of their moved places. `MoveState::join_from`
    /// detects a fixpoint by comparing the result against `self`, so the
    /// union keeps `self`'s order and `self`'s site for a place both sides
    /// hold.
    fn join(self, other: &StorageLiveness) -> StorageLiveness {
        match (self, other) {
            (StorageLiveness::Moved { site }, _) => StorageLiveness::Moved { site },
            (_, StorageLiveness::Moved { site }) => StorageLiveness::Moved { site: *site },
            (StorageLiveness::Alive, StorageLiveness::Alive) => StorageLiveness::Alive,
            (StorageLiveness::Alive, partly @ StorageLiveness::PartlyMoved { .. }) => {
                partly.clone()
            }
            (partly, StorageLiveness::Alive) => partly,
            (
                StorageLiveness::PartlyMoved { mut parts },
                StorageLiveness::PartlyMoved { parts: other },
            ) => {
                for part in other {
                    if !parts.iter().any(|kept| kept.path == part.path) {
                        parts.push(part.clone());
                    }
                }
                StorageLiveness::partly(parts)
            }
        }
    }
}

/// Tracks move state for both ValueIds and $variables.
#[derive(Debug, Clone)]
struct MoveState {
    values: FxHashMap<ValueId, Liveness>,
    /// Variable/param liveness, keyed by storage slot ValueId.
    vars: FxHashMap<ValueId, StorageLiveness>,
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

    fn get_var(&self, slot: ValueId) -> StorageLiveness {
        self.vars
            .get(&slot)
            .cloned()
            .unwrap_or(StorageLiveness::Alive)
    }

    fn set_var(&mut self, slot: ValueId, liveness: StorageLiveness) {
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
        for (&name, liveness) in &other.vars {
            use std::collections::hash_map::Entry;
            match self.vars.entry(name) {
                Entry::Vacant(e) => {
                    e.insert(liveness.clone());
                    changed = true;
                }
                Entry::Occupied(mut e) => {
                    let joined = e.get().clone().join(liveness);
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
    let debug = &body.debug;
    let cfg = promote(body.clone());
    if cfg.blocks.is_empty() {
        return;
    }

    // `Commit` is emitted by lowering alone -- RFC-0025 puts the page ops at
    // entry, exit, around a call and around a spawn, and nowhere else -- so a
    // `Take` whose value a `Commit` consumes is a write-back to the page and
    // not a use the source wrote.
    let commit_of: FxHashMap<ValueId, Astr> = cfg
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.kind {
            InstKind::Commit { context, value, .. } => Some((*value, context.name)),
            _ => None,
        })
        .collect();

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
                &commit_of,
                debug,
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
                    if propagate_state(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        &mut block_entry[target_idx.0],
                    ) {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            }
            | Terminator::Diamond {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
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
                        if propagate_state(
                            &block_exit[idx.0],
                            &cfg.blocks[target_idx.0].params,
                            &mut block_entry[target_idx.0],
                        ) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            // A `For` leaves through two edges. The body's parameters are
            // the terminator's own (RFC-0089 rule 1), so that edge passes no
            // argument; the exit takes the loop's carried values, after the
            // trip count where the edge defines one (RFC-0057 rule 9).
            Terminator::For {
                stages,
                exit,
                exit_trip,
                exit_args,
                ..
            } => {
                let body = stages.body();
                for (label, args) in [(body, &[][..]), (*exit, &exit_args[..])] {
                    if let Some(&target_idx) = cfg.label_to_block.get(&label) {
                        let params = &cfg.blocks[target_idx.0].params;
                        let taking: &[ValueId] = match label == body {
                            true => &[],
                            false => exit_trip.carried_params(params),
                        };
                        propagate_args(
                            scope,
                            &block_exit[idx.0],
                            args,
                            taking,
                            &cfg.val_types,
                            errors,
                            &mut block_entry[target_idx.0],
                        );
                        if propagate_state(
                            &block_exit[idx.0],
                            params,
                            &mut block_entry[target_idx.0],
                        ) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            // Every arm of a `Switch` is an edge that forwards its own
            // arguments, exactly as a `JumpIf`'s two are.
            Terminator::Switch { arms, default, .. } => {
                let edges = arms
                    .iter()
                    .map(|(_, label, args)| (label, args))
                    .chain(default.iter().map(|(label, args)| (label, args)));
                for (label, args) in edges {
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
                        if propagate_state(
                            &block_exit[idx.0],
                            &cfg.blocks[target_idx.0].params,
                            &mut block_entry[target_idx.0],
                        ) {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n
                    && propagate_state(
                        &block_exit[idx.0],
                        &cfg.blocks[next].params,
                        &mut block_entry[next],
                    )
                {
                    worklist.push_back(BlockIdx(next));
                }
            }
            Terminator::Return { value, .. } => {
                let span = block.insts.last().map(|i| i.span).unwrap_or(Span::ZERO);
                let mut state = block_exit[idx.0].clone();
                try_consume_value(
                    scope,
                    block.insts.len(),
                    MoveSite {
                        span,
                        by: MovedBy::TheProgram,
                    },
                    *value,
                    &cfg.val_types,
                    debug,
                    &mut state,
                    errors,
                );
            }
            Terminator::Diverge => {}
        }
    }
}

/// Carry the predecessor's exit state into the successor's entry. The
/// successor's params are defined by the jump's arguments, so their state
/// comes from `propagate_args` alone and not from the predecessor.
fn propagate_state(source: &MoveState, params: &[ValueId], target: &mut MoveState) -> bool {
    let mut carried = source.clone();
    for p in params {
        carried.values.remove(p);
    }
    target.join_from(&carried)
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
    // Map arg liveness -> param liveness.
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
    site: MoveSite,
    id: ValueId,
    val_types: &FxHashMap<ValueId, Ty>,
    debug: &DebugInfo,
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
    if let Some(Liveness::Moved { site: moved } | Liveness::PartlyMoved { site: moved }) =
        state.get_value(id)
    {
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span: site.span,
            kind: ValidationErrorKind::UseAfterMove {
                value_id: id.to_raw() as u32,
                ty: ty.clone(),
                origin: debug.get(id).cloned(),
                labels: moved.label(),
            },
        });
        return true;
    }

    // Mark as moved
    state.set_value(id, Liveness::Moved { site });
    true
}

/// RFC-0018 rule 2.
pub(crate) fn moves_out(ty: &Ty) -> bool {
    ty.copies() == Some(false)
}

/// The register this instruction leaves owning nothing, and the one
/// statement of it that the move check and drop insertion both read: what
/// one calls "consumed here" the other calls "no drop after here", and a
/// register with two answers is a register released twice or not at all.
///
/// An option has no storage of its own — `Some(v)` is `v` and a `None` is a
/// word counting the `Some`s around it (RFC-0039, `docs/runtime-value.md`) —
/// so a take that reaches a payload through nothing but options empties the
/// whole storage. A `String` payload is copied out rather than moved under
/// RFC-0018 rule 2, which is what `moves_out` asks. An unwrap empties its source in
/// every variant form: `variant::unwrap_option`, `unwrap_result` and
/// `unwrap_variant` each open with `machine.take(op.b)` and hand the payload
/// on, so the box a `Result` or an enum carried is gone with them.
pub(crate) fn emptied_by(kind: &InstKind, val_types: &FxHashMap<ValueId, Ty>) -> Option<ValueId> {
    match kind {
        InstKind::Take { target, path, .. } => {
            let storage = crate::analysis::inst_info::storage(target)?;
            let ty = val_types.get(&storage)?;
            under_options(ty, path)
                .is_some_and(moves_out)
                .then_some(storage)
        }
        InstKind::UnwrapVariant { src, .. } => Some(*src),
        _ => None,
    }
}

/// The type a path of payload steps reaches while every type it steps
/// through is an option.
fn under_options<'a>(ty: &'a Ty, path: &[PathSeg]) -> Option<&'a Ty> {
    let mut at = ty;
    for seg in path {
        let (PathSeg::Payload, Ty::Option(payload)) = (seg, at) else {
            return None;
        };
        at = payload;
    }
    Some(at)
}

fn extract_part(
    scope: &str,
    inst_idx: usize,
    site: MoveSite,
    container: ValueId,
    dst: ValueId,
    val_types: &FxHashMap<ValueId, Ty>,
    debug: &DebugInfo,
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(ty) = val_types.get(&dst)
        && moves_out(ty)
    {
        match state.get_value(container) {
            Some(Liveness::Moved { site: moved }) => errors.push(ValidationError {
                scope: scope.to_string(),
                inst_index: inst_idx,
                span: site.span,
                kind: ValidationErrorKind::UseAfterMove {
                    value_id: container.to_raw() as u32,
                    ty: ty.clone(),
                    origin: debug.get(container).cloned(),
                    labels: moved.label(),
                },
            }),
            _ => state.set_value(container, Liveness::PartlyMoved { site }),
        }
    }
    state.set_value(dst, Liveness::Alive);
}

/// A place inside a storage is touched only while it is alive: by a `Take` of
/// it, by a `Ref` to it, by a store into it. A place named through a
/// reference is not touched here -- the reference is a value operand, checked
/// where it is used.
///
/// `commits_to` names the context this touch writes back to, for the `Take`
/// that lowering emits in front of a `Commit`; the refusal of such a touch is
/// the program's failure to assign the context again, and is stated at the
/// move rather than at the write-back the program never wrote.
fn touch_storage(
    scope: &str,
    inst_idx: usize,
    span: Span,
    place: Place<'_>,
    touch: Touch,
    commits_to: Option<Astr>,
    val_types: &FxHashMap<ValueId, Ty>,
    debug: &DebugInfo,
    state: &MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let (RefTarget::Var(slot) | RefTarget::Param(slot)) = place.target else {
        return;
    };
    let Some(site) = state.get_var(*slot).refusal(place.path, touch) else {
        return;
    };
    if site.by == MovedBy::LentToACall {
        let touch = match touch {
            Touch::Read => ConflictTouch::Read,
            Touch::Store | Touch::Restore => ConflictTouch::Written,
        };
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span,
            kind: ValidationErrorKind::LentToCall {
                storage: debug.get(*slot).cloned(),
                touch,
                labels: site.label(),
            },
        });
        return;
    }
    if let Some(context) = commits_to {
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span: site.span,
            kind: ValidationErrorKind::ContextMovedOut { context },
        });
        return;
    }
    let Some(ty) = val_types.get(slot) else {
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: inst_idx,
            span,
            kind: ValidationErrorKind::MissingType {
                value_id: slot.to_raw() as u32,
            },
        });
        return;
    };
    errors.push(ValidationError {
        scope: scope.to_string(),
        inst_index: inst_idx,
        span,
        kind: ValidationErrorKind::UseAfterMove {
            value_id: slot.to_raw() as u32,
            ty: ty.clone(),
            origin: debug.get(*slot).cloned(),
            labels: site.label(),
        },
    });
}

/// Process a single instruction: check uses and update move state.
fn process_inst(
    scope: &str,
    inst_idx: usize,
    inst: &Inst,
    val_types: &FxHashMap<ValueId, Ty>,
    commit_of: &FxHashMap<ValueId, Astr>,
    debug: &DebugInfo,
    state: &mut MoveState,
    errors: &mut Vec<ValidationError>,
) {
    let span = inst.span;
    let plain = MoveSite {
        span,
        by: MovedBy::TheProgram,
    };
    let into_call = MoveSite {
        span,
        by: MovedBy::ACall,
    };

    match &inst.kind {
        // === No operands / define only ===
        InstKind::Merge { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::Const { dst, .. }
        | InstKind::ConstStr { dst, .. }
        | InstKind::Poison { dst }
        | InstKind::Undef { dst } => {
            state.set_value(*dst, Liveness::Alive);
        }
        // A reference names the place, so it reads it.
        InstKind::Ref {
            dst, target, path, ..
        } => {
            let place = Place { target, path };
            touch_storage(
                scope,
                inst_idx,
                span,
                place,
                Touch::Read,
                None,
                val_types,
                debug,
                state,
                errors,
            );
            state.set_value(*dst, Liveness::Alive);
        }
        // Take: a move-only value leaves the place it was read from; a second
        // take of a place overlapping it is a use after move. A take-out
        // leaves its place whatever its type, a word's included.
        InstKind::Take {
            dst,
            target,
            path,
            taken_out,
        } => {
            let place = Place { target, path };
            touch_storage(
                scope,
                inst_idx,
                span,
                place,
                Touch::Read,
                commit_of.get(dst).copied(),
                val_types,
                debug,
                state,
                errors,
            );
            if let RefTarget::Var(slot) | RefTarget::Param(slot) = target
                && let Some(ty) = val_types.get(dst)
                && (moves_out(ty) || *taken_out)
            {
                let site = MoveSite {
                    span,
                    by: match taken_out {
                        true => MovedBy::LentToACall,
                        false => MovedBy::TheProgram,
                    },
                };
                let whole: &[PathSeg] = &[];
                let moved = match emptied_by(&inst.kind, val_types) == Some(*slot) {
                    true => whole,
                    false => path,
                };
                state.set_var(*slot, state.get_var(*slot).moved(moved, site));
            }
            state.set_value(*dst, Liveness::Alive);
        }
        // Assign consumes the value and gives the place it stores into a
        // value again. A whole store revives the storage as the slot value a
        // Drop consumes, too.
        InstKind::Assign {
            target,
            path,
            value,
            restores,
        } => {
            try_consume_value(
                scope, inst_idx, plain, *value, val_types, debug, state, errors,
            );
            let place = Place { target, path };
            let touch = match restores {
                true => Touch::Restore,
                false => Touch::Store,
            };
            touch_storage(
                scope, inst_idx, span, place, touch, None, val_types, debug, state, errors,
            );
            if let RefTarget::Var(slot) | RefTarget::Param(slot) = target {
                state.set_var(*slot, state.get_var(*slot).stored(path, touch));
                if path.is_empty() {
                    state.set_value(*slot, Liveness::Alive);
                }
            }
        }
        InstKind::Fetch { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::Commit { value, .. } => {
            try_consume_value(
                scope, inst_idx, plain, *value, val_types, debug, state, errors,
            );
        }
        InstKind::BlockLabel { params, .. } => {
            for p in params {
                state.values.insert(*p, Liveness::Alive);
            }
        }
        InstKind::Nop => {}

        // === Consuming operations (move operands) ===
        InstKind::Return { value, .. } => {
            try_consume_value(
                scope, inst_idx, plain, *value, val_types, debug, state, errors,
            );
        }
        InstKind::Diverge => {}
        InstKind::Drop { src } => {
            try_consume_value(
                scope, inst_idx, plain, *src, val_types, debug, state, errors,
            );
        }

        // Functions
        InstKind::LoadFunction { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        // Calls - all args are consumed; indirect callee is also consumed
        InstKind::FunctionCall {
            dst, callee, args, ..
        } => {
            if let Callee::Indirect(closure) = callee {
                try_consume_value(
                    scope, inst_idx, into_call, *closure, val_types, debug, state, errors,
                );
            }
            for arg in args {
                try_consume_value(
                    scope, inst_idx, into_call, *arg, val_types, debug, state, errors,
                );
            }
            state.set_value(*dst, Liveness::Alive);
        }

        // Constructors - elements are consumed
        InstKind::StringEq { dst, .. }
        | InstKind::StringClone { dst, .. }
        | InstKind::StructuralEq { dst, .. }
        | InstKind::StructuralClone { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::StringConcat { dst, parts } => {
            for p in parts {
                try_consume_value(scope, inst_idx, plain, *p, val_types, debug, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ArrayBegin { dst, .. } => state.set_value(*dst, Liveness::Alive),
        InstKind::ArrayPush { dst, array, value } => {
            for moved in [array, value] {
                try_consume_value(
                    scope, inst_idx, plain, *moved, val_types, debug, state, errors,
                );
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeObject { dst, fields } => {
            for (_, v) in fields {
                try_consume_value(scope, inst_idx, plain, *v, val_types, debug, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeTuple { dst, elements } => {
            for e in elements {
                try_consume_value(scope, inst_idx, plain, *e, val_types, debug, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeClosure { dst, captures, .. } => {
            for cap in captures {
                try_consume_value(
                    scope, inst_idx, plain, *cap, val_types, debug, state, errors,
                );
            }
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::MakeVariant { dst, payload, .. } => {
            if let Some(p) = payload {
                try_consume_value(scope, inst_idx, plain, *p, val_types, debug, state, errors);
            }
            state.set_value(*dst, Liveness::Alive);
        }

        // === Non-consuming operations (borrow operands) ===
        // These read the value but don't take ownership.
        InstKind::FieldGet { dst, object, .. } => {
            extract_part(
                scope, inst_idx, plain, *object, *dst, val_types, debug, state, errors,
            );
        }
        InstKind::FieldSet {
            dst,
            object: _,
            value,
            ..
        } => {
            try_consume_value(
                scope, inst_idx, plain, *value, val_types, debug, state, errors,
            );
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::ObjectGet { dst, object, .. } => {
            extract_part(
                scope, inst_idx, plain, *object, *dst, val_types, debug, state, errors,
            );
        }
        InstKind::TupleIndex { dst, tuple, .. } => {
            extract_part(
                scope, inst_idx, plain, *tuple, *dst, val_types, debug, state, errors,
            );
        }
        InstKind::ArrayIndex { dst, array, .. } => {
            extract_part(
                scope, inst_idx, plain, *array, *dst, val_types, debug, state, errors,
            );
        }

        // Slices (RFC-0047). A slice borrows its container and an `Index`
        // borrows the slice, as a `Ref` borrows a place; `IndexSet` is the
        // one that moves, and what it moves is the element written.
        InstKind::AsSlice { dst, .. } | InstKind::Index { dst, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::IndexSet { value, .. } => {
            try_consume_value(
                scope, inst_idx, plain, *value, val_types, debug, state, errors,
            );
        }
        InstKind::StringAppend { part, .. } => {
            try_consume_value(
                scope, inst_idx, plain, *part, val_types, debug, state, errors,
            );
        }
        InstKind::UnwrapVariant { dst, src } => {
            if emptied_by(&inst.kind, val_types) == Some(*src) {
                try_consume_value(
                    scope, inst_idx, plain, *src, val_types, debug, state, errors,
                );
            }
            state.set_value(*dst, Liveness::Alive);
        }

        // Pattern tests - read-only, always produce Bool
        InstKind::TestLiteral { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestObjectKey { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::TestVariant { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }

        // Arithmetic - operands are always pure scalars, no move
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
        InstKind::Cast { dst, src: _, .. } => {
            state.set_value(*dst, Liveness::Alive);
        }
        InstKind::Check { .. } | InstKind::CheckSteps { .. } => {}

        // Spawn - consumes args (and indirect callee), defines dst
        InstKind::Spawn {
            dst, callee, args, ..
        } => {
            if let Callee::Indirect(closure) = callee {
                try_consume_value(
                    scope, inst_idx, into_call, *closure, val_types, debug, state, errors,
                );
            }
            for arg in args {
                try_consume_value(
                    scope, inst_idx, into_call, *arg, val_types, debug, state, errors,
                );
            }
            state.set_value(*dst, Liveness::Alive);
        }
        // Eval - consumes Handle (move-only), defines dst
        InstKind::Eval { dst, src, .. } => {
            try_consume_value(
                scope, inst_idx, plain, *src, val_types, debug, state, errors,
            );
            state.set_value(*dst, Liveness::Alive);
        }

        // Control flow - handled at block level
        InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Diamond { .. }
        | InstKind::Switch { .. }
        | InstKind::For { .. } => {}
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
    use crate::ty::{Param, TypeArg};
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
            declared_params: 0,
            main: MirBody {
                demoted_diamonds: Default::default(),
                insts,
                val_types,
                params: Vec::new(),
                captures: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 10,
                order_param: None,
                task: crate::ty::Task::Sync,
            },
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
            flows: crate::ty::Flows::Every,
        }
    }

    // -- is_move_only tests --

    /// A user-defined type with an identity: a source of its own, so move-only.
    fn test_user_defined() -> Ty {
        let i = Interner::new();
        Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestType")),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![crate::ty::IdentityTerm::Known(
                <crate::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
            )],
            region_params: 0,
        }
    }

    #[test]
    fn an_option_moves_exactly_when_its_payload_does() {
        let option = |ty| Ty::Option(Box::new(ty));
        assert_eq!(is_move_only(&option(Ty::Float)), Some(false));
        assert_eq!(is_move_only(&option(Ty::String)), Some(true));
        assert_eq!(is_move_only(&option(option(Ty::Float))), Some(false));
        assert_eq!(
            is_move_only(&option(option(test_user_defined()))),
            Some(true)
        );
    }

    #[test]
    fn words_copy_and_owners_move() {
        assert_eq!(is_move_only(&Ty::I64), Some(false));
        assert_eq!(is_move_only(&Ty::Bool), Some(false));
        assert_eq!(
            is_move_only(&Ty::Ref(
                crate::ty::Mutability::Shared,
                Box::new(TypeArg::uniform(Ty::String))
            )),
            Some(false)
        );
        assert_eq!(is_move_only(&Ty::String), Some(true));
        assert_eq!(
            is_move_only(&Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3))),
            Some(true)
        );
    }

    #[test]
    fn user_defined_without_identity_moves() {
        let i = Interner::new();
        let plain = Ty::UserDefined {
            id: QualifiedRef::root(i.intern("Plain")),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        assert_eq!(is_move_only(&plain), Some(true));
    }

    #[test]
    fn user_defined_is_move() {
        assert_eq!(is_move_only(&test_user_defined()), Some(true));
    }

    #[test]
    fn tuple_with_user_defined_is_move() {
        let ty = Ty::Tuple(vec![Ty::I64, test_user_defined()]);
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_with_user_defined_capture_is_move() {
        let ty = Ty::Fn {
            params: vec![param(Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![test_user_defined()],
            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::Every.into(),
        };
        assert_eq!(is_move_only(&ty), Some(true));
    }

    #[test]
    fn fn_moves() {
        let ty = Ty::Fn {
            params: vec![param(Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![Ty::I64, Ty::String],
            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::Every.into(),
        };
        assert_eq!(is_move_only(&ty), Some(true));
    }

    // -- move check integration tests --

    #[test]
    fn error_for_user_defined_reuse() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        // v0 = UserDefined (move-only), used twice -> ERROR
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, test_user_defined());
        val_types.insert(
            v1,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );
        val_types.insert(
            v2,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );

        let call = |dst, span| Inst {
            span,
            kind: InstKind::FunctionCall {
                dst,
                callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                callee_ty: Ty::error(),
                args: vec![v0],
                order: None,
            },
        };
        let first = Span { start: 0, end: 7 };
        let module = make_module(
            vec![call(v1, first), call(v2, Span { start: 9, end: 16 })],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "UserDefined reuse should be rejected");
        assert!(matches!(
            errors[0].kind,
            ValidationErrorKind::UseAfterMove { .. }
        ));
        assert_eq!(
            errors[0].labels(),
            [Label::at(first, "moved into this call")]
        );
    }

    #[test]
    fn no_error_for_single_use_user_defined() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        // v0 = UserDefined, used once -> OK
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, test_user_defined());
        val_types.insert(
            v1,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );

        let module = make_module(
            vec![inst(InstKind::FunctionCall {
                dst: v1,
                callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                callee_ty: Ty::error(),
                args: vec![v0],
                order: None,
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
        let a = vf.next(); // storage slot for variable "a"
        // $a = move-only (v0), Take (v1) -> moved, $a = new value (v2) -> alive, Take (v3) -> OK
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(a, move_ty.clone());
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
        val_types.insert(v3, move_ty.clone());
        val_types.insert(
            v4,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );
        val_types.insert(
            v5,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );

        let module = make_module(
            vec![
                // $a = v0 (move-only)
                inst(InstKind::Assign {
                    target: RefTarget::Var(a),
                    path: vec![],
                    value: v0,
                    restores: false,
                }),
                // v1 = $a -> moves $a
                inst(InstKind::Take {
                    dst: v1,
                    target: RefTarget::Var(a),
                    path: vec![],
                    taken_out: false,
                }),
                // use v1
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    callee_ty: Ty::error(),
                    args: vec![v1],
                    order: None,
                }),
                // $a = v2 (new value) -> revives $a
                inst(InstKind::Assign {
                    target: RefTarget::Var(a),
                    path: vec![],
                    value: v2,
                    restores: false,
                }),
                // v3 = $a -> OK (new value)
                inst(InstKind::Take {
                    dst: v3,
                    target: RefTarget::Var(a),
                    path: vec![],
                    taken_out: false,
                }),
                // use v3
                inst(InstKind::FunctionCall {
                    dst: v5,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    callee_ty: Ty::error(),
                    args: vec![v3],
                    order: None,
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
        let a = vf.next(); // storage slot for variable "a"
        // $a = move-only, Take -> moved, Take again -> ERROR
        let mut val_types = FxHashMap::default();
        let move_ty = test_user_defined();
        val_types.insert(a, move_ty.clone());
        val_types.insert(v0, move_ty.clone());
        val_types.insert(v1, move_ty.clone());
        val_types.insert(v2, move_ty.clone());
        val_types.insert(
            v3,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );
        val_types.insert(
            v4,
            Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
        );

        let module = make_module(
            vec![
                inst(InstKind::Assign {
                    target: RefTarget::Var(a),
                    path: vec![],
                    value: v0,
                    restores: false,
                }),
                inst(InstKind::Take {
                    dst: v1,
                    target: RefTarget::Var(a),
                    path: vec![],
                    taken_out: false,
                }),
                inst(InstKind::FunctionCall {
                    dst: v3,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    callee_ty: Ty::error(),
                    args: vec![v1],
                    order: None,
                }),
                // Second take - $a already moved
                inst(InstKind::Take {
                    dst: v2,
                    target: RefTarget::Var(a),
                    path: vec![],
                    taken_out: false,
                }),
                inst(InstKind::FunctionCall {
                    dst: v4,
                    callee: Callee::Direct(QualifiedRef::root(Interner::new().intern("test"))),
                    callee_ty: Ty::error(),
                    args: vec![v2],
                    order: None,
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "use after move of $var should be rejected");
    }

    /// The shape lowering emits for a context whose value the source took
    /// and never gave back: the slot is filled, taken at the source's move,
    /// then taken again by the `Commit` that ends the run. The refusal of
    /// that second take is stated at the first, with the source's span.
    #[test]
    fn a_context_committed_after_its_move_is_named_at_the_move() {
        let i = Interner::new();
        let mut vf = LocalFactory::<ValueId>::new();
        let slot = vf.next();
        let filled = vf.next();
        let taken = vf.next();
        let committed = vf.next();
        let move_ty = test_user_defined();
        let mut val_types = FxHashMap::default();
        for v in [slot, filled, taken, committed] {
            val_types.insert(v, move_ty.clone());
        }
        let query = QualifiedRef::root(i.intern("query"));
        let moved_at = Span { start: 8, end: 14 };
        let whole = Span { start: 0, end: 40 };

        let module = make_module(
            vec![
                Inst {
                    span: whole,
                    kind: InstKind::Assign {
                        target: RefTarget::Var(slot),
                        path: vec![],
                        value: filled,
                        restores: false,
                    },
                },
                Inst {
                    span: moved_at,
                    kind: InstKind::Take {
                        dst: taken,
                        target: RefTarget::Var(slot),
                        path: vec![],
                        taken_out: false,
                    },
                },
                Inst {
                    span: whole,
                    kind: InstKind::Take {
                        dst: committed,
                        target: RefTarget::Var(slot),
                        path: vec![],
                        taken_out: false,
                    },
                },
                Inst {
                    span: whole,
                    kind: InstKind::Commit {
                        context: query,
                        value: committed,
                        wrote: true,
                    },
                },
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].span, moved_at);
        assert!(
            matches!(
                errors[0].kind,
                ValidationErrorKind::ContextMovedOut { context }
                    if context == query.name
            ),
            "{:?}",
            errors[0].kind
        );
    }

    // -- a storage's parts --

    /// `o` holds two move-only fields, `v` and `w`. Each test drives the
    /// storage through takes and stores of those places and reads the errors.
    struct Storage {
        slot: ValueId,
        vf: LocalFactory<ValueId>,
        val_types: FxHashMap<ValueId, Ty>,
        insts: Vec<Inst>,
        v: PathSeg,
        w: PathSeg,
    }

    impl Storage {
        fn new() -> Self {
            let interner = Interner::new();
            let mut vf = LocalFactory::<ValueId>::new();
            let slot = vf.next();
            let mut val_types = FxHashMap::default();
            val_types.insert(slot, test_user_defined());
            Self {
                slot,
                vf,
                val_types,
                insts: Vec::new(),
                v: PathSeg::Field(interner.intern("v")),
                w: PathSeg::Field(interner.intern("w")),
            }
        }

        /// The span of the instruction at index `at`, which is the place a
        /// label at that instruction names.
        fn span_at(at: usize) -> Span {
            Span {
                start: at * 10,
                end: at * 10 + 1,
            }
        }

        fn push(&mut self, kind: InstKind) {
            let span = Self::span_at(self.insts.len());
            self.insts.push(Inst { span, kind });
        }

        fn take(&mut self, path: &[PathSeg]) {
            self.take_as(path, test_user_defined(), false);
        }

        /// The take-out of a place lent to a call, of a value of type `ty`.
        fn take_out(&mut self, path: &[PathSeg], ty: Ty) {
            self.take_as(path, ty, true);
        }

        fn take_as(&mut self, path: &[PathSeg], ty: Ty, taken_out: bool) {
            let dst = self.vf.next();
            self.val_types.insert(dst, ty);
            self.push(InstKind::Take {
                dst,
                target: RefTarget::Var(self.slot),
                path: path.to_vec(),
                taken_out,
            });
        }

        fn assign(&mut self, path: &[PathSeg]) {
            self.assign_as(path, false);
        }

        /// The store that restores a place taken out for a call.
        fn restore(&mut self, path: &[PathSeg]) {
            self.assign_as(path, true);
        }

        fn assign_as(&mut self, path: &[PathSeg], restores: bool) {
            let value = self.vf.next();
            self.val_types.insert(value, test_user_defined());
            self.push(InstKind::Assign {
                target: RefTarget::Var(self.slot),
                path: path.to_vec(),
                value,
                restores,
            });
        }

        fn lend(&mut self, path: &[PathSeg]) {
            let dst = self.vf.next();
            self.val_types.insert(
                dst,
                Ty::Ref(
                    crate::ty::Mutability::Shared,
                    Box::new(TypeArg::uniform(test_user_defined())),
                ),
            );
            self.push(InstKind::Ref {
                dst,
                target: RefTarget::Var(self.slot),
                path: path.to_vec(),
                mutability: crate::ty::Mutability::Shared,
            });
        }

        fn errors(self) -> Vec<ValidationError> {
            check_moves(&make_module(self.insts, self.val_types))
        }
    }

    #[test]
    fn a_take_of_a_part_leaves_its_siblings_alive() {
        let mut o = Storage::new();
        o.assign(&[]);
        let (v, w) = (o.v, o.w);
        o.take(&[v]);
        o.take(&[w]);
        assert!(o.errors().is_empty());
    }

    #[test]
    fn a_take_of_a_moved_part_is_refused() {
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take(&[v]);
        o.take(&[v]);
        let errors = o.errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(
            errors[0].labels(),
            [Label::at(Storage::span_at(1), "moved here")]
        );
    }

    #[test]
    fn a_store_into_a_moved_part_revives_it() {
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take(&[v]);
        o.assign(&[v]);
        o.lend(&[v]);
        o.take(&[v]);
        assert!(o.errors().is_empty());
    }

    #[test]
    fn a_reference_to_the_whole_of_a_partly_moved_storage_is_refused() {
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take(&[v]);
        o.lend(&[]);
        let errors = o.errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(
            errors[0].labels(),
            [Label::at(Storage::span_at(1), "moved here")]
        );
    }

    #[test]
    fn a_store_into_a_part_of_a_wholly_moved_storage_is_refused() {
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take(&[]);
        o.assign(&[v]);
        let errors = o.errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(
            errors[0].labels(),
            [Label::at(Storage::span_at(1), "moved here")]
        );
    }

    /// RFC-0041: a store into a place taken out for a call, or into one
    /// overlapping it, is refused and revives nothing, so the use after it
    /// is refused too; the restore is the store that revives it.
    #[test]
    fn a_place_taken_out_for_a_call_is_revived_by_its_restore_alone() {
        let lent_here = [Label::at(Storage::span_at(1), "lent to the call here")];
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take_out(&[v], test_user_defined());
        o.assign(&[v]);
        o.assign(&[]);
        o.lend(&[v]);
        o.restore(&[v]);
        o.lend(&[v]);
        let errors = o.errors();
        let touched: Vec<_> = errors
            .iter()
            .map(|error| match &error.kind {
                ValidationErrorKind::LentToCall { touch, labels, .. } => {
                    assert_eq!(labels, &lent_here);
                    (error.span, *touch)
                }
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            touched,
            [
                (Storage::span_at(2), ConflictTouch::Written),
                (Storage::span_at(3), ConflictTouch::Written),
                (Storage::span_at(4), ConflictTouch::Read),
            ]
        );
    }

    /// RFC-0041: a word taken out for a call is taken out as any value is,
    /// though its take copies.
    #[test]
    fn a_word_taken_out_for_a_call_is_refused_until_its_restore() {
        let mut o = Storage::new();
        o.assign(&[]);
        let v = o.v;
        o.take_out(&[v], Ty::I64);
        o.take(&[v]);
        o.restore(&[v]);
        o.lend(&[v]);
        let errors = o.errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            matches!(
                errors[0].kind,
                ValidationErrorKind::LentToCall {
                    touch: ConflictTouch::Read,
                    ..
                }
            ),
            "{:?}",
            errors[0].kind
        );
    }

    #[test]
    fn ty_param_skipped() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        // v0 = Ty::Error (unresolved), used twice -> no error (analysis mode)
        let mut val_types = FxHashMap::default();
        val_types.insert(v0, Ty::error());

        let module = make_module(
            vec![
                inst(InstKind::Return {
                    value: v0,
                    order: None,
                }),
                inst(InstKind::Return {
                    value: v0,
                    order: None,
                }),
            ],
            val_types,
        );

        let errors = check_moves(&module);
        assert!(errors.is_empty(), "Ty::Error should be skipped");
    }

    // E2E compile pipeline tests have been migrated to acvus-mir-test/tests/e2e.rs
    // (they depend on ExternFn registries which are only available there).
}
