//! Definite-assignment analysis - field-level uninit check.
//!
//! Runs on CfgBody (pre-SSA). Tracks which fields of each named storage
//! (Var, Context, Param) are definitely initialized at each program point.
//! A storage is written by `Assign` and read by `Take` and `Ref`.
//!
//! At function call sites, checks that arguments have all fields required
//! by the callee's parameter type. Required fields come from the instruction's
//! `callee_ty`, NOT from val_types (which may have been widened by unification).

use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::dataflow::{DataflowAnalysis, DataflowState, forward_analysis};
use crate::analysis::domain::SemiLattice;
use crate::cfg::CfgBody;
use crate::error::{DidYouMean, MirError, MirErrorKind, ShownValue};
use crate::ir::{Callee, Inst, InstKind, PathSeg, RefTarget, ValOrigin, ValueId};
use crate::ty::Ty;
use acvus_ast::Span;

// -- Domain ----------------------------------------------------------

/// Per-field init domain.
///
/// Lattice: Init (bottom) -> Uninit (top).
/// join(Init, Uninit) = Uninit - if ANY path leaves a field uninit,
/// it is possibly uninit at the merge point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldInit {
    Init,
    Uninit,
}

impl SemiLattice for FieldInit {
    fn bottom() -> Self {
        FieldInit::Init
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        match (*self, *other) {
            (FieldInit::Init, FieldInit::Uninit) => {
                *self = FieldInit::Uninit;
                true
            }
            _ => false,
        }
    }
}

// -- Errors ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UninitError {
    pub span: Span,
    pub subject: UninitSubject,
    pub uninit_fields: Vec<Astr>,
    /// The field names the subject does carry, so a refusal over a
    /// misspelling can offer the one that was meant.
    pub stored_fields: Vec<Astr>,
}

/// What lacked the fields: a named storage, or a value built in place and
/// passed on without a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UninitSubject {
    Storage(RefTarget),
    Value(ValueId),
}

// -- Pre-pass data ---------------------------------------------------

/// Maps ValueId -> set of field names that the value definitely contains.
type ValueFields = FxHashMap<ValueId, FxHashSet<Astr>>;

fn build_value_fields(cfg: &CfgBody) -> ValueFields {
    let mut value_fields = ValueFields::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::MakeObject { dst, fields } => {
                    let names: FxHashSet<Astr> = fields.iter().map(|(k, _)| *k).collect();
                    value_fields.insert(*dst, names);
                }
                InstKind::FieldSet {
                    dst, object, field, ..
                } => {
                    if let Some(obj_fields) = value_fields.get(object) {
                        let mut fields = obj_fields.clone();
                        fields.insert(*field);
                        value_fields.insert(*dst, fields);
                    }
                }
                _ => {}
            }
        }
    }
    value_fields
}

/// Every (storage, field) the body names: a whole-storage `Take`/`Assign`
/// contributes the fields of its object type, a field path proves its first
/// field exists, and a call contributes the fields its callee requires of a
/// storage passed whole, so a field the body never wrote is tracked where
/// the callee reads it.
fn collect_var_fields(cfg: &CfgBody) -> FxHashMap<RefTarget, FxHashSet<Astr>> {
    let mut target_fields: FxHashMap<RefTarget, FxHashSet<Astr>> = FxHashMap::default();
    let mut note = |target: &RefTarget, path: &[PathSeg], whole: Option<&Ty>| match path.first() {
        Some(PathSeg::Field(field)) => {
            target_fields.entry(*target).or_default().insert(*field);
        }
        _ => {
            if let Some(Ty::Object(fields)) = whole {
                target_fields
                    .entry(*target)
                    .or_default()
                    .extend(fields.keys().copied());
            }
        }
    };
    for block in &cfg.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Take { dst, target, path } => {
                    note(target, path, cfg.val_types.get(dst));
                }
                InstKind::Assign {
                    target,
                    path,
                    value,
                } => {
                    note(target, path, cfg.val_types.get(value));
                }
                InstKind::Ref {
                    dst, target, path, ..
                } => {
                    let inner = match cfg.val_types.get(dst) {
                        Some(Ty::Ref(_, inner)) => Some(&inner.ty),
                        _ => None,
                    };
                    note(target, path, inner);
                }
                _ => {}
            }
        }
    }
    for block in &cfg.blocks {
        for inst in &block.insts {
            let (InstKind::FunctionCall {
                callee_ty, args, ..
            }
            | InstKind::Spawn {
                callee_ty, args, ..
            }) = &inst.kind
            else {
                continue;
            };
            let Ty::Fn { params, .. } = callee_ty else {
                continue;
            };
            for (arg, param) in args.iter().zip(params.iter()) {
                let Ty::Object(required) = &param.ty else {
                    continue;
                };
                let Some(target) = find_arg_source(arg, cfg) else {
                    continue;
                };
                target_fields
                    .entry(target)
                    .or_default()
                    .extend(required.keys().copied());
            }
        }
    }
    target_fields
}

// -- Analysis --------------------------------------------------------

struct InitCheckAnalysis {
    value_fields: ValueFields,
    /// val_types from CfgBody - for fallback field lookup.
    val_types: FxHashMap<ValueId, Ty>,
}

impl DataflowAnalysis for InitCheckAnalysis {
    type Key = (RefTarget, Astr);
    type Domain = FieldInit;

    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<(RefTarget, Astr), FieldInit>) {
        let InstKind::Assign {
            target,
            path,
            value,
        } = &inst.kind
        else {
            return;
        };
        if let Some(PathSeg::Field(field)) = path.first() {
            state.set((*target, *field), FieldInit::Init);
            return;
        }
        let fields: Option<Vec<Astr>> = if let Some(known) = self.value_fields.get(value) {
            Some(known.iter().copied().collect())
        } else if let Some(ty) = self.val_types.get(value) {
            extract_object_fields(ty)
        } else {
            None
        };
        if let Some(fields) = fields {
            for f in fields {
                state.set((*target, f), FieldInit::Init);
            }
        }
    }

    fn propagate_forward(
        &self,
        source_exit: &DataflowState<(RefTarget, Astr), FieldInit>,
        _params: &[ValueId],
        _first: usize,
        _args: &[ValueId],
        target_entry: &mut DataflowState<(RefTarget, Astr), FieldInit>,
    ) -> bool {
        // Storage init state is not SSA - no param/arg mapping. Pure join.
        target_entry.join_from(source_exit)
    }

    fn propagate_backward(
        &self,
        _succ_entry: &DataflowState<(RefTarget, Astr), FieldInit>,
        _succ_params: &[ValueId],
        _first: usize,
        _term_args: &[ValueId],
        _exit_state: &mut DataflowState<(RefTarget, Astr), FieldInit>,
    ) {
        unreachable!("init check is forward-only")
    }
}

// -- Public API ------------------------------------------------------

fn stored_of(var_fields: &FxHashMap<RefTarget, FxHashSet<Astr>>, target: &RefTarget) -> Vec<Astr> {
    var_fields
        .get(target)
        .map(|fields| sorted(fields.iter().copied()))
        .unwrap_or_default()
}

fn sorted<I>(names: I) -> Vec<Astr>
where
    I: Iterator<Item = Astr>,
{
    let mut names: Vec<Astr> = names.collect();
    names.sort();
    names
}

/// Run field-level definite-assignment check on a CfgBody.
pub fn check_init(cfg: &CfgBody) -> Vec<UninitError> {
    let value_fields = build_value_fields(cfg);
    let var_fields = collect_var_fields(cfg);

    let mut initial = DataflowState::new();
    for (target, fields) in &var_fields {
        let is_external = match target {
            RefTarget::Var(_) => false,
            RefTarget::Param(_) | RefTarget::Through(_) => true,
        };
        for f in fields {
            initial.set(
                (*target, *f),
                if is_external {
                    FieldInit::Init
                } else {
                    FieldInit::Uninit
                },
            );
        }
    }

    let analysis = InitCheckAnalysis {
        value_fields,
        val_types: cfg.val_types.clone(),
    };
    let result = forward_analysis(cfg, &analysis, initial);

    // Post-pass: replay transfer per block and check at reads and call sites.
    let mut errors = Vec::new();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let mut state = result.block_entry[bi].clone();

        for inst in &block.insts {
            match &inst.kind {
                InstKind::Take { target, path, .. } | InstKind::Ref { target, path, .. } => {
                    if let Some(PathSeg::Field(field)) = path.first()
                        && state.get((*target, *field)) == FieldInit::Uninit
                    {
                        errors.push(UninitError {
                            span: inst.span,
                            subject: UninitSubject::Storage(*target),
                            uninit_fields: vec![*field],
                            stored_fields: stored_of(&var_fields, target),
                        });
                    }
                }
                InstKind::FunctionCall {
                    callee,
                    callee_ty,
                    args,
                    ..
                }
                | InstKind::Spawn {
                    callee,
                    callee_ty,
                    args,
                    ..
                } => {
                    check_call_args(
                        &state,
                        cfg,
                        &analysis.value_fields,
                        &var_fields,
                        callee,
                        callee_ty,
                        args,
                        inst.span,
                        &mut errors,
                    );
                }
                _ => {}
            }
            analysis.transfer_inst(inst, &mut state);
        }
    }

    errors
}

/// Check that all fields required by the callee's parameter types are
/// initialized for each argument: in the storage it was taken from, or in
/// the value it was built as.
#[allow(clippy::too_many_arguments)]
fn check_call_args(
    state: &DataflowState<(RefTarget, Astr), FieldInit>,
    cfg: &CfgBody,
    value_fields: &ValueFields,
    var_fields: &FxHashMap<RefTarget, FxHashSet<Astr>>,
    callee: &Callee,
    callee_ty: &Ty,
    args: &[ValueId],
    span: Span,
    errors: &mut Vec<UninitError>,
) {
    // Can't statically check indirect calls.
    if matches!(callee, Callee::Indirect(_)) {
        return;
    }
    let param_types = match callee_ty {
        Ty::Fn { params, .. } => params,
        _ => return,
    };

    for (arg, param) in args.iter().zip(param_types.iter()) {
        let required_fields = match &param.ty {
            Ty::Object(fields) => fields.keys().copied().collect::<Vec<_>>(),
            _ => continue,
        };
        if required_fields.is_empty() {
            continue;
        }
        let (subject, uninit_fields, stored_fields): (UninitSubject, Vec<Astr>, Vec<Astr>) =
            match (find_arg_source(arg, cfg), value_fields.get(arg)) {
                (Some(target), _) => (
                    UninitSubject::Storage(target),
                    required_fields
                        .into_iter()
                        .filter(|f| state.get((target, *f)) == FieldInit::Uninit)
                        .collect(),
                    stored_of(var_fields, &target),
                ),
                (None, Some(built)) => (
                    UninitSubject::Value(*arg),
                    required_fields
                        .into_iter()
                        .filter(|f| !built.contains(f))
                        .collect(),
                    sorted(built.iter().copied()),
                ),
                (None, None) => continue,
            };
        if !uninit_fields.is_empty() {
            errors.push(UninitError {
                span,
                subject,
                uninit_fields,
                stored_fields,
            });
        }
    }
}

/// Extract field names from an Object type. Returns None for non-Object types.
fn extract_object_fields(ty: &Ty) -> Option<Vec<Astr>> {
    match ty {
        Ty::Object(fields) => Some(fields.keys().copied().collect()),
        _ => None,
    }
}

/// The storage `arg` was taken whole from, if it was.
fn find_arg_source(arg: &ValueId, cfg: &CfgBody) -> Option<RefTarget> {
    cfg.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find_map(|inst| match &inst.kind {
            InstKind::Take { dst, target, path } if dst == arg && path.is_empty() => Some(*target),
            _ => None,
        })
}

// -- Tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_init_lattice_laws() {
        // bottom is identity
        let mut x = FieldInit::Init;
        assert!(!x.join_mut(&FieldInit::Init));
        assert_eq!(x, FieldInit::Init);

        // join(Init, Uninit) = Uninit
        let mut x = FieldInit::Init;
        assert!(x.join_mut(&FieldInit::Uninit));
        assert_eq!(x, FieldInit::Uninit);

        // join(Uninit, Init) = Uninit (already top)
        let mut x = FieldInit::Uninit;
        assert!(!x.join_mut(&FieldInit::Init));
        assert_eq!(x, FieldInit::Uninit);

        // idempotent
        let mut x = FieldInit::Uninit;
        assert!(!x.join_mut(&FieldInit::Uninit));
    }
}

fn near_stored(interner: &Interner, error: &UninitError) -> DidYouMean {
    let [wanted] = error.uninit_fields.as_slice() else {
        return DidYouMean::default();
    };
    DidYouMean::of(
        interner.resolve(*wanted),
        error
            .stored_fields
            .iter()
            .map(|field| interner.resolve(*field).to_string())
            .collect::<Vec<_>>(),
    )
}

/// The definite-assignment refusals of a body, in the words a reader of the
/// source knows it by.
pub fn refusals(interner: &Interner, cfg: &CfgBody) -> Vec<MirError> {
    check_init(cfg)
        .into_iter()
        .map(|error| MirError {
            kind: MirErrorKind::FieldNotStored {
                subject: subject_of(interner, cfg, &error.subject),
                near: near_stored(interner, &error),
                fields: error
                    .uninit_fields
                    .iter()
                    .map(|field| interner.resolve(*field).to_string())
                    .collect(),
            },
            span: error.span,
            labels: Vec::new(),
        })
        .collect()
}

fn subject_of(interner: &Interner, cfg: &CfgBody, subject: &UninitSubject) -> ShownValue {
    let value = match subject {
        UninitSubject::Value(value) => *value,
        UninitSubject::Storage(RefTarget::Var(value) | RefTarget::Param(value)) => *value,
        UninitSubject::Storage(RefTarget::Through(value)) => *value,
    };
    match cfg.debug.val_origins.get(&value) {
        Some(ValOrigin::Named(name)) => ShownValue::Named(interner.resolve(*name).to_string()),
        Some(ValOrigin::Context(name)) => {
            ShownValue::Named(format!("@{}", interner.resolve(*name)))
        }
        Some(ValOrigin::ExternParam(name)) => {
            ShownValue::Named(format!("${}", interner.resolve(*name)))
        }
        _ => ShownValue::Anonymous,
    }
}
