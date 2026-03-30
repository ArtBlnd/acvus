//! Definite-assignment analysis — field-level uninit check.
//!
//! Runs on CfgBody (pre-SSA). Tracks which fields of each named storage
//! (Var, Context, Param) are definitely initialized at each program point.
//!
//! At function call sites, checks that arguments have all fields required
//! by the callee's parameter type. Required fields come from fn_types,
//! NOT from val_types (which may have been widened by unification).

use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::dataflow::{forward_analysis, DataflowAnalysis, DataflowState};
use crate::analysis::domain::SemiLattice;
use crate::cfg::{CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{Callee, Inst, InstKind, RefTarget, ValueId};
use acvus_ast::Span;
use crate::ty::{Param as TyParam, Ty};

// ── Domain ──────────────────────────────────────────────────────────

/// Per-field init domain.
///
/// Lattice: Init (bottom) → Uninit (top).
/// join(Init, Uninit) = Uninit — if ANY path leaves a field uninit,
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
            (FieldInit::Uninit, _) => false,
            (FieldInit::Init, FieldInit::Uninit) => {
                *self = FieldInit::Uninit;
                true
            }
            (FieldInit::Init, FieldInit::Init) => false,
        }
    }
}

// ── Error ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UninitError {
    pub span: Span,
    pub target: RefTarget,
    pub uninit_fields: Vec<Astr>,
}

// ── Pre-pass data ───────────────────────────────────────────────────

/// Maps Ref dst ValueId → (target, path).
type RefMap = FxHashMap<ValueId, (RefTarget, Vec<Astr>)>;

/// Maps ValueId → set of field names that the value definitely contains.
type ValueFields = FxHashMap<ValueId, FxHashSet<Astr>>;

fn build_prepass(cfg: &CfgBody) -> (RefMap, ValueFields) {
    let mut ref_map = RefMap::default();
    let mut value_fields = ValueFields::default();

    for block in &cfg.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Ref { dst, target, path } => {
                    ref_map.insert(*dst, (*target, path.clone()));
                }
                InstKind::MakeObject { dst, fields } => {
                    let names: FxHashSet<Astr> = fields.iter().map(|(k, _)| *k).collect();
                    value_fields.insert(*dst, names);
                }
                InstKind::FieldSet {
                    dst, object, field, ..
                } => {
                    // FieldSet produces a new value with the same fields as object.
                    // The field is modified but the set of field names stays the same.
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

    (ref_map, value_fields)
}

/// Collect all (RefTarget, field) pairs from Ref instructions.
/// Uses BOTH identity Refs (type's fields) and field Refs (path proves field exists).
fn collect_var_fields(cfg: &CfgBody, ref_map: &RefMap) -> FxHashMap<RefTarget, FxHashSet<Astr>> {
    let mut target_fields: FxHashMap<RefTarget, FxHashSet<Astr>> = FxHashMap::default();

    for (val, (target, path)) in ref_map {
        if path.is_empty() {
            // Identity ref — extract fields from Ref<T>'s T.
            if let Some(Ty::Ref(inner, _)) = cfg.val_types.get(val) {
                if let Ty::Object(fields) = inner.as_ref() {
                    target_fields
                        .entry(*target)
                        .or_default()
                        .extend(fields.keys().copied());
                }
            }
        } else if let Some(field) = path.first() {
            // Field ref — this path proves the field exists on the target.
            target_fields
                .entry(*target)
                .or_default()
                .insert(*field);
        }
    }

    target_fields
}

// ── Analysis ────────────────────────────────────────────────────────

struct InitCheckAnalysis {
    ref_map: RefMap,
    value_fields: ValueFields,
    /// val_types from CfgBody — for fallback field lookup.
    val_types: FxHashMap<ValueId, Ty>,
}

impl DataflowAnalysis for InitCheckAnalysis {
    type Key = (RefTarget, Astr);
    type Domain = FieldInit;

    fn transfer_inst(
        &self,
        inst: &Inst,
        state: &mut DataflowState<(RefTarget, Astr), FieldInit>,
    ) {
        if let InstKind::Store { dst, value, .. } = &inst.kind {
            if let Some((target, path)) = self.ref_map.get(dst) {
                if path.is_empty() {
                    // Identity store: determine which fields the value actually has.
                    let fields: Option<Vec<Astr>> = if let Some(known) = self.value_fields.get(value) {
                        // MakeObject/FieldSet — known exact fields.
                        Some(known.iter().copied().collect())
                    } else if let Some(ty) = self.val_types.get(value) {
                        // Fallback: use type's fields (function return, Load, etc. — assume complete).
                        extract_object_fields(ty)
                    } else {
                        None
                    };
                    if let Some(fields) = fields {
                        for f in fields {
                            state.set((*target, f), FieldInit::Init);
                        }
                    }
                } else if let Some(field) = path.first() {
                    // Field store: mark this specific field as Init.
                    state.set((*target, *field), FieldInit::Init);
                }
            }
        }
    }

    fn propagate_forward(
        &self,
        source_exit: &DataflowState<(RefTarget, Astr), FieldInit>,
        _params: &[ValueId],
        _args: &[ValueId],
        target_entry: &mut DataflowState<(RefTarget, Astr), FieldInit>,
    ) -> bool {
        // Storage init state is not SSA — no param/arg mapping. Pure join.
        target_entry.join_from(source_exit)
    }

    fn propagate_backward(
        &self,
        _succ_entry: &DataflowState<(RefTarget, Astr), FieldInit>,
        _succ_params: &[ValueId],
        _term_args: &[ValueId],
        _exit_state: &mut DataflowState<(RefTarget, Astr), FieldInit>,
    ) {
        unreachable!("init check is forward-only")
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Run field-level definite-assignment check on a CfgBody.
///
/// `fn_types` maps function QualifiedRef → Ty (function type) for
/// determining required fields at call sites.
/// Run field-level definite-assignment check on a CfgBody.
///
/// `fn_types`: function QualifiedRef → Ty for call-site checking.
/// `external_contexts`: contexts provided by the host — these start as Init.
///   Script-created contexts (not in this set) start as Uninit.
pub fn check_init(
    cfg: &CfgBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    external_contexts: &FxHashSet<QualifiedRef>,
) -> Vec<UninitError> {
    let (ref_map, value_fields) = build_prepass(cfg);
    let var_fields = collect_var_fields(cfg, &ref_map);

    // Build initial state: Var fields start Uninit, Context/Param start Init.
    let mut initial = DataflowState::new();
    for (target, fields) in &var_fields {
        let is_external = match target {
            RefTarget::Var(_) => false,
            RefTarget::Param(_) => true,
            RefTarget::Context(qref) => external_contexts.contains(qref),
        };
        for f in fields {
            initial.set((*target, *f), if is_external { FieldInit::Init } else { FieldInit::Uninit });
        }
    }

    let analysis = InitCheckAnalysis {
        ref_map: ref_map.clone(),
        value_fields,
        val_types: cfg.val_types.clone(),
    };
    let result = forward_analysis(cfg, &analysis, initial);

    // Post-pass: replay transfer per block and check at FunctionCall/Spawn sites.
    let mut errors = Vec::new();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let mut state = result.block_entry[bi].clone();

        for inst in &block.insts {
            // Check before transfer (state reflects point before this inst).
            match &inst.kind {
                // Check field loads: loading a possibly-uninit field.
                InstKind::Load { src, .. } => {
                    if let Some((target, path)) = ref_map.get(src) {
                        if let Some(field) = path.first() {
                            if state.get((*target, *field)) == FieldInit::Uninit {
                                errors.push(UninitError {
                                    span: inst.span,
                                    target: *target,
                                    uninit_fields: vec![*field],
                                });
                            }
                        }
                    }
                }
                // Check function calls: all required fields of args must be init.
                InstKind::FunctionCall {
                    callee, args, ..
                }
                | InstKind::Spawn {
                    callee, args, ..
                } => {
                    check_call_args(
                        &state,
                        &ref_map,
                        cfg,
                        fn_types,
                        callee,
                        args,
                        inst.span,
                        &mut errors,
                    );
                }
                _ => {}
            }

            // Apply transfer.
            analysis.transfer_inst(inst, &mut state);
        }
    }

    errors
}

/// Check that all fields required by the callee's parameter types
/// are initialized for each argument.
fn check_call_args(
    state: &DataflowState<(RefTarget, Astr), FieldInit>,
    ref_map: &RefMap,
    cfg: &CfgBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    callee: &Callee,
    args: &[ValueId],
    span: Span,
    errors: &mut Vec<UninitError>,
) {
    // Resolve callee type.
    let fn_ty = match callee {
        Callee::Direct(qref) => fn_types.get(qref),
        Callee::Indirect(_) => return, // Can't statically check indirect calls.
    };
    let fn_ty = match fn_ty {
        Some(ty) => ty,
        None => return,
    };

    // Extract parameter types from function type.
    let param_types = match fn_ty {
        Ty::Fn { params, .. } => params,
        _ => return,
    };

    // For each argument, trace back to its source storage and check fields.
    for (arg, param) in args.iter().zip(param_types.iter()) {
        let required_fields = match &param.ty {
            Ty::Object(fields) => fields.keys().copied().collect::<Vec<_>>(),
            _ => continue,
        };
        if required_fields.is_empty() {
            continue;
        }

        // Trace arg back to its source storage.
        // Pattern: Load { dst: arg, src: ref_val } where ref_val → (target, path)
        let source_target = find_arg_source(arg, ref_map, cfg);
        let target = match source_target {
            Some(t) => t,
            None => continue, // Can't trace — skip check (conservative).
        };

        let mut uninit_fields = Vec::new();
        for f in &required_fields {
            if state.get((target, *f)) == FieldInit::Uninit {
                uninit_fields.push(*f);
            }
        }

        if !uninit_fields.is_empty() {
            errors.push(UninitError {
                span,
                target,
                uninit_fields,
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

/// Trace a ValueId back to its source RefTarget.
/// Looks for the pattern: `arg` was defined by `Load { dst: arg, src }`,
/// and `src` was defined by `Ref { dst: src, target, path: [] }`.
fn find_arg_source(
    arg: &ValueId,
    ref_map: &RefMap,
    cfg: &CfgBody,
) -> Option<RefTarget> {
    // Find the Load that defined arg.
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Load { dst, src, .. } = &inst.kind {
                if dst == arg {
                    // Found the Load. Now check if src is an identity Ref.
                    if let Some((target, path)) = ref_map.get(src) {
                        if path.is_empty() {
                            return Some(*target);
                        }
                    }
                }
            }
        }
    }
    None
}

// ── Tests ───────────────────────────────────────────────────────────

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
