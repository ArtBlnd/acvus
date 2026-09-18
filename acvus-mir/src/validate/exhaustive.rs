//! A `match` is exhaustive, and it is decided where the variant set is
//! known (RFC-0051 §3-§4).
//!
//! The language's enums are structural and unify by union, so a value's
//! variant set can grow anywhere the value flows, and a check at the match
//! site does not in general see the closed set. This pass fixes the
//! boundary of what is decided now. Everything it asks goes through
//! [`known_variants`]; the close-phase answer and Maranget's matrix widen
//! its `Closed` later and move neither the rule nor its place.
//!
//! The pass reads `InstKind::Switch`, the one shape that names a `match`.
//! An `if let` is two arms and always exhaustive, and never wears one.

use std::collections::BTreeSet;

use acvus_utils::Astr;
use rustc_hash::FxHashSet;

use crate::analysis::{escape, inst_info};
use crate::ir::{Inst, InstKind, Label, MirBody, MirModule, RefTarget, ValueId};
use crate::ty::Ty;
use crate::validate::type_check::{ValidationError, ValidationErrorKind};

/// A builtin enum whose type names its variants: the count is the whole
/// answer, because typeck has already refused any arm outside the set
/// (RFC-0039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Builtin {
    pub name: &'static str,
    pub arity: usize,
}

const OPTION: Builtin = Builtin {
    name: "Option",
    arity: 2,
};
const RESULT: Builtin = Builtin {
    name: "Result",
    arity: 2,
};

/// What this stage knows about the variants a value can hold: the seam's
/// three answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Known {
    /// (c) The set is written down in this body: these tags and no others.
    Closed(BTreeSet<Astr>),
    /// (b) The type names the set. This stage holds no interner and so
    /// names no tag itself; the count is what it checks against.
    ClosedBuiltin(Builtin),
    /// The set is not decidable here. A `match` on such a value needs a
    /// `_` arm, which states the intent the checker cannot yet verify.
    Open,
}

pub fn check_exhaustive(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    check_body(&module.main, "main", &mut errors);
    for (label, closure) in &module.closures {
        check_body(closure, &format!("closure({label:?})"), &mut errors);
    }
    errors
}

fn check_body(body: &MirBody, scope: &str, errors: &mut Vec<ValidationError>) {
    for (at, inst) in body.insts.iter().enumerate() {
        let InstKind::Switch { tag, arms, default } = &inst.kind else {
            continue;
        };
        // A catch-all is always the way through: `default` is present
        // exactly when the source wrote one.
        if default.is_some() {
            continue;
        }
        let covered: BTreeSet<Astr> = arms.iter().map(|(tag, _, _)| *tag).collect();
        let kind = match known_variants(body, *tag) {
            Known::Open => ValidationErrorKind::NonExhaustiveMatch,
            Known::ClosedBuiltin(builtin) if covered.len() < builtin.arity => {
                ValidationErrorKind::MatchMissesBuiltinVariants {
                    enum_name: builtin.name,
                    arity: builtin.arity,
                    covered: covered.len(),
                }
            }
            Known::ClosedBuiltin(_) => continue,
            Known::Closed(set) => {
                let missing: Vec<Astr> = set.difference(&covered).copied().collect();
                if missing.is_empty() {
                    continue;
                }
                ValidationErrorKind::MatchMissesVariants {
                    enum_name: enum_name_of(body, *tag),
                    missing,
                }
            }
        };
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: at,
            span: inst.span,
            kind,
        });
    }
}

/// The name the source gave the enum the value carries, for the refusal to
/// write `E::B` rather than a bare tag.
fn enum_name_of(body: &MirBody, value: ValueId) -> Option<Astr> {
    let ty = match body.val_types.get(&value)? {
        Ty::Ref(_, inner) => &inner.ty,
        ty => ty,
    };
    match ty {
        Ty::Enum { name, .. } => Some(*name),
        _ => None,
    }
}

/// The variants the value a `Switch` reads its tag from can hold
/// (RFC-0051 §4). This is the whole seam: a later stage widens `Closed`
/// here and nothing else moves.
///
/// - (b) an `Option` or a `Result`: [`Known::ClosedBuiltin`], the two tags
///   the type names;
/// - (c) a storage of this body whose every definition is a `MakeVariant`
///   or a join of them, that comes from nowhere outside and goes nowhere
///   outside: [`Known::Closed`] on the union of those tags;
/// - [`Known::Open`] otherwise.
pub fn known_variants(body: &MirBody, value: ValueId) -> Known {
    if let Some(builtin) = builtin_of(body, value) {
        return Known::ClosedBuiltin(builtin);
    }
    let Some(storage) = local_storage(body, value) else {
        return Known::Open;
    };
    // What a callee reached could write a new variant back, and this stage
    // cannot see whether it did.
    if escape::of_insts(body.insts.iter().map(|inst| &inst.kind)).escapes(storage) {
        return Known::Open;
    }
    let mut tags = BTreeSet::new();
    let mut defined_here = false;
    for inst in &body.insts {
        let InstKind::Assign {
            target,
            path,
            value: written,
        } = &inst.kind
        else {
            continue;
        };
        if *target != RefTarget::Var(storage) || !path.is_empty() {
            continue;
        }
        defined_here = true;
        let Some(written) = constructed_tags(body, *written, &mut FxHashSet::default()) else {
            return Known::Open;
        };
        tags.extend(written);
    }
    match defined_here {
        // A storage with no definition in this body holds what something
        // outside put there.
        false => Known::Open,
        true => Known::Closed(tags),
    }
}

/// (b) A reference is read through: a place scrutinee is lent for the tag
/// read, and what it lends is what names the variants.
fn builtin_of(body: &MirBody, value: ValueId) -> Option<Builtin> {
    let ty = match body.val_types.get(&value)? {
        Ty::Ref(_, inner) => &inner.ty,
        ty => ty,
    };
    match ty {
        Ty::Option(_) => Some(OPTION),
        Ty::Result(..) => Some(RESULT),
        _ => None,
    }
}

/// The storage of this body the value names. The register a `Switch` reads
/// is either a lend or a read of a local slot, or the slot's own value. A
/// parameter, a capture, or a read through one of them names no local
/// storage, and the answer there is `Open`.
fn local_storage(body: &MirBody, value: ValueId) -> Option<ValueId> {
    let from_outside: FxHashSet<ValueId> = body
        .params
        .iter()
        .chain(&body.captures)
        .map(|(_, v)| *v)
        .collect();
    if from_outside.contains(&value) {
        return None;
    }
    let defining = body.insts.iter().find(|inst| defines(inst, value));
    match defining.map(|inst| &inst.kind) {
        Some(InstKind::Ref { target, path, .. } | InstKind::Take { target, path, .. })
            if path.is_empty() =>
        {
            match target {
                RefTarget::Var(slot) if !from_outside.contains(slot) => Some(*slot),
                _ => None,
            }
        }
        // Not a read of a storage: the value is the storage's own register.
        _ => Some(value),
    }
}

fn defines(inst: &Inst, value: ValueId) -> bool {
    match &inst.kind {
        InstKind::Assign { target, .. } => *target == RefTarget::Var(value),
        kind => inst_info::defs(kind).contains(&value),
    }
}

/// The tags a value is built from: one `MakeVariant`, or a join of them
/// through a block parameter. Anything else -- a call's result, a field
/// read, an undefined value -- is not a constructor this body wrote, and
/// the answer is `None`, which the caller reads as `Open`.
fn constructed_tags(
    body: &MirBody,
    value: ValueId,
    seen: &mut FxHashSet<ValueId>,
) -> Option<BTreeSet<Astr>> {
    if !seen.insert(value) {
        // Already on the path: a loop carrying a variant reaches its
        // constructors on the edge that is not this one.
        return Some(BTreeSet::new());
    }
    let defining = body.insts.iter().find(|inst| defines(inst, value))?;
    match &defining.kind {
        InstKind::MakeVariant { tag, .. } => Some(BTreeSet::from([*tag])),
        InstKind::Assign { value: written, .. } => constructed_tags(body, *written, seen),
        InstKind::BlockLabel { label, params, .. } => {
            let index = params.iter().position(|p| *p == value)?;
            let mut tags = BTreeSet::new();
            for inst in &body.insts {
                for args in incoming(&inst.kind, *label) {
                    tags.extend(constructed_tags(body, *args.get(index)?, seen)?);
                }
            }
            Some(tags)
        }
        _ => None,
    }
}

/// The argument lists an instruction hands `label`, over every edge it has
/// to it.
fn incoming(kind: &InstKind, label: Label) -> Vec<&Vec<ValueId>> {
    match kind {
        InstKind::Jump { label: to, args } if *to == label => vec![args],
        InstKind::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => [(then_label, then_args), (else_label, else_args)]
            .into_iter()
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| args)
            .collect(),
        InstKind::Switch { arms, default, .. } => arms
            .iter()
            .map(|(_, to, args)| (to, args))
            .chain(default.iter().map(|(to, args)| (to, args)))
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| args)
            .collect(),
        _ => Vec::new(),
    }
}
