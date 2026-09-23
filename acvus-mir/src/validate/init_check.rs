//! Definite initialization of object fields (RFC-0042 rule 1, RFC-0050
//! rule 8).

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::analysis::dataflow::{DataflowAnalysis, DataflowState, forward_analysis};
use crate::analysis::domain::SemiLattice;
use crate::analysis::inst_info;
use crate::cfg::{CfgBody, Terminator};
use crate::error::{DidYouMean, MirError, MirErrorKind, ShownValue};
use crate::ir::{Inst, InstKind, PathSeg, RefTarget, ValOrigin, ValueId};
use crate::ty::Ty;
use acvus_ast::Span;

// -- Domain ----------------------------------------------------------

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Holder {
    Storage(RefTarget),
    Register(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LeafId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Leaf {
    holder: Holder,
    path: LeafId,
}

type State = DataflowState<Leaf, FieldInit>;

pub type FieldPath = Vec<Astr>;

#[derive(Default)]
struct LeafPaths {
    ids: FxHashMap<FieldPath, LeafId>,
    paths: Vec<FieldPath>,
}

impl LeafPaths {
    fn intern(&mut self, path: FieldPath) -> LeafId {
        if let Some(id) = self.ids.get(&path) {
            return *id;
        }
        let id =
            LeafId(u32::try_from(self.paths.len()).expect("a body has fewer than 2^32 leaf paths"));
        self.paths.push(path.clone());
        self.ids.insert(path, id);
        id
    }

    fn id(&self, path: &[Astr]) -> LeafId {
        *self
            .ids
            .get(path)
            .expect("a field path below a holder is a leaf of the holder's type")
    }

    fn path(&self, id: LeafId) -> &[Astr] {
        &self.paths[id.0 as usize]
    }
}

fn leaf_paths(ty: &Ty) -> Vec<FieldPath> {
    let Ty::Object(object) = ty else {
        return vec![Vec::new()];
    };
    let mut out = Vec::new();
    for (name, field) in object.iter() {
        for mut rest in leaf_paths(field) {
            rest.insert(0, *name);
            out.push(rest);
        }
    }
    out
}

/// The field names of `path`, or `None` where it steps into an element or a
/// payload. An element or a payload is whole: `whole_uses` refuses an
/// incomplete value where a container or a variant takes it.
fn field_path(path: &[PathSeg]) -> Option<FieldPath> {
    path.iter()
        .map(|seg| match seg {
            PathSeg::Field(name) => Some(*name),
            PathSeg::Index(_) | PathSeg::Payload => None,
        })
        .collect()
}

// -- Errors ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UninitError {
    pub span: Span,
    pub subject: UninitSubject,
    pub missing: Vec<FieldPath>,
    pub stored_fields: Vec<Astr>,
}

/// What lacked the fields: a named storage, or a value built in place and
/// passed on without a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UninitSubject {
    Storage(RefTarget),
    Value(ValueId),
}

struct Named {
    subject: UninitSubject,
    prefix: FieldPath,
}

// -- Analysis --------------------------------------------------------

#[derive(Debug, Clone)]
struct PartOf {
    holder: Holder,
    prefix: FieldPath,
}

struct PartRead {
    dst: ValueId,
    part: PartOf,
}

fn register_part(object: ValueId, prefix: FieldPath) -> PartOf {
    PartOf {
        holder: Holder::Register(object),
        prefix,
    }
}

fn part_read(kind: &InstKind) -> Option<PartRead> {
    let (dst, part) = match kind {
        InstKind::Take {
            dst, target, path, ..
        } => {
            let prefix = field_path(path)?;
            let part = PartOf {
                holder: Holder::Storage(*target),
                prefix,
            };
            (dst, part)
        }
        InstKind::FieldGet {
            dst,
            object,
            field,
            rest,
        } => {
            let prefix = std::iter::once(*field)
                .chain(rest.iter().copied())
                .collect();
            (dst, register_part(*object, prefix))
        }
        InstKind::ObjectGet { dst, object, key } => (dst, register_part(*object, vec![*key])),
        _ => return None,
    };
    Some(PartRead { dst: *dst, part })
}

fn whole_uses(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Assign {
            target: RefTarget::Var(_),
            path,
            ..
        } if field_path(path).is_some() => Vec::new(),
        InstKind::MakeObject { .. }
        | InstKind::FieldSet { .. }
        | InstKind::FieldGet { .. }
        | InstKind::ObjectGet { .. } => Vec::new(),
        other => inst_info::uses(other).into_iter().collect(),
    }
}

struct InitCheck<'a> {
    cfg: &'a CfgBody,
    paths: LeafPaths,
    leaves: FxHashMap<ValueId, Vec<LeafId>>,
    parts: FxHashMap<ValueId, PartOf>,
}

impl<'a> InitCheck<'a> {
    fn new(cfg: &'a CfgBody) -> Self {
        let mut paths = LeafPaths::default();
        let mut leaves = FxHashMap::default();
        for (value, ty) in &cfg.val_types {
            let ids = leaf_paths(ty)
                .into_iter()
                .map(|p| paths.intern(p))
                .collect();
            leaves.insert(*value, ids);
        }
        let parts = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| part_read(&inst.kind))
            .map(|read| (read.dst, read.part))
            .collect();
        Self {
            cfg,
            paths,
            leaves,
            parts,
        }
    }

    fn ty_of(&self, value: ValueId) -> &Ty {
        self.cfg
            .val_types
            .get(&value)
            .expect("every register of a lowered body has a type")
    }

    fn leaves_of(&self, value: ValueId) -> &[LeafId] {
        self.leaves
            .get(&value)
            .expect("every register of a lowered body has a type")
    }

    fn is_object(&self, value: ValueId) -> bool {
        matches!(self.ty_of(value), Ty::Object(_))
    }

    fn leaf_below(&self, prefix: &[Astr], leaf: LeafId) -> LeafId {
        let mut path = prefix.to_vec();
        path.extend_from_slice(self.paths.path(leaf));
        self.paths.id(&path)
    }

    fn read_part(&self, state: &mut State, dst: ValueId, part: &PartOf) {
        for &leaf in self.leaves_of(dst) {
            let from = Leaf {
                holder: part.holder,
                path: self.leaf_below(&part.prefix, leaf),
            };
            let to = Leaf {
                holder: Holder::Register(dst),
                path: leaf,
            };
            state.set(to, state.get(from));
        }
    }

    fn store(&self, state: &mut State, holder: Holder, prefix: &[Astr], value: ValueId) {
        if !self.is_object(value) {
            let path = self.paths.id(prefix);
            state.set(Leaf { holder, path }, FieldInit::Init);
            return;
        }
        for &leaf in self.leaves_of(value) {
            let from = Leaf {
                holder: Holder::Register(value),
                path: leaf,
            };
            let to = Leaf {
                holder,
                path: self.leaf_below(prefix, leaf),
            };
            state.set(to, state.get(from));
        }
    }

    fn transfer(&self, kind: &InstKind, state: &mut State) {
        match kind {
            InstKind::MakeObject { dst, fields } => {
                let built = Holder::Register(*dst);
                for &path in self.leaves_of(*dst) {
                    state.set(
                        Leaf {
                            holder: built,
                            path,
                        },
                        FieldInit::Uninit,
                    );
                }
                for (name, value) in fields {
                    self.store(state, built, &[*name], *value);
                }
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let whole = PartOf {
                    holder: Holder::Register(*object),
                    prefix: Vec::new(),
                };
                self.read_part(state, *dst, &whole);
                let prefix: FieldPath = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .collect();
                self.store(state, Holder::Register(*dst), &prefix, *value);
            }
            InstKind::Assign {
                target: target @ RefTarget::Var(_),
                path,
                value,
                ..
            } => {
                if let Some(prefix) = field_path(path) {
                    self.store(state, Holder::Storage(*target), &prefix, *value);
                }
            }
            other => {
                if let Some(read) = part_read(other)
                    && self.is_object(read.dst)
                {
                    self.read_part(state, read.dst, &read.part);
                }
            }
        }
    }

    fn missing_below(&self, state: &State, part: &PartOf, of: ValueId) -> Vec<FieldPath> {
        self.leaves_of(of)
            .iter()
            .filter(|leaf| self.paths.path(**leaf).starts_with(&part.prefix))
            .filter(|leaf| {
                let at = Leaf {
                    holder: part.holder,
                    path: **leaf,
                };
                state.get(at) == FieldInit::Uninit
            })
            .map(|leaf| self.paths.path(*leaf).to_vec())
            .collect()
    }

    fn stored_fields(&self, state: &State, holder: Holder, of: ValueId) -> Vec<Astr> {
        let mut names: Vec<Astr> = self
            .leaves_of(of)
            .iter()
            .filter(|leaf| {
                state.get(Leaf {
                    holder,
                    path: **leaf,
                }) == FieldInit::Init
            })
            .filter_map(|leaf| self.paths.path(*leaf).first().copied())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    fn named(&self, value: ValueId) -> Named {
        match self.parts.get(&value) {
            Some(PartOf {
                holder: Holder::Storage(target),
                prefix,
            }) => Named {
                subject: UninitSubject::Storage(*target),
                prefix: prefix.clone(),
            },
            Some(PartOf {
                holder: Holder::Register(object),
                prefix,
            }) => {
                let mut outer = self.named(*object);
                outer.prefix.extend_from_slice(prefix);
                outer
            }
            None => Named {
                subject: UninitSubject::Value(value),
                prefix: Vec::new(),
            },
        }
    }

    fn refuse_incomplete(&self, state: &State, value: ValueId, span: Span) -> Option<UninitError> {
        if !self.is_object(value) {
            return None;
        }
        let whole = PartOf {
            holder: Holder::Register(value),
            prefix: Vec::new(),
        };
        let missing = self.missing_below(state, &whole, value);
        if missing.is_empty() {
            return None;
        }
        let Named { subject, prefix } = self.named(value);
        Some(UninitError {
            span,
            subject,
            missing: missing
                .into_iter()
                .map(|leaf| prefix.iter().copied().chain(leaf).collect())
                .collect(),
            stored_fields: self.stored_fields(state, whole.holder, value),
        })
    }

    fn refuse_storage_part(
        &self,
        state: &State,
        target: RefTarget,
        prefix: FieldPath,
        span: Span,
    ) -> Option<UninitError> {
        let RefTarget::Var(slot) = target else {
            return None;
        };
        let part = PartOf {
            holder: Holder::Storage(target),
            prefix,
        };
        let missing = self.missing_below(state, &part, slot);
        if missing.is_empty() {
            return None;
        }
        Some(UninitError {
            span,
            subject: UninitSubject::Storage(target),
            missing,
            stored_fields: self.stored_fields(state, part.holder, slot),
        })
    }

    fn refuse_register_part(
        &self,
        state: &State,
        object: ValueId,
        prefix: FieldPath,
        span: Span,
    ) -> Option<UninitError> {
        let part = register_part(object, prefix);
        let missing = self.missing_below(state, &part, object);
        if missing.is_empty() {
            return None;
        }
        let Named { subject, prefix } = self.named(object);
        Some(UninitError {
            span,
            subject,
            missing: missing
                .into_iter()
                .map(|leaf| prefix.iter().copied().chain(leaf).collect())
                .collect(),
            stored_fields: self.stored_fields(state, part.holder, object),
        })
    }

    fn step(&self, inst: &Inst, state: &mut State, errors: &mut Vec<UninitError>) {
        match &inst.kind {
            InstKind::Take {
                dst, target, path, ..
            } if !self.is_object(*dst) => {
                if let Some(prefix) = field_path(path) {
                    errors.extend(self.refuse_storage_part(state, *target, prefix, inst.span));
                }
            }
            InstKind::Ref { target, path, .. } => {
                if let Some(prefix) = field_path(path) {
                    errors.extend(self.refuse_storage_part(state, *target, prefix, inst.span));
                }
            }
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } if !self.is_object(*dst) => {
                let prefix = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .collect();
                errors.extend(self.refuse_register_part(state, *object, prefix, inst.span));
            }
            InstKind::ObjectGet { dst, object, key } if !self.is_object(*dst) => {
                errors.extend(self.refuse_register_part(state, *object, vec![*key], inst.span));
            }
            _ => {}
        }
        for value in whole_uses(&inst.kind) {
            errors.extend(self.refuse_incomplete(state, value, inst.span));
        }
        self.transfer(&inst.kind, state);
    }
}

impl DataflowAnalysis for InitCheck<'_> {
    type Key = Leaf;
    type Domain = FieldInit;

    fn transfer_inst(&self, inst: &Inst, state: &mut State) {
        self.transfer(&inst.kind, state);
    }

    fn propagate_forward(
        &self,
        source_exit: &State,
        params: &[ValueId],
        first: usize,
        args: &[ValueId],
        target_entry: &mut State,
    ) -> bool {
        let mut changed = target_entry.join_from(source_exit);
        for (param, arg) in params[first..].iter().zip(args) {
            for &path in self.leaves_of(*param) {
                let incoming = source_exit.get(Leaf {
                    holder: Holder::Register(*arg),
                    path,
                });
                let at = Leaf {
                    holder: Holder::Register(*param),
                    path,
                };
                let mut entry = target_entry.get(at);
                if entry.join_mut(&incoming) {
                    target_entry.set(at, entry);
                    changed = true;
                }
            }
        }
        changed
    }

    fn propagate_backward(
        &self,
        _succ_entry: &State,
        _succ_params: &[ValueId],
        _first: usize,
        _term_args: &[ValueId],
        _exit_state: &mut State,
    ) {
        unreachable!("init check is forward-only")
    }
}

// -- Public API ------------------------------------------------------

pub fn check_init(cfg: &CfgBody) -> Vec<UninitError> {
    let analysis = InitCheck::new(cfg);

    let mut initial = State::new();
    for (value, ty) in &cfg.val_types {
        if !matches!(ty, Ty::Object(_)) {
            continue;
        }
        let holder = Holder::Storage(RefTarget::Var(*value));
        for &path in analysis.leaves_of(*value) {
            initial.set(Leaf { holder, path }, FieldInit::Uninit);
        }
    }

    let result = forward_analysis(cfg, &analysis, initial);

    let mut errors = Vec::new();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let mut state = result.block_entry[bi].clone();
        for inst in &block.insts {
            analysis.step(inst, &mut state, &mut errors);
        }
        if let Terminator::Return { value, span, .. } = &block.terminator {
            errors.extend(analysis.refuse_incomplete(&state, *value, *span));
        }
    }
    errors
}

fn near_stored(interner: &Interner, error: &UninitError) -> DidYouMean {
    let [wanted] = error.missing.as_slice() else {
        return DidYouMean::default();
    };
    let [wanted] = wanted.as_slice() else {
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
                    .missing
                    .iter()
                    .map(|path| {
                        path.iter()
                            .map(|field| interner.resolve(*field).to_string())
                            .collect::<Vec<_>>()
                            .join(".")
                    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_init_lattice_laws() {
        let mut x = FieldInit::Init;
        assert!(!x.join_mut(&FieldInit::Init));
        assert_eq!(x, FieldInit::Init);

        let mut x = FieldInit::Init;
        assert!(x.join_mut(&FieldInit::Uninit));
        assert_eq!(x, FieldInit::Uninit);

        let mut x = FieldInit::Uninit;
        assert!(!x.join_mut(&FieldInit::Init));
        assert_eq!(x, FieldInit::Uninit);

        let mut x = FieldInit::Uninit;
        assert!(!x.join_mut(&FieldInit::Uninit));
    }
}
