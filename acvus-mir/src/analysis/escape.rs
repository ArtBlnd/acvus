//! Does a storage slot of this body reach anything outside it?
//!
//! Two callers ask, and a missed escape costs them opposite amounts.
//! [`crate::validate::exhaustive`] asks to decide whether the variant set a
//! `match` sees is the set this body wrote; a missed escape only makes it
//! refuse an exhaustiveness claim it could have allowed.
//! [`crate::optimize::sroa`] asks to decide whether an aggregate has to
//! exist at all; a missed escape there is a miscompile. So the arms of
//! [`escaping_uses`] are enumerated over the whole instruction set rather
//! than listed, and an arm that names no value is the claim that the
//! instruction reads its operands and hands them nowhere.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::{Callee, InstKind, RefTarget, ValueId};

/// The storages a body lets out, and the values that name them.
#[derive(Debug, Default)]
pub struct EscapeScan {
    /// A whole `Ref`/`Take` result and the storage it is a second name for.
    names: FxHashMap<ValueId, ValueId>,
    escaped: FxHashSet<ValueId>,
}

/// Which storages of a body escape it.
#[derive(Debug, Default)]
pub struct Escaped {
    storages: FxHashSet<ValueId>,
}

impl Escaped {
    pub fn escapes(&self, storage: ValueId) -> bool {
        self.storages.contains(&storage)
    }
}

pub fn named_storage(target: &RefTarget) -> ValueId {
    match target {
        RefTarget::Var(slot) | RefTarget::Param(slot) | RefTarget::Through(slot) => *slot,
    }
}

impl EscapeScan {
    pub fn observe(&mut self, kind: &InstKind) {
        self.observe_names(kind);
        escaping_uses(kind, &mut |v| {
            self.escaped.insert(v);
        });
    }

    /// The value a `Return` carries, which `CfgBody` holds as a terminator
    /// and not as an instruction.
    pub fn observe_returned(&mut self, value: ValueId) {
        self.escaped.insert(value);
    }

    pub fn finish(self) -> Escaped {
        let Self { names, escaped } = self;
        let storages = escaped
            .iter()
            .map(|v| named_through(&names, *v))
            .chain(escaped.iter().copied())
            .collect();
        Escaped { storages }
    }

    fn observe_names(&mut self, kind: &InstKind) {
        match kind {
            InstKind::Ref {
                dst, target, path, ..
            } => match path.is_empty() {
                true => {
                    self.names.insert(*dst, named_storage(target));
                }
                // The address of a part is an address into what holds it.
                false => {
                    self.escaped.insert(named_storage(target));
                }
            },
            InstKind::Take {
                dst, target, path, ..
            } if path.is_empty() => {
                self.names.insert(*dst, named_storage(target));
            }
            _ => {}
        }
    }
}

/// The storage `value` is a name for, following every `Ref` and `Take` on
/// the way. A name is defined after what it names, so the walk descends.
fn named_through(names: &FxHashMap<ValueId, ValueId>, value: ValueId) -> ValueId {
    let mut at = value;
    let mut seen = FxHashSet::default();
    while let Some(&of) = names.get(&at) {
        assert!(seen.insert(at), "a place names itself: {at:?}");
        at = of;
    }
    at
}

/// The values this instruction lets outlive it.
fn escaping_uses(kind: &InstKind, out: &mut impl FnMut(ValueId)) {
    match kind {
        InstKind::FunctionCall { callee, args, .. } | InstKind::Spawn { callee, args, .. } => {
            if let Callee::Indirect(f) = callee {
                out(*f);
            }
            args.iter().copied().for_each(&mut *out);
        }
        InstKind::MakeClosure { captures, .. } => captures.iter().copied().for_each(out),
        InstKind::Return { value, .. } => out(*value),
        InstKind::MakeArray { elements, .. } | InstKind::MakeTuple { elements, .. } => {
            elements.iter().copied().for_each(out)
        }
        InstKind::StringConcat { parts, .. } => parts.iter().copied().for_each(out),
        InstKind::StringAppend { target, part } => {
            out(*target);
            out(*part);
        }
        InstKind::MakeObject { fields, .. } => fields.iter().for_each(|(_, v)| out(*v)),
        InstKind::FieldSet { object, value, .. }
        | InstKind::IndexSet {
            slice: object,
            value,
            ..
        } => {
            out(*object);
            out(*value);
        }
        InstKind::MakeVariant { payload, .. } => payload.iter().copied().for_each(out),
        InstKind::Commit { value, .. } | InstKind::Assign { value, .. } => out(*value),
        InstKind::AsSlice {
            container: borrowed,
            ..
        }
        | InstKind::Index {
            slice: borrowed, ..
        }
        | InstKind::Eval { src: borrowed, .. } => out(*borrowed),

        // The instruction reads its operands and hands them nowhere; a
        // block argument stays inside the body.
        InstKind::Const { .. }
        | InstKind::ConstStr { .. }
        | InstKind::Undef { .. }
        | InstKind::Poison { .. }
        | InstKind::Nop
        | InstKind::Diverge
        | InstKind::Fetch { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::Ref { .. }
        | InstKind::Take { .. }
        | InstKind::BinOp { .. }
        | InstKind::UnaryOp { .. }
        | InstKind::Check { .. }
        | InstKind::Cast { .. }
        | InstKind::FieldGet { .. }
        | InstKind::Merge { .. }
        | InstKind::StringEq { .. }
        | InstKind::StringClone { .. }
        | InstKind::StructuralEq { .. }
        | InstKind::StructuralClone { .. }
        | InstKind::TupleIndex { .. }
        | InstKind::TestLiteral { .. }
        | InstKind::TestObjectKey { .. }
        | InstKind::ArrayIndex { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::TestVariant { .. }
        | InstKind::UnwrapVariant { .. }
        | InstKind::Drop { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Diamond { .. }
        | InstKind::Switch { .. } => {}

        // The source outlives the terminator: every iteration reads an
        // element through it (RFC-0057).
        InstKind::For { source, .. } => source.uses().into_iter().for_each(out),
    }
}
