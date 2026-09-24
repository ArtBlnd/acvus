//! What is known of a value before the body runs (RFC-0071 rule 5).
//!
//! A value is known when every run of the body gives it one value this
//! module can name: a constant, text, or a variant, object, tuple or array
//! of known parts (RFC-0087 rule 5). `optimize::branch` decides a dispatch
//! on a known value, so a bound `$` holding a structured value narrows the
//! inputs a template requires as a bound string or integer does.
//!
//! A value of a reference type is known by its referent, and a reader reads
//! through it exactly where `acvus-interpreter`'s preparation does: where
//! the reader's operand is typed `Ty::Ref`.
//!
//! Obligation across artifacts: every answer here is the answer
//! `acvus-interpreter`'s operation for the same instruction computes at run
//! time — `ops::pattern` for `TestLiteral` and `TestObjectKey`,
//! `ops::variant` for `TestVariant` and `UnwrapVariant`, `ops::string` for
//! `StringEq`, and `prepare::constant` for the word a constant holds. Change
//! one there and the matching arm here moves with it. Where an answer cannot
//! be computed identically, the value is left unknown.

use acvus_ast::Literal;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::cfg::{CfgBody, Terminator, reachable};
use crate::ir::{InstKind, Label, PathSeg, RefTarget, SwitchKey, ValueId};
use crate::ty::{Mutability, Ty};

/// A value every run gives, or the referent every run names.
#[derive(Debug, Clone, PartialEq)]
pub enum Known {
    Word(Word),
    Text(String),
    Variant {
        tag: Astr,
        payload: Option<Box<Known>>,
    },
    Object(FxHashMap<Astr, Known>),
    Tuple(Vec<Known>),
    Array(Vec<Known>),
}

/// A value one machine word holds, as the machine reads it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Word {
    /// The value the word spells at the integer type's width.
    Int(i128),
    /// The bits of the `f64`, so that two words are one value exactly when
    /// the machine holds one pattern.
    Float(u64),
    Bool(bool),
    Char(char),
    Unit,
}

impl Word {
    /// The word `prepare::constant` writes for `value` at `ty`. A list is
    /// not a word, and an integer the checker did not type as one is not
    /// a constant the machine builds.
    fn of_constant(value: &Literal, ty: &Ty) -> Option<Word> {
        match (value.desugared(), ty) {
            (Literal::Int(n), Ty::Int(width)) => Some(Word::Int(width.read(register_word(n)))),
            (Literal::Float(x), _) => Some(Word::Float(x.to_bits())),
            (Literal::Bool(b), _) => Some(Word::Bool(b)),
            (Literal::Char(c), _) => Some(Word::Char(c)),
            (Literal::Unit, _) => Some(Word::Unit),
            (Literal::Int(_), _)
            | (Literal::String(_), _)
            | (Literal::List(_), _)
            | (Literal::IntOf(_), _)
            | (Literal::Bytes(_), _) => None,
        }
    }
}

/// The register word `prepare::constant` stores for an integer constant,
/// `*n as u64`, which the width then reads back.
fn register_word(n: i128) -> u64 {
    n as u64
}

impl Known {
    fn at(&self, path: &[PathSeg]) -> Option<&Known> {
        path.iter().try_fold(self, |held, seg| match (seg, held) {
            (PathSeg::Field(key), Known::Object(fields)) => fields.get(key),
            (PathSeg::Index(at), Known::Tuple(parts) | Known::Array(parts)) => parts.get(*at),
            (PathSeg::Payload, Known::Variant { payload, .. }) => payload.as_deref(),
            _ => None,
        })
    }

    fn bool(held: bool) -> Known {
        Known::Word(Word::Bool(held))
    }
}

/// The known values of a body as it stands.
pub struct KnownValues {
    values: FxHashMap<ValueId, Known>,
}

/// The two tag names an `Option`'s variants carry (RFC-0039).
struct OptionTags {
    some: Astr,
    none: Astr,
}

impl KnownValues {
    pub fn of(interner: &Interner, cfg: &CfgBody) -> KnownValues {
        let alive = reachable(cfg);
        let once = defined_once(cfg);
        let slots = Slots::of(cfg);
        let incoming = Incoming::of(cfg, &alive);
        let tags = OptionTags {
            some: interner.intern("Some"),
            none: interner.intern("None"),
        };
        let mut known = KnownValues {
            values: FxHashMap::default(),
        };
        let live_blocks = || {
            cfg.blocks
                .iter()
                .zip(&alive)
                .filter(|(_, alive)| **alive)
                .map(|(block, _)| block)
        };
        loop {
            let mut changed = false;
            for block in live_blocks() {
                let from = incoming.of_block(block.label);
                for (at, param) in block.params.iter().enumerate() {
                    if known.values.contains_key(param) || !once.contains(param) {
                        continue;
                    }
                    if let Some(value) = known.param(from, at) {
                        known.values.insert(*param, value);
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    let Some(dst) = sole_def(&inst.kind) else {
                        continue;
                    };
                    if known.values.contains_key(&dst) || !once.contains(&dst) {
                        continue;
                    }
                    if let Some(value) = known.eval(cfg, &slots, &tags, &inst.kind) {
                        known.values.insert(dst, value);
                        changed = true;
                    }
                }
            }
            if !changed {
                return known;
            }
        }
    }

    pub fn bool(&self, v: ValueId) -> Option<bool> {
        match self.values.get(&v)? {
            Known::Word(Word::Bool(held)) => Some(*held),
            _ => None,
        }
    }

    /// The key a `Switch` reading `tag`, typed `tag_ty`, compares against
    /// arms of `kind`'s kind, in the form `prepare::dispatch_form` reads it.
    pub fn switch_key(
        &self,
        interner: &Interner,
        tag: ValueId,
        tag_ty: &Ty,
        kind: SwitchKey,
    ) -> Option<SwitchKey> {
        match (kind, self.values.get(&tag)?) {
            (SwitchKey::Tag(_), Known::Variant { tag, .. }) => match scrutinee(tag_ty).as_ref() {
                Ty::Option(_) | Ty::Result(..) | Ty::Enum { .. } => Some(SwitchKey::Tag(*tag)),
                _ => None,
            },
            (SwitchKey::Int(_), Known::Word(Word::Int(n))) => match tag_ty {
                Ty::Int(_) => Some(SwitchKey::Int(*n)),
                _ => None,
            },
            (SwitchKey::Char(_), Known::Word(Word::Char(c))) => match tag_ty {
                Ty::Char => Some(SwitchKey::Char(*c)),
                _ => None,
            },
            (SwitchKey::Bool(_), Known::Word(Word::Bool(b))) => match tag_ty {
                Ty::Bool => Some(SwitchKey::Bool(*b)),
                _ => None,
            },
            (SwitchKey::Str(_), Known::Text(text)) => Some(SwitchKey::Str(interner.intern(text))),
            _ => None,
        }
    }

    fn param(&self, from: &Edges, at: usize) -> Option<Known> {
        let Edges::Args(edges) = from else {
            return None;
        };
        let mut args = edges.iter().map(|args| args.get(at));
        let first = self.values.get(args.next()??)?;
        for arg in args {
            if self.values.get(arg?)? != first {
                return None;
            }
        }
        Some(first.clone())
    }

    fn eval(
        &self,
        cfg: &CfgBody,
        slots: &Slots,
        tags: &OptionTags,
        kind: &InstKind,
    ) -> Option<Known> {
        let known = |v: &ValueId| self.values.get(v);
        let ty = |v: &ValueId| cfg.val_types.get(v);
        match kind {
            InstKind::Const {
                value: Literal::String(text),
                ..
            } => Some(Known::Text(text.clone())),
            InstKind::Const { dst, value } => Some(Known::Word(Word::of_constant(value, ty(dst)?)?)),
            InstKind::ConstStr { text, .. } => Some(Known::Text(text.clone())),
            InstKind::StringConcat { parts, .. } => {
                let texts = parts
                    .iter()
                    .map(|part| match known(part)? {
                        Known::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Option<Vec<&str>>>()?;
                Some(Known::Text(texts.concat()))
            }
            InstKind::StringClone { src, .. } => match known(src)? {
                Known::Text(text) => Some(Known::Text(text.clone())),
                _ => None,
            },
            InstKind::StringEq { a, b, .. } => match (known(a)?, known(b)?) {
                (Known::Text(a), Known::Text(b)) => Some(Known::bool(a == b)),
                _ => None,
            },

            InstKind::MakeVariant { dst, tag, payload } => {
                let payload = match payload {
                    Some(part) => Some(Box::new(known(part)?.clone())),
                    None => None,
                };
                // Obligation across artifacts: `prepare::make_variant` builds
                // an `Option` from whether a payload is present and
                // `prepare::lay_variant` from the tag, so only a value on
                // which the two agree is known, and a test may read either.
                if let Ty::Option(_) = ty(dst)? {
                    let some = *tag == tags.some && payload.is_some();
                    let none = *tag == tags.none && payload.is_none();
                    if !(some || none) {
                        return None;
                    }
                }
                Some(Known::Variant { tag: *tag, payload })
            }
            InstKind::MakeObject { fields, .. } => {
                let fields = fields
                    .iter()
                    .map(|(key, part)| Some((*key, known(part)?.clone())))
                    .collect::<Option<FxHashMap<Astr, Known>>>()?;
                Some(Known::Object(fields))
            }
            InstKind::MakeTuple { elements, .. } => Some(Known::Tuple(self.all(elements)?)),
            InstKind::MakeArray { elements, .. } => Some(Known::Array(self.all(elements)?)),

            InstKind::Ref {
                dst, target, path, ..
            } => {
                // Decision: a reference to a place that holds a reference is
                // not known. `Known` holds one referent, and a reader of
                // such a value would strip one level where two are stacked.
                let Ty::Ref(_, place) = ty(dst)? else {
                    return None;
                };
                if let Ty::Ref(..) = place.ty().as_ref() {
                    return None;
                }
                self.place(slots, target)?.at(path).cloned()
            }
            InstKind::Take { target, path, .. } => self.place(slots, target)?.at(path).cloned(),

            InstKind::TestVariant { src, tag, .. } => {
                let Known::Variant { tag: held, payload } = known(src)? else {
                    return None;
                };
                let matches = match scrutinee(ty(src)?).as_ref() {
                    // `ops::variant::TestOption` compares whether the value
                    // is a `Some` with whether the tested tag is `Some`. The
                    // lowering tests an `Option` for `Some` or `None` only,
                    // and any other tag is left unfolded.
                    Ty::Option(_) if *tag == tags.some || *tag == tags.none => {
                        payload.is_some() == (*tag == tags.some)
                    }
                    Ty::Option(_) => return None,
                    Ty::Result(..) | Ty::Enum { .. } => held == tag,
                    _ => return None,
                };
                Some(Known::bool(matches))
            }
            InstKind::TestLiteral { src, value, .. } => {
                Some(Known::bool(test_literal(ty(src)?, known(src), value)?))
            }
            // `ops::pattern::TestObjectKey` answers whether the field's
            // position holds a value rather than the `Undef` a construction
            // silent on the field leaves. A known object was built from known
            // parts, none of them `Undef`, and a field it lacks is one its
            // construction was silent on.
            InstKind::TestObjectKey { src, key, .. } => match known(src)? {
                Known::Object(fields) => Some(Known::bool(fields.contains_key(key))),
                _ => None,
            },

            // `ops::variant::UnwrapVariant` takes the payload out of the value
            // itself and never through a reference. It turns a variant
            // without a payload into `()`, and that is left unfolded.
            InstKind::UnwrapVariant { src, .. } => {
                if let Ty::Ref(..) = ty(src)? {
                    return None;
                }
                match known(src)? {
                    Known::Variant {
                        payload: Some(payload),
                        ..
                    } => Some((**payload).clone()),
                    _ => None,
                }
            }
            InstKind::ObjectGet { object, key, .. } => {
                known(object)?.at(&[PathSeg::Field(*key)]).cloned()
            }
            InstKind::FieldGet {
                object,
                field,
                rest,
                ..
            } => {
                let path: Vec<PathSeg> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(PathSeg::Field)
                    .collect();
                known(object)?.at(&path).cloned()
            }
            InstKind::TupleIndex { tuple, index, .. } => match known(tuple)? {
                Known::Tuple(parts) => parts.get(*index).cloned(),
                _ => None,
            },
            InstKind::ArrayIndex { array, index, .. } => match known(array)? {
                Known::Array(parts) => parts.get(*index).cloned(),
                _ => None,
            },

            _ => None,
        }
    }

    fn all(&self, elements: &[ValueId]) -> Option<Vec<Known>> {
        elements
            .iter()
            .map(|part| self.values.get(part).cloned())
            .collect()
    }

    fn place(&self, slots: &Slots, target: &RefTarget) -> Option<&Known> {
        match target {
            RefTarget::Var(slot) => self.values.get(slots.stored.get(slot)?),
            RefTarget::Param(_) => None,
            RefTarget::Through(reference) => self.values.get(reference),
        }
    }
}

/// `prepare::test_literal`'s operation for `value`, run on `held`.
///
/// Decision: a float literal is not folded. `ops::pattern::TestFloat`
/// compares with `==` on the `f64` it reads, where `Word::Float` holds bits,
/// and a fold that reproduced the machine's `==` would be a second float
/// semantics to keep in step with it for no dispatch the lowering writes:
/// RFC-0051 gives a float no switch key.
fn test_literal(src: &Ty, held: Option<&Known>, value: &Literal) -> Option<bool> {
    match value.desugared() {
        // `ops::pattern::TestUnit` writes `true` without reading.
        Literal::Unit => Some(true),
        // `TestInt::<T>` reads the operand's own word at the operand's width,
        // never through a reference, and compares it with the literal as
        // written.
        Literal::Int(want) => match (src, held?) {
            (Ty::Int(_), Known::Word(Word::Int(n))) => Some(*n == want),
            _ => None,
        },
        Literal::Char(want) => match (src, held?) {
            (Ty::Char, Known::Word(Word::Char(c))) => Some(*c == want),
            _ => None,
        },
        Literal::Bool(want) => match held? {
            Known::Word(Word::Bool(b)) => Some(*b == want),
            _ => None,
        },
        Literal::String(want) => match held? {
            Known::Text(text) => Some(*text == want),
            _ => None,
        },
        Literal::Float(_) => None,
        // `prepare::test_literal` refuses a list, and `desugared` leaves
        // neither sugared form.
        Literal::List(_) | Literal::IntOf(_) | Literal::Bytes(_) => None,
    }
}

/// The type a reader of a `ty` operand tests, as `prepare::scrutinee_ty`
/// strips one reference.
fn scrutinee(ty: &Ty) -> std::borrow::Cow<'_, Ty> {
    match ty {
        Ty::Ref(_, referent) => referent.ty(),
        ty => std::borrow::Cow::Borrowed(ty),
    }
}

fn sole_def(kind: &InstKind) -> Option<ValueId> {
    match inst_info::defs(kind).as_slice() {
        [dst] => Some(*dst),
        _ => None,
    }
}

fn defined_once(cfg: &CfgBody) -> FxHashSet<ValueId> {
    let mut count: FxHashMap<ValueId, usize> = FxHashMap::default();
    let defs = cfg.blocks.iter().flat_map(|block| {
        block
            .params
            .iter()
            .copied()
            .chain(block.insts.iter().flat_map(|inst| inst_info::defs(&inst.kind)))
    });
    for def in defs.chain(cfg.entry_defs()) {
        *count.entry(def).or_default() += 1;
    }
    count
        .into_iter()
        .filter(|(_, n)| *n == 1)
        .map(|(v, _)| v)
        .collect()
}

/// What one instruction does to a storage slot of this body.
enum SlotEffect {
    /// Writes the whole slot with a value.
    Store { slot: ValueId, value: ValueId },
    /// Writes the slot some other way, or lends it to a write.
    Rewrite(ValueId),
    Nothing,
}

impl SlotEffect {
    /// `Ref`, `Take` and `Assign` are the instructions that name a
    /// `RefTarget`, and each case of them is decided here.
    fn of(kind: &InstKind) -> SlotEffect {
        match kind {
            InstKind::Assign {
                target: RefTarget::Var(slot),
                path,
                value,
                restores: false,
            } if path.is_empty() => SlotEffect::Store {
                slot: *slot,
                value: *value,
            },
            // A store at a path, and the store that puts back a place a
            // `Take` took out for a call (RFC-0041).
            InstKind::Assign {
                target: RefTarget::Var(slot),
                ..
            } => SlotEffect::Rewrite(*slot),
            // Obligation across passes: this is the only instruction that
            // makes a `&mut` to a slot of this body. A `Through` reborrow
            // starts from a reference that is already mutable, and the
            // checker admits no write through a shared one.
            InstKind::Ref {
                target: RefTarget::Var(slot),
                mutability: Mutability::Mut,
                ..
            } => SlotEffect::Rewrite(*slot),
            InstKind::Ref {
                target: RefTarget::Var(_),
                mutability: Mutability::Shared,
                ..
            }
            | InstKind::Take {
                target: RefTarget::Var(_),
                ..
            } => SlotEffect::Nothing,
            InstKind::Assign {
                target: RefTarget::Param(_) | RefTarget::Through(_),
                ..
            }
            | InstKind::Ref {
                target: RefTarget::Param(_) | RefTarget::Through(_),
                ..
            }
            | InstKind::Take {
                target: RefTarget::Param(_) | RefTarget::Through(_),
                ..
            } => SlotEffect::Nothing,
            _ => SlotEffect::Nothing,
        }
    }
}

/// The slots each read of which sees the one value stored into it, and
/// that value.
///
/// Obligation across passes: a slot with one whole store and no other
/// write holds that store's value at every read because
/// `validate::init_check`, run on the lowered body in `graph::lower`
/// before SSA, refuses a read no store reaches first. A read of a part a
/// `Take` moved out is refused by `validate::move_check`.
struct Slots {
    stored: FxHashMap<ValueId, ValueId>,
}

impl Slots {
    fn of(cfg: &CfgBody) -> Slots {
        let mut stores: FxHashMap<ValueId, Vec<ValueId>> = FxHashMap::default();
        let mut rewritten: FxHashSet<ValueId> = FxHashSet::default();
        for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
            match SlotEffect::of(&inst.kind) {
                SlotEffect::Store { slot, value } => stores.entry(slot).or_default().push(value),
                SlotEffect::Rewrite(slot) => {
                    rewritten.insert(slot);
                }
                SlotEffect::Nothing => {}
            }
        }
        let stored = stores
            .into_iter()
            .filter(|(slot, _)| !rewritten.contains(slot))
            .filter_map(|(slot, values)| match values.as_slice() {
                [value] => Some((slot, *value)),
                _ => None,
            })
            .collect();
        Slots { stored }
    }
}

/// Where a block's parameters come from.
enum Edges {
    /// The argument list of each live edge into the block, positioned as the
    /// block's parameters are, as a terminator's `args` are.
    Args(Vec<Vec<ValueId>>),
    /// A source this analysis does not read: the caller for the entry, a
    /// `For` filling its targets' leading parameters, a fall-through that
    /// passes none.
    Opaque,
}

struct Incoming {
    into: FxHashMap<Label, Edges>,
}

impl Incoming {
    fn of(cfg: &CfgBody, alive: &[bool]) -> Incoming {
        let mut incoming = Incoming {
            into: FxHashMap::default(),
        };
        if let Some(entry) = cfg.blocks.first() {
            incoming.opaque(entry.label);
        }
        for (at, block) in cfg.blocks.iter().enumerate() {
            if !alive[at] {
                continue;
            }
            match &block.terminator {
                Terminator::Jump { label, args } => incoming.edge(*label, args),
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
                    incoming.edge(*then_label, then_args);
                    incoming.edge(*else_label, else_args);
                }
                Terminator::Switch { arms, default, .. } => {
                    for (_, label, args) in arms {
                        incoming.edge(*label, args);
                    }
                    if let Some((label, args)) = default {
                        incoming.edge(*label, args);
                    }
                }
                Terminator::For { body, exit, .. } => {
                    incoming.opaque(*body);
                    incoming.opaque(*exit);
                }
                Terminator::Fallthrough => {
                    if let Some(next) = cfg.blocks.get(at + 1) {
                        incoming.opaque(next.label);
                    }
                }
                Terminator::Return { .. } | Terminator::Diverge => {}
            }
        }
        incoming
    }

    fn edge(&mut self, label: Label, args: &[ValueId]) {
        let edges = self
            .into
            .entry(label)
            .or_insert_with(|| Edges::Args(Vec::new()));
        if let Edges::Args(edges) = edges {
            edges.push(args.to_vec());
        }
    }

    fn opaque(&mut self, label: Label) {
        self.into.insert(label, Edges::Opaque);
    }

    /// A block no live edge enters has no argument to read.
    fn of_block(&self, label: Label) -> &Edges {
        self.into.get(&label).unwrap_or(&Edges::Opaque)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::IntTy;

    #[test]
    fn an_integer_constant_is_the_word_its_width_reads_back() {
        assert_eq!(
            Word::of_constant(&Literal::Int(255), &Ty::Int(IntTy::I8)),
            Some(Word::Int(-1))
        );
        assert_eq!(
            Word::of_constant(&Literal::Int(7), &Ty::Int(IntTy::U8)),
            Some(Word::Int(7))
        );
    }
}
