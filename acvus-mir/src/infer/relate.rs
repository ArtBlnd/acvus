//! The relations A speaks in: `eq`, `join`, `enter` (typeck-structure
//! section 2). Each either records what it learned or refuses in B's terms.
//!
//! `eq` and `join` share one walk. It returns the term both sides are made
//! into, and rebinds the last variable on each side's chain to it: a node
//! is never changed in place, so a variable is where a term grows or
//! becomes opaque, and two variables made one are linked, so that what one
//! learns later the other learns too.

use std::cmp::Ordering;

use acvus_utils::{Astr, QualifiedRef};

use super::cells::{End as CellEnd, Exceeds, Unequal, ValuesTerm};
use super::{End, Infer, Level, Term, VarState};
use super::closed::poly_head;
use super::Closed;
use crate::decl::{Base, Declarations, Head, VerifiedConversion};
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{FnValue, FnValues, Reach, ValuesVarId, is_transparent};
use crate::ty::{
    Effect, EffectTerm, Flows, IdentityTerm, IdentityVarId, Infer as InferPhase, Label, Mark, MarkTerm, Mutability,
    TyKind, TyVarId,
};

/// What a mismatch holds of each kind of term: the check's own terms while
/// solving, and what they froze to once the check is done
/// (`typeck::FrozenMismatch`).
pub trait MismatchTerms: std::fmt::Debug + Clone + PartialEq {
    type Ty: std::fmt::Debug + Clone + PartialEq;
    type TyVar: std::fmt::Debug + Clone + PartialEq;
    type Effect: std::fmt::Debug + Clone + PartialEq;
}

/// A mismatch's terms while solving: the check's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solving;

impl MismatchTerms for Solving {
    type Ty = Term;
    type TyVar = TyVarId;
    type Effect = EffectTerm<InferPhase>;
}

/// Each term of one mismatch taken to another representation of its terms.
pub trait MapMismatch<From: MismatchTerms, Into: MismatchTerms> {
    fn ty(&mut self, ty: From::Ty) -> Into::Ty;
    fn ty_var(&mut self, var: From::TyVar) -> Into::TyVar;
    fn effect(&mut self, effect: From::Effect) -> Into::Effect;
}

/// Why two terms do not relate: B's refusals (refusals).
#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch<T: MismatchTerms = Solving> {
    /// Two heads that are not one head.
    Heads { a: T::Ty, b: T::Ty },
    /// A variable would contain itself (occurs check).
    Occurs { var: T::TyVar, in_term: T::Ty },
    /// A field an opaque object lacks, which a transparent one would add.
    OpaqueLacksField { object: T::Ty, field: Astr },
    /// A variant a closed enum lacks (patterns §2).
    OpaqueLacksVariant { r#enum: T::Ty, tag: Astr },
    /// Two enums of two names (types §2).
    TwoEnums { a: QualifiedRef, b: QualifiedRef },
    /// A variable's bound does not admit the term.
    Bound { var: T::TyVar, term: T::Ty },
    /// No subtyping relates a value and its slot, and not exactly one
    /// declared conversion takes the value's head to the slot's: these are
    /// the ones that do, none or several (types §7).
    Conversions { candidates: Box<[QualifiedRef]> },
    /// A reference whose target would be a reference (types §4).
    ReferenceToReference { target: T::Ty },
    /// Two function types whose effects differ, or whose effect one side
    /// fixed and the other would grow past (types §2, D94): the two effect
    /// terms, and what each is.
    Effects {
        a: T::Effect,
        b: T::Effect,
        a_is: Effect,
        b_is: Effect,
    },
    /// A function value one side may be that the other, which may not grow
    /// to it, may not: `value` is not one of `of` (types §1-2).
    NotOneOf { value: FnValue, of: FnValues },
    /// Two function types `eq` made one whose function values differ, each
    /// as known so far (types §2).
    Values { a: Option<FnValues>, b: Option<FnValues> },
    /// Two marks that are not one: a mark fixed marked, by a `#`
    /// parameter, that a use would raise (types §8).
    Mark { a: T::Ty, b: T::Ty },
    /// An identity made in a body instantiation would enter what was made
    /// outside it, with no ancestor carrying no identity to take instead
    /// (types §3).
    IdentityEscapes { term: T::Ty },
}

impl<T: MismatchTerms> Mismatch<T> {
    /// Which relation failed, without its terms.
    pub fn shape(&self) -> &'static str {
        match self {
            Mismatch::Heads { .. } => "Heads",
            Mismatch::Occurs { .. } => "Occurs",
            Mismatch::OpaqueLacksField { .. } => "OpaqueLacksField",
            Mismatch::OpaqueLacksVariant { .. } => "OpaqueLacksVariant",
            Mismatch::TwoEnums { .. } => "TwoEnums",
            Mismatch::Bound { .. } => "Bound",
            Mismatch::Conversions { .. } => "Conversions",
            Mismatch::ReferenceToReference { .. } => "ReferenceToReference",
            Mismatch::Effects { .. } => "Effects",
            Mismatch::NotOneOf { .. } => "NotOneOf",
            Mismatch::Values { .. } => "Values",
            Mismatch::Mark { .. } => "Mark",
            Mismatch::IdentityEscapes { .. } => "IdentityEscapes",
        }
    }

    /// This mismatch with each of its terms taken through `map`, every
    /// other part as it is.
    pub fn map<Into: MismatchTerms>(self, map: &mut impl MapMismatch<T, Into>) -> Mismatch<Into> {
        match self {
            Mismatch::Heads { a, b } => Mismatch::Heads { a: map.ty(a), b: map.ty(b) },
            Mismatch::Occurs { var, in_term } => Mismatch::Occurs { var: map.ty_var(var), in_term: map.ty(in_term) },
            Mismatch::OpaqueLacksField { object, field } => Mismatch::OpaqueLacksField { object: map.ty(object), field },
            Mismatch::OpaqueLacksVariant { r#enum, tag } => Mismatch::OpaqueLacksVariant { r#enum: map.ty(r#enum), tag },
            Mismatch::TwoEnums { a, b } => Mismatch::TwoEnums { a, b },
            Mismatch::Bound { var, term } => Mismatch::Bound { var: map.ty_var(var), term: map.ty(term) },
            Mismatch::Conversions { candidates } => Mismatch::Conversions { candidates },
            Mismatch::ReferenceToReference { target } => Mismatch::ReferenceToReference { target: map.ty(target) },
            Mismatch::Effects { a, b, a_is, b_is } => Mismatch::Effects { a: map.effect(a), b: map.effect(b), a_is, b_is },
            Mismatch::NotOneOf { value, of } => Mismatch::NotOneOf { value, of },
            Mismatch::Values { a, b } => Mismatch::Values { a, b },
            Mismatch::Mark { a, b } => Mismatch::Mark { a: map.ty(a), b: map.ty(b) },
            Mismatch::IdentityEscapes { term } => Mismatch::IdentityEscapes { term: map.ty(term) },
        }
    }
}

/// What an `enter` did (types §1, §3, §7).
#[derive(Debug, Clone, PartialEq)]
pub enum Entered {
    /// The slot took the value's type, or folded it up the tree.
    Folded,
    /// The value converts to the slot through one declared conversion.
    Converted(VerifiedConversion),
    /// The slot is a transparent structured type written elsewhere, and
    /// the value one of the same kind: the value is coerced by its kind's
    /// shape, and each part it keeps enters the slot's part in turn
    /// (types §1, §3). The parts are entered by whoever fires them, once
    /// each part's head is closed.
    Coerced(Coerced),
}

/// A structured value coerced to a slot of its kind written elsewhere
/// (types §3): what its shape did, and each part it keeps with the slot's
/// part that part enters.
#[derive(Debug, Clone, PartialEq)]
pub struct Coerced {
    pub shape: ShapeCoercion,
    /// `Some` where the coercion is of a reference's target, each part a
    /// reference of this mutability into the original storage (ownership
    /// §5); `None` where it is of the value itself.
    pub through: Option<Mutability>,
    pub parts: Box<[CoercedPart]>,
}

/// What a coercion did to a structure's shape (types §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeCoercion {
    /// A tuple, an array, `Option` and `Result`, or an object or an enum
    /// already of the slot's members.
    Kept,
    /// An object left the fields the slot does not name.
    Narrowed { left: Box<[Astr]> },
    /// An enum took the variants the slot names and the value does not.
    Widened { taken: Box<[Astr]> },
}

/// One part a coercion keeps: its key, the value's part, and the slot's
/// part it enters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoercedPart {
    pub key: PartKey,
    pub value: Term,
    pub slot: Term,
}

/// How two terms are related: made one, or joined with `a` reached by the
/// reach it carries and `b`, the value or the use's own form, by name
/// (types §1-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Eq,
    Join(Reach),
}

/// Which side of a relation a member lacks on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    A,
    B,
}

impl Relation {
    /// How `side` was reached, `None` under `eq`, where no side grows.
    fn reach_of(self, side: Side) -> Option<Reach> {
        match (self, side) {
            (Relation::Eq, _) => None,
            (Relation::Join(reach), Side::A) => Some(reach),
            (Relation::Join(_), Side::B) => Some(Reach::Named),
        }
    }
}

/// Why two identities are not one (types §3): two identities, or one made
/// in a body instantiation entering a variable made outside it, on `Side`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdentitiesApart {
    Two,
    Escapes(Side),
}

/// What a term holds that a variable's level bounds (types §3): each type
/// variable on its way and each identity variable at the end of its chain,
/// which take that level where theirs is higher, and whether it holds an
/// identity made above it, which escapes.
struct HeldBelow {
    ty_vars: Vec<TyVarId>,
    identity_vars: Vec<IdentityVarId>,
    escapes: bool,
}

/// A member of two sorted member lists: on both sides, or on one.
enum Member<T> {
    Both(Astr, T, T),
    OnlyA(Astr, T),
    OnlyB(Astr, T),
}

fn members<T: Copy>(a: &[(Astr, T)], b: &[(Astr, T)]) -> Vec<Member<T>> {
    let (mut a, mut b) = (a.iter().peekable(), b.iter().peekable());
    let mut out = Vec::new();
    loop {
        let member = match (a.peek(), b.peek()) {
            (None, None) => return out,
            (Some(&&(name, x)), None) => {
                a.next();
                Member::OnlyA(name, x)
            }
            (None, Some(&&(name, y))) => {
                b.next();
                Member::OnlyB(name, y)
            }
            (Some(&&(name_a, x)), Some(&&(name_b, y))) => match name_a.cmp(&name_b) {
                Ordering::Less => {
                    a.next();
                    Member::OnlyA(name_a, x)
                }
                Ordering::Greater => {
                    b.next();
                    Member::OnlyB(name_b, y)
                }
                Ordering::Equal => {
                    a.next();
                    b.next();
                    Member::Both(name_a, x, y)
                }
            },
        };
        out.push(member);
    }
}

/// `head` and every head up the tree from it, in order.
fn heads_up(decls: &Declarations, head: Head) -> Vec<Head> {
    let mut heads = vec![head];
    let mut at = head;
    while let Some(step) = decls.casts().step_up(at) {
        at = poly_head(decls.types(), step.rule().to);
        heads.push(at);
    }
    heads
}

/// Whether a value of `head` carries an identity (types §3): its type, or
/// a reference's target, is an extern-defined type that carries one.
fn carries_identity(decls: &Declarations, head: Head) -> bool {
    match head {
        Head::Base(Base::Named(id)) | Head::Ref(_, Base::Named(id)) => {
            decls.user_type(id).is_some_and(|declared| declared.carries_identity)
        }
        Head::Base(_) | Head::Ref(..) => false,
    }
}

/// Two shapes made one were written elsewhere where either was (types §1).
fn either_elsewhere(a: Label, b: Label) -> Label {
    match (a, b) {
        (Label::Here, Label::Here) => Label::Here,
        (Label::Elsewhere, _) | (_, Label::Elsewhere) => Label::Elsewhere,
    }
}

impl Infer {
    /// Unification with the occurs check; every constructor invariant
    /// (types §4).
    pub fn eq(&mut self, a: Term, b: Term) -> Result<(), Mismatch> {
        let related = self.relate(a, b, Relation::Eq).map(drop);
        self.propagate_effects();
        related
    }

    /// The join of two terms (types §2): a side grows by union toward the
    /// other only where it is transparent (types §1), `a` reached by
    /// `reach` and `b`, the use's own form, by name; afterwards the two are
    /// `eq`. Function types join their effects, flows, calls, split and
    /// function values.
    pub fn join(&mut self, a: Term, b: Term, reach: Reach) -> Result<(), Mismatch> {
        let joined = self.join_terms(a, b, reach);
        self.propagate_effects();
        joined
    }

    fn join_terms(&mut self, a: Term, b: Term, reach: Reach) -> Result<(), Mismatch> {
        let joined = self.relate(a, b, Relation::Join(reach))?;
        for side in [a, b] {
            if let End::Node { holder: None, node } = self.end(side)
                && !matches!(self.types.kind(node), TyKind::Never)
            {
                self.relate(side, joined, Relation::Eq)?;
            }
        }
        Ok(())
    }

    /// A value entering a slot (types §3, §7): once the value's head is
    /// closed, the slot folds it by ⊔ along the cast tree; where the two
    /// have no common ancestor, the value converts by the one declared
    /// conversion from its head to the slot's.
    /// The value is closed: a decision fires only on closed terms (types §6),
    /// and the caller checked with `closed`.
    /// The slot's type grows to hold the value where it is transparent at
    /// `reach`, how the slot was reached, and the value is coerced to it
    /// otherwise (types §1).
    ///
    /// `of` is the value's own term, `value` its closed head and node: the
    /// decision reads the head, and the value is related through `of`, so
    /// that a value folded into a slot is linked to it and the two are one
    /// type from then on, whatever either learns later (types §1).
    pub fn enter(&mut self, decls: &Declarations, of: Term, value: Closed, slot: Term, reach: Reach) -> Result<Entered, Mismatch> {
        let entered = self.enter_slot(decls, of, value, slot, reach);
        self.propagate_effects();
        entered
    }

    /// Whether `value`, a node, entering `slot` at `reach` is coerced by
    /// its kind's shape (types §3) rather than folded: the slot is a
    /// structured type not transparent there, or a reference to one
    /// written elsewhere. The decision reads the members of both, so it
    /// waits until neither can change (`fixed`).
    pub fn coerces(&self, value: Term, slot: Term, reach: Reach) -> bool {
        let End::Node { node: slot_node, .. } = self.end(slot) else {
            return false;
        };
        let opaque = structure(&self.types, slot_node).is_some_and(|slot| !is_transparent(slot.label(), reach));
        (opaque && structure(&self.types, value).is_some()) || self.through_reference(value, slot_node).is_some()
    }

    fn enter_slot(&mut self, decls: &Declarations, of: Term, value: Closed, slot: Term, reach: Reach) -> Result<Entered, Mismatch> {
        let (holder, slot_node) = match self.end(slot) {
            // `!` is the bottom type (types §4): a slot holds it whatever
            // it becomes, so it decides nothing, and the slot is left to
            // what else enters it or to its default (types §5).
            End::Unbound(..) if value.head == Head::Base(Base::Never) => return Ok(Entered::Folded),
            // A value holding an identity made above the slot's level
            // takes its nearest ancestor carrying no identity (types §3).
            End::Unbound(var, _) if self.held_below(self.ty_levels[var.0 as usize], value.term).escapes => {
                let Some(target) = heads_up(decls, value.head).into_iter().find(|head| !carries_identity(decls, *head)) else {
                    return Err(Mismatch::IdentityEscapes { term: value.term });
                };
                let climbed = self.climb(decls, value.term, value.head, target)?;
                self.bind(var, None, climbed)?;
                return Ok(Entered::Folded);
            }
            // The slot takes the value's type by being linked to the
            // value's last variable where it has one (types §1).
            End::Unbound(var, _) => {
                let value_holder = match self.end(of) {
                    End::Node { holder, .. } => holder,
                    End::Unbound(..) => None,
                };
                self.bind(var, value_holder, value.term)?;
                return Ok(Entered::Folded);
            }
            End::Node { holder, node } => (holder, node),
        };
        let opaque = structure(&self.types, slot_node).is_some_and(|slot| !is_transparent(slot.label(), reach));
        if opaque && structure(&self.types, value.term).is_some() {
            return self.coerce_structure(value.term, slot_node).map(Entered::Coerced);
        }
        if let Some((value_target, slot_target, through)) = self.through_reference(value.term, slot_node) {
            return self.coerce_structure_through(value_target, slot_target, through).map(Entered::Coerced);
        }
        let mismatch = Mismatch::Heads {
            a: value.term,
            b: slot_node,
        };
        let never = Head::Base(Base::Never);
        let target = match self.head_of(slot_node) {
            None => None,
            Some(slot_head) if value.head == never || slot_head == never => None,
            Some(slot_head) => {
                let apart = self.identities_apart(value.term, slot_node, holder);
                match (self.nearest_common_ancestor(decls, value, slot_head, apart.is_some()), apart) {
                    (Some(target), _) => Some((slot_head, target)),
                    (None, Some(IdentitiesApart::Escapes(_))) => return Err(Mismatch::IdentityEscapes { term: value.term }),
                    (None, _) => return self.convert(decls, value, (slot_head, reach), holder, slot_node),
                }
            }
        };
        // Folded at one head, neither side climbing, the value is of the
        // slot's type: it is linked to the slot. Where either climbs, the
        // two are of two types, and the steps between are read at
        // writeback.
        let one_head = matches!(target, Some((slot_head, at)) if at == slot_head && at == value.head);
        let (value_at, slot_at) = match target {
            None => (value.term, slot_node),
            Some((slot_head, target)) => (
                self.climb(decls, value.term, value.head, target)?,
                self.climb(decls, slot_node, slot_head, target)?,
            ),
        };
        self.settle_slot(holder, slot_node, slot_at, value_at, reach, mismatch)?;
        if one_head {
            self.link_to_slot(of, slot)?;
        }
        Ok(Entered::Folded)
    }

    /// Links the last variable on `of`'s chain to `slot`'s, so that the
    /// value and the slot are one type from then on (types §1). A value no
    /// variable holds is a node written once, which nothing grows.
    fn link_to_slot(&mut self, of: Term, slot: Term) -> Result<(), Mismatch> {
        let End::Node { holder: Some((value_var, _)), .. } = self.end(of) else {
            return Ok(());
        };
        let to = match self.end(slot) {
            End::Node { holder: Some((slot_var, slot_term)), .. } if slot_var != value_var => slot_term,
            End::Node { holder: Some(_), .. } => return Ok(()),
            End::Node { holder: None, node } => node,
            End::Unbound(..) => panic!("link_to_slot: the slot was settled to a node"),
        };
        self.rebind(value_var, to)
    }

    /// A structured value entering a slot of a transparent structured type
    /// written elsewhere (types §1): coerced by the slot's kind's shape
    /// (types §3). An object narrows to the fields the slot names, which
    /// the value holds each of; an enum widens to the variants the slot
    /// names, which name each of the value's; a tuple keeps its arity, an
    /// array its length, and `Option` and `Result` their variants. Two
    /// kinds that differ, or a shape the rule does not take to the slot's,
    /// are refused as two heads.
    fn coerce_structure(&mut self, value: Term, slot: Term) -> Result<Coerced, Mismatch> {
        let heads = Mismatch::Heads { a: value, b: slot };
        let kept = |pairs: Vec<(PartKey, Term, Term)>| Coerced {
            shape: ShapeCoercion::Kept,
            through: None,
            parts: pairs.into_iter().map(|(key, value, slot)| CoercedPart { key, value, slot }).collect(),
        };
        let (Some(of_value), Some(of_slot)) = (structure(&self.types, value), structure(&self.types, slot)) else {
            panic!("coerce_structure: `enter` routes only a structured value into a structured slot here");
        };
        match (of_value, of_slot) {
            (Structure::Tuple(xs, _), Structure::Tuple(ys, _)) => {
                if xs.len() != ys.len() {
                    return Err(heads);
                }
                let pairs = xs.iter().zip(ys.iter()).enumerate().map(|(at, (x, y))| (PartKey::Index(at), *x, *y)).collect();
                Ok(kept(pairs))
            }
            (Structure::Array { element: x, len: len_x, .. }, Structure::Array { element: y, len: len_y, .. }) => {
                self.len_vars.unify(&len_x, &len_y).map_err(|Unequal| heads)?;
                Ok(kept(vec![(PartKey::Element, x, y)]))
            }
            (Structure::Option(x, _), Structure::Option(y, _)) => Ok(kept(vec![(PartKey::Some, x, y)])),
            (Structure::Result { ok: ok_x, err: err_x, .. }, Structure::Result { ok: ok_y, err: err_y, .. }) => {
                Ok(kept(vec![(PartKey::Ok, ok_x, ok_y), (PartKey::Err, err_x, err_y)]))
            }
            (Structure::Object(fields_x, _), Structure::Object(fields_y, _)) => {
                let mut parts = Vec::new();
                let mut left = Vec::new();
                for member in members(fields_x, fields_y) {
                    match member {
                        Member::Both(name, x, y) => parts.push(CoercedPart { key: PartKey::Field(name), value: x, slot: y }),
                        Member::OnlyA(name, _) => left.push(name),
                        Member::OnlyB(..) => return Err(heads),
                    }
                }
                let shape = match left.is_empty() {
                    true => ShapeCoercion::Kept,
                    false => ShapeCoercion::Narrowed { left: left.into() },
                };
                Ok(Coerced { shape, through: None, parts: parts.into() })
            }
            (Structure::Enum(variants_x, _), Structure::Enum(variants_y, _)) => {
                if self.enum_name(value) != self.enum_name(slot) {
                    return Err(heads);
                }
                let mut parts = Vec::new();
                let mut taken = Vec::new();
                for member in members(variants_x, variants_y) {
                    match member {
                        Member::Both(tag, Some(x), Some(y)) => parts.push(CoercedPart { key: PartKey::Variant(tag), value: x, slot: y }),
                        Member::Both(_, None, None) => {}
                        Member::OnlyB(tag, _) => taken.push(tag),
                        Member::Both(..) | Member::OnlyA(..) => return Err(heads),
                    }
                }
                let shape = match taken.is_empty() {
                    true => ShapeCoercion::Kept,
                    false => ShapeCoercion::Widened { taken: taken.into() },
                };
                Ok(Coerced { shape, through: None, parts: parts.into() })
            }
            _ => Err(heads),
        }
    }

    /// The name of the enum `node` is.
    ///
    /// # Panics
    /// `node` is no enum: callers take it from an enum's structure.
    fn enum_name(&self, node: Term) -> QualifiedRef {
        match self.types.kind(node) {
            TyKind::Enum { name, .. } => *name,
            other => panic!("enum_name: an enum's structure is an enum's, not {other:?}"),
        }
    }

    /// Where the slot is `&m S`, `S` a transparent structured type written
    /// elsewhere, and the value `&m' S'`, `S'` one too, with `m'` reaching
    /// `m`: the two targets and `m`. A reference to a structured value is the
    /// structure of references to its parts (ownership §5), so it is coerced
    /// by its target's shape.
    fn through_reference(&self, value: Term, slot: Term) -> Option<(Term, Term, Mutability)> {
        let (TyKind::Ref(held, value_target), TyKind::Ref(through, slot_target)) = (self.types.kind(value), self.types.kind(slot)) else {
            return None;
        };
        let reaches = *held == Mutability::Mut || *through == Mutability::Shared;
        let (End::Node { node: value_target, .. }, End::Node { node: slot_target, .. }) =
            (self.end(*value_target), self.end(*slot_target))
        else {
            return None;
        };
        let slot_elsewhere = structure(&self.types, slot_target).is_some_and(|target| target.label() == Label::Elsewhere);
        let value_structured = structure(&self.types, value_target).is_some();
        (reaches && slot_elsewhere && value_structured).then_some((value_target, slot_target, *through))
    }

    /// The target `value` of a reference coerced to the target `slot` of a
    /// reference of mutability `through` (types §3, ownership §5): by
    /// `slot`'s kind's shape as `coerce_structure`, each kept part paired as
    /// a reference of `through` to the value's part and to the slot's part,
    /// which enter in turn. Through a `&mut`, an enum does not widen: the
    /// place of its variant is a part, a `&mut` to it, and a `&mut`'s
    /// target is never coerced.
    fn coerce_structure_through(&mut self, value: Term, slot: Term, through: Mutability) -> Result<Coerced, Mismatch> {
        let Coerced { shape, parts, .. } = self.coerce_structure(value, slot)?;
        if through == Mutability::Mut && matches!(shape, ShapeCoercion::Widened { .. }) {
            return Err(Mismatch::Heads { a: value, b: slot });
        }
        let parts = parts
            .iter()
            .map(|part| CoercedPart {
                key: part.key,
                value: self.part_reference(through, part.value),
                slot: self.part_reference(through, part.slot),
            })
            .collect();
        Ok(Coerced { shape, through: Some(through), parts })
    }

    /// A reference of `through` to a part of type `part`. A part that is a
    /// reference is reborrowed at the weaker of the two mutabilities, since
    /// a reference's target is never a reference (types §4).
    fn part_reference(&mut self, through: Mutability, part: Term) -> Term {
        let reborrowed = match self.end(part) {
            End::Node { node, .. } => match self.types.kind(node) {
                TyKind::Ref(held, target) => Some((*held, *target)),
                _ => None,
            },
            End::Unbound(..) => None,
        };
        let kind = match reborrowed {
            Some((Mutability::Mut, target)) if through == Mutability::Mut => TyKind::Ref(Mutability::Mut, target),
            Some((_, target)) => TyKind::Ref(Mutability::Shared, target),
            None => TyKind::Ref(through, part),
        };
        self.intern(kind)
    }

    /// A value of no common ancestor with its slot converts by the one
    /// declared conversion from its head to the slot's (types §7): the
    /// conversion's `from` pattern is made one with the value, and its `to`
    /// pattern enters the slot as a value of the slot's head.
    fn convert(
        &mut self,
        decls: &Declarations,
        value: Closed,
        (slot_head, reach): (Head, Reach),
        holder: Option<(TyVarId, Term)>,
        slot_node: Term,
    ) -> Result<Entered, Mismatch> {
        let applying: Vec<&VerifiedConversion> = decls
            .conversions_from(value.head)
            .iter()
            .filter(|conversion| poly_head(decls.types(), conversion.rule().to) == slot_head)
            .collect();
        let [conversion] = applying[..] else {
            return Err(Mismatch::Conversions {
                candidates: applying.iter().map(|conversion| conversion.rule().via).collect(),
            });
        };
        let rule = conversion.rule();
        let (pattern_from, pattern_to) = self.instantiate_ends(decls.types(), rule.from, rule.to);
        // A via gives its result as a call does: a mark written marked there
        // starts fresh, raised by the uses that take it unmarked (types §8).
        let pattern_to = self.fresh_origin_marks(pattern_to);
        self.relate(pattern_from, value.term, Relation::Eq)?;
        let mismatch = Mismatch::Heads {
            a: value.term,
            b: slot_node,
        };
        self.settle_slot(holder, slot_node, slot_node, pattern_to, reach, mismatch)?;
        Ok(Entered::Converted(conversion.clone()))
    }

    /// The slot takes the join of `slot_at`, reached by `reach`, and
    /// `value_at`: its variable is rebound to it, or a slot no variable
    /// holds must already be it.
    fn settle_slot(
        &mut self,
        holder: Option<(TyVarId, Term)>,
        slot_node: Term,
        slot_at: Term,
        value_at: Term,
        reach: Reach,
        mismatch: Mismatch,
    ) -> Result<(), Mismatch> {
        let made = self.relate(slot_at, value_at, Relation::Join(reach))?;
        match holder {
            Some((var, _)) => self.rebind(var, made),
            None => self.relate(slot_node, made, Relation::Eq).map(drop).map_err(|failed| match failed {
                kept @ (Mismatch::Effects { .. } | Mismatch::Mark { .. }) => kept,
                _ => mismatch,
            }),
        }
    }

    /// Where `value` and a slot of `slot_head` meet (types §3): the first
    /// head up from the value's that is also up from the slot's; where
    /// their identities are `apart`, the first such head that carries none.
    /// `None` where there is no such head.
    fn nearest_common_ancestor(&self, decls: &Declarations, value: Closed, slot_head: Head, apart: bool) -> Option<Head> {
        let value_chain = heads_up(decls, value.head);
        let slot_chain = heads_up(decls, slot_head);
        let mut meeting = value_chain
            .into_iter()
            .filter(|head| slot_chain.contains(head))
            .filter(|head| !apart || !carries_identity(decls, *head));
        meeting.next()
    }

    /// Whether `value` entering `slot_node`, held by `holder`, keeps no one
    /// identity (types §3): the two carry two identities, or the value
    /// holds one made above the holder's level, or above the level of the
    /// slot's identity variable. Identities that may be one are made one.
    fn identities_apart(&mut self, value: Term, slot_node: Term, holder: Option<(TyVarId, Term)>) -> Option<IdentitiesApart> {
        if let Some((var, _)) = holder
            && self.held_below(self.ty_levels[var.0 as usize], value).escapes
        {
            return Some(IdentitiesApart::Escapes(Side::A));
        }
        match (self.identity_of(value), self.identity_of(slot_node)) {
            (Some(x), Some(y)) => self.unify_identities(&x, &y).err(),
            _ => None,
        }
    }

    /// Makes two identities one (types §3): two known identities are one
    /// only where they are the same; a variable takes a known identity
    /// only where it was made at or below the variable's level; two
    /// variables made one are of the lower of their levels.
    fn unify_identities(&mut self, a: &IdentityTerm<InferPhase>, b: &IdentityTerm<InferPhase>) -> Result<IdentityTerm<InferPhase>, IdentitiesApart> {
        let (end_a, end_b) = (self.identity_vars.end(a), self.identity_vars.end(b));
        let known = |end: &CellEnd<IdentityTerm<InferPhase>>| match end {
            CellEnd::Known(identity) | CellEnd::Var { value: Some(identity), .. } => Some(*identity),
            CellEnd::Var { value: None, .. } => None,
        };
        let open = |end: &CellEnd<IdentityTerm<InferPhase>>| match end {
            CellEnd::Var { var, value: None } => Some(*var),
            CellEnd::Known(_) | CellEnd::Var { value: Some(_), .. } => None,
        };
        for (var, identity, side) in [(open(&end_a), known(&end_b), Side::B), (open(&end_b), known(&end_a), Side::A)] {
            if let (Some(var), Some(identity)) = (var, identity)
                && self.made_at(identity) > self.identity_levels[var.0 as usize]
            {
                return Err(IdentitiesApart::Escapes(side));
            }
        }
        let lowest = [open(&end_a), open(&end_b)].into_iter().flatten().map(|var| self.identity_levels[var.0 as usize]).min();
        let one = self.identity_vars.unify(a, b).map_err(|Unequal| IdentitiesApart::Two)?;
        if let (CellEnd::Var { var, value: None }, Some(lowest)) = (self.identity_vars.end(&one), lowest) {
            let level = &mut self.identity_levels[var.0 as usize];
            *level = (*level).min(lowest);
        }
        Ok(one)
    }

    /// What `term` holds that a variable of `level` bounds, bound
    /// variables followed, a function type's result left out.
    fn held_below(&self, level: Level, term: Term) -> HeldBelow {
        let mut held = HeldBelow { ty_vars: Vec::new(), identity_vars: Vec::new(), escapes: false };
        let mut seen = rustc_hash::FxHashSet::default();
        let mut stack = vec![term];
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            let kind = self.types.kind(at);
            match kind {
                TyKind::Var(var) => {
                    held.ty_vars.push(*var);
                    if let VarState::Bound(to) = &self.ty_vars[var.0 as usize] {
                        stack.push(*to);
                    }
                }
                TyKind::UserDefined { identity: Some(identity), .. } => match self.identity_vars.end(identity) {
                    CellEnd::Known(made) | CellEnd::Var { value: Some(made), .. } => held.escapes |= self.made_at(made) > level,
                    CellEnd::Var { var, value: None } => held.identity_vars.push(var),
                },
                // A function type's result binds the identities its body
                // makes, fresh at each call (types §3): what it holds
                // there is its own, never the variable's it enters.
                TyKind::Fn { params, .. } => {
                    stack.extend(params.iter().copied());
                    continue;
                }
                _ => {}
            }
            kind.for_each_child(|child| stack.push(child));
        }
        held
    }

    /// A variable of `level` becoming `term` (types §3): every variable
    /// `term` holds takes `level` where its own is higher, so that what an
    /// outer variable holds is outer; refused where `term` holds an
    /// identity made above `level`, which would escape the body
    /// instantiation it belongs to.
    fn lower_into(&mut self, level: Level, term: Term) -> Result<(), Mismatch> {
        let held = self.held_below(level, term);
        if held.escapes {
            return Err(Mismatch::IdentityEscapes { term });
        }
        for var in held.ty_vars {
            let own = &mut self.ty_levels[var.0 as usize];
            *own = (*own).min(level);
        }
        for var in held.identity_vars {
            let own = &mut self.identity_levels[var.0 as usize];
            *own = (*own).min(level);
        }
        Ok(())
    }

    /// The identity a value of `node` carries: the one an extern-defined
    /// type carries, through references.
    fn identity_of(&self, node: Term) -> Option<IdentityTerm<InferPhase>> {
        match self.types.kind(node) {
            TyKind::UserDefined { identity, .. } => *identity,
            TyKind::Ref(_, target) => match self.end(*target) {
                End::Node { node, .. } => self.identity_of(node),
                End::Unbound(..) => None,
            },
            _ => None,
        }
    }

    /// `node`, of head `from`, stepped up the tree to `to` through each
    /// step: a cast's `from` pattern is made one with the type reached, and
    /// its `to` pattern is the type above. A cast is declared from an
    /// unmarked type, so a step taken from a marked value raises its mark
    /// first (types §8).
    ///
    /// # Panics
    /// `to` is not up from `from`: callers climb to a common ancestor.
    fn climb(&mut self, decls: &Declarations, node: Term, from: Head, to: Head) -> Result<Term, Mismatch> {
        let (mut at, mut head) = (node, from);
        while head != to {
            let step = decls
                .casts()
                .step_up(head)
                .expect("climb: the target is up the tree from the start");
            let rule = step.rule();
            let (pattern_from, pattern_to) = self.instantiate_ends(decls.types(), rule.from, rule.to);
        // A via gives its result as a call does: a mark written marked there
        // starts fresh, raised by the uses that take it unmarked (types §8).
        let pattern_to = self.fresh_origin_marks(pattern_to);
            self.raise_mark(at)?;
            self.relate(pattern_from, at, Relation::Eq)?;
            (at, head) = (pattern_to, poly_head(decls.types(), rule.to));
        }
        Ok(at)
    }

    /// The join of two effects (types §2). An effect `eq` fixed does not
    /// grow (D94).
    pub fn join_effects(
        &mut self,
        a: &EffectTerm<InferPhase>,
        b: &EffectTerm<InferPhase>,
    ) -> Result<EffectTerm<InferPhase>, Exceeds<Effect>> {
        let joined = self.effect_vars.join(a, b, Effect::join);
        self.propagate_effects();
        joined
    }

    fn relate(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        acvus_utils::grow(|| self.relate_level(a, b, how))
    }

    fn relate_level(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        if a == b {
            return Ok(a);
        }
        match (self.end(a), self.end(b)) {
            (End::Unbound(v, _), End::Unbound(w, w_term)) => {
                if v != w {
                    self.merge_unbound(v, w, w_term)?;
                }
                Ok(w_term)
            }
            // `!` is the bottom type (types §4): joined with a variable it
            // decides nothing, and the variable is left to what else it
            // meets or to its default (types §5), whichever came first.
            (End::Unbound(_, term), End::Node { node, .. }) | (End::Node { node, .. }, End::Unbound(_, term))
                if matches!(how, Relation::Join(_)) && matches!(self.types.kind(node), TyKind::Never) =>
            {
                Ok(term)
            }
            (End::Unbound(var, _), End::Node { holder, node }) | (End::Node { holder, node }, End::Unbound(var, _)) => {
                self.bind(var, holder, node)
            }
            (End::Node { holder: holder_a, node: a }, End::Node { holder: holder_b, node: b }) => {
                let made = match (how, self.types.kind(a), self.types.kind(b)) {
                    (Relation::Join(_), TyKind::Never, _) => b,
                    (Relation::Join(_), _, TyKind::Never) => a,
                    _ => self.relate_nodes(a, b, how)?,
                };
                self.settle(holder_a, holder_b, made)
            }
        }
    }

    /// Makes two unbound variables one: `v` is linked to `w`, which takes
    /// what both bounds admit.
    fn merge_unbound(&mut self, v: TyVarId, w: TyVarId, w_term: Term) -> Result<(), Mismatch> {
        let (VarState::Unbound(bound_v), VarState::Unbound(bound_w)) =
            (&self.ty_vars[v.0 as usize], &self.ty_vars[w.0 as usize])
        else {
            panic!("merge_unbound: both ends are unbound variables")
        };
        let bound = bound_v
            .meet(bound_w)
            .ok_or(Mismatch::Bound { var: v, term: w_term })?;
        self.lower_into(self.ty_levels[v.0 as usize], w_term)?;
        self.ty_vars[w.0 as usize] = VarState::Unbound(bound);
        self.ty_vars[v.0 as usize] = VarState::Bound(w_term);
        self.keep_exclusions(v, w_term);
        Ok(())
    }

    /// `var`, linked to `to`, passes its mark as a reference's target to
    /// the last variable on `to`'s chain, the one later rebound: two
    /// variables made one keep the mark if either has it (types §4).
    fn keep_exclusions(&mut self, var: TyVarId, to: Term) {
        let last = match self.end(to) {
            End::Unbound(last, _) | End::Node { holder: Some((last, _)), .. } => last,
            End::Node { holder: None, .. } => return,
        };
        if self.reference_targets.contains(&var) {
            self.reference_targets.insert(last);
        }
        if self.unmarked.contains(&var) {
            self.unmarked.insert(last);
        }
    }

    /// Raises the mark of `term`'s extern-defined type, or of what the
    /// reference `term` lends, to unmarked (types §8): refused where a `#`
    /// parameter fixed it marked.
    fn raise_mark(&mut self, term: Term) -> Result<(), Mismatch> {
        let End::Node { node, .. } = self.end(term) else {
            return Ok(());
        };
        match self.types.kind(node).clone() {
            // A mark written marked, a `#` parameter's, is fixed as a
            // known effect is, and joining it would leave it as it is.
            TyKind::UserDefined { marked: MarkTerm::Known(Mark::Marked), .. } => Err(Mismatch::Mark { a: node, b: node }),
            TyKind::UserDefined { marked, .. } => self
                .mark_vars
                .join(&marked, &MarkTerm::Known(Mark::Unmarked), Mark::join)
                .map(drop)
                .map_err(|_| Mismatch::Mark { a: node, b: node }),
            TyKind::Ref(_, target) => self.raise_mark(target),
            _ => Ok(()),
        }
    }

    /// What a declaration's type variable stands for is unmarked wherever it
    /// would hold a marked type as itself or as a part (types §8): each mark
    /// it holds is raised, and each variable it holds is kept so. What a
    /// reference lends and a function type's parameters and result are not
    /// parts it holds.
    fn demand_unmarked(&mut self, term: Term) -> Result<(), Mismatch> {
        acvus_utils::grow(|| match self.end(term) {
            End::Unbound(var, _) => {
                self.unmarked.insert(var);
                Ok(())
            }
            End::Node { node, .. } => match self.types.kind(node).clone() {
                TyKind::UserDefined { type_args, .. } => {
                    self.raise_mark(node)?;
                    type_args.iter().try_for_each(|arg| self.demand_unmarked(*arg))
                }
                TyKind::Ref(..) | TyKind::Fn { .. } => Ok(()),
                kind => {
                    let mut parts = Vec::new();
                    kind.for_each_child(|part| parts.push(part));
                    parts.into_iter().try_for_each(|part| self.demand_unmarked(part))
                }
            },
        })
    }

    /// Refuses `var` becoming `to` where `var` is a reference's target and
    /// `to` is a reference (types §4).
    fn refuse_excluded(&self, var: TyVarId, to: Term) -> Result<(), Mismatch> {
        let End::Node { node, .. } = self.end(to) else {
            return Ok(());
        };
        let kind = self.types.kind(node);
        if self.reference_targets.contains(&var) && matches!(kind, TyKind::Ref(..)) {
            return Err(Mismatch::ReferenceToReference { target: node });
        }
        Ok(())
    }

    /// Binds the unbound `var` to `node`, through the variable that holds
    /// it where one does.
    fn bind(&mut self, var: TyVarId, holder: Option<(TyVarId, Term)>, node: Term) -> Result<Term, Mismatch> {
        if self.occurs(var, node) {
            return Err(Mismatch::Occurs { var, in_term: node });
        }
        let VarState::Unbound(bound) = &self.ty_vars[var.0 as usize] else {
            panic!("bind: the variable is an unbound end")
        };
        if !bound.admits(self.types.kind(node)) {
            return Err(Mismatch::Bound { var, term: node });
        }
        self.refuse_excluded(var, node)?;
        let target = holder.map_or(node, |(_, held)| held);
        self.lower_into(self.ty_levels[var.0 as usize], target)?;
        self.ty_vars[var.0 as usize] = VarState::Bound(target);
        self.keep_exclusions(var, target);
        if self.unmarked.contains(&var) {
            self.demand_unmarked(target)?;
        }
        Ok(target)
    }

    /// Records `made` as what both sides are now: each side's last variable
    /// is rebound to it, and the second linked to the first.
    fn settle(
        &mut self,
        holder_a: Option<(TyVarId, Term)>,
        holder_b: Option<(TyVarId, Term)>,
        made: Term,
    ) -> Result<Term, Mismatch> {
        match (holder_a, holder_b) {
            (Some((a, a_term)), Some((b, _))) => {
                self.rebind(a, made)?;
                if a != b {
                    self.rebind(b, a_term)?;
                }
                Ok(a_term)
            }
            (Some((var, term)), None) | (None, Some((var, term))) => {
                self.rebind(var, made)?;
                Ok(term)
            }
            (None, None) => Ok(made),
        }
    }

    fn rebind(&mut self, var: TyVarId, to: Term) -> Result<(), Mismatch> {
        if self.ty_vars[var.0 as usize] == VarState::Bound(to) {
            return Ok(());
        }
        if self.occurs(var, to) {
            return Err(Mismatch::Occurs { var, in_term: to });
        }
        self.refuse_excluded(var, to)?;
        self.lower_into(self.ty_levels[var.0 as usize], to)?;
        self.ty_vars[var.0 as usize] = VarState::Bound(to);
        self.keep_exclusions(var, to);
        if self.unmarked.contains(&var) {
            self.demand_unmarked(to)?;
        }
        Ok(())
    }

    /// Whether `var` occurs in `term`, bound variables followed.
    fn occurs(&self, var: TyVarId, term: Term) -> bool {
        let mut seen = rustc_hash::FxHashSet::default();
        let mut stack = vec![term];
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            match self.types.kind(at) {
                TyKind::Var(found) if *found == var => return true,
                TyKind::Var(found) => {
                    if let VarState::Bound(to) = &self.ty_vars[found.0 as usize] {
                        stack.push(*to);
                    }
                }
                kind => kind.for_each_child(|child| stack.push(child)),
            }
        }
        false
    }

    /// The node both nodes are made into. Under `join` only a node's own
    /// member set grows, and a function type's effect, flows, calls and
    /// holds join (types §2); every part is related by `eq`, since every
    /// constructor is invariant (types §4).
    fn relate_nodes(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        if a == b {
            return Ok(a);
        }
        let heads = Mismatch::Heads { a, b };
        let kind = match (self.types.kind(a).clone(), self.types.kind(b).clone()) {
            (TyKind::Handle(x), TyKind::Handle(y)) => TyKind::Handle(self.relate(x, y, Relation::Eq)?),
            (TyKind::Array(x, len_x, label_x), TyKind::Array(y, len_y, label_y)) => {
                let element = self.relate(x, y, Relation::Eq)?;
                let len = self.len_vars.unify(&len_x, &len_y).map_err(|Unequal| heads)?;
                TyKind::Array(element, len, either_elsewhere(label_x, label_y))
            }
            (TyKind::Slice(x), TyKind::Slice(y)) => TyKind::Slice(self.relate(x, y, Relation::Eq)?),
            (TyKind::Option(x, label_x), TyKind::Option(y, label_y)) => {
                TyKind::Option(self.relate(x, y, Relation::Eq)?, either_elsewhere(label_x, label_y))
            }
            (TyKind::Result(ok_x, err_x, label_x), TyKind::Result(ok_y, err_y, label_y)) => TyKind::Result(
                self.relate(ok_x, ok_y, Relation::Eq)?,
                self.relate(err_x, err_y, Relation::Eq)?,
                either_elsewhere(label_x, label_y),
            ),
            (TyKind::Tuple(xs, label_x), TyKind::Tuple(ys, label_y)) => {
                if xs.len() != ys.len() {
                    return Err(heads);
                }
                let items = xs
                    .iter()
                    .zip(ys.iter())
                    .map(|(x, y)| self.relate(*x, *y, Relation::Eq))
                    .collect::<Result<_, _>>()?;
                TyKind::Tuple(items, either_elsewhere(label_x, label_y))
            }
            (TyKind::Object { fields: fields_x, label: label_x }, TyKind::Object { fields: fields_y, label: label_y }) => {
                self.relate_objects(a, b, (fields_x, label_x), (fields_y, label_y), how)?
            }
            (
                TyKind::Enum {
                    name: name_a,
                    variants: variants_a,
                    label: label_a,
                },
                TyKind::Enum {
                    name: name_b,
                    variants: variants_b,
                    label: label_b,
                },
            ) => {
                if name_a != name_b {
                    return Err(Mismatch::TwoEnums { a: name_a, b: name_b });
                }
                let variants = self.relate_variants((a, b), (&variants_a, label_a), (&variants_b, label_b), how)?;
                TyKind::Enum {
                    name: name_a,
                    variants,
                    label: either_elsewhere(label_a, label_b),
                }
            }
            (
                TyKind::Fn {
                    params: params_a,
                    ret: ret_a,
                    effect: effect_a,
                    flows: flows_a,
                    calls: calls_a,
                    holds: holds_a,
                    values: values_a,
                },
                TyKind::Fn {
                    params: params_b,
                    ret: ret_b,
                    effect: effect_b,
                    flows: flows_b,
                    calls: calls_b,
                    holds: holds_b,
                    values: values_b,
                },
            ) => {
                if params_a.len() != params_b.len() {
                    return Err(heads);
                }
                let params = params_a
                    .iter()
                    .zip(params_b.iter())
                    .map(|(x, y)| self.relate(*x, *y, Relation::Eq))
                    .collect::<Result<_, Mismatch>>()?;
                let ret = self.relate(ret_a, ret_b, Relation::Eq)?;
                let (effect, flows, calls, holds) = match how {
                    Relation::Eq => (
                        self.effect_vars
                            .unify(&effect_a, &effect_b)
                            .map_err(|Unequal| self.effects_mismatch(&effect_a, &effect_b))?,
                        self.flow_vars.unify(&flows_a, &flows_b).map_err(|Unequal| heads.clone())?,
                        self.calls_vars.unify(&calls_a, &calls_b).map_err(|Unequal| heads.clone())?,
                        self.holds_vars.unify(&holds_a, &holds_b).map_err(|Unequal| heads)?,
                    ),
                    Relation::Join(_) => (
                        self.effect_vars
                            .join(&effect_a, &effect_b, Effect::join)
                            .map_err(|_| self.effects_mismatch(&effect_a, &effect_b))?,
                        self.flow_vars.join(&flows_a, &flows_b, Flows::join).map_err(|_| heads.clone())?,
                        self.calls_vars.join(&calls_a, &calls_b, |a, b| a.join(*b)).map_err(|_| heads.clone())?,
                        self.holds_vars.join(&holds_a, &holds_b, |a, b| a.join(*b)).map_err(|_| heads)?,
                    ),
                };
                let values = self.relate_values(how, values_a, values_b)?;
                TyKind::Fn {
                    params,
                    ret,
                    effect,
                    flows,
                    calls,
                    holds,
                    values,
                }
            }
            (
                TyKind::UserDefined {
                    id: id_a,
                    type_args: type_args_a,
                    effect: effect_a,
                    identity: identity_a,
                    marked: marked_a,
                },
                TyKind::UserDefined {
                    id: id_b,
                    type_args: type_args_b,
                    effect: effect_b,
                    identity: identity_b,
                    marked: marked_b,
                },
            ) => {
                if id_a != id_b
                    || type_args_a.len() != type_args_b.len()
                    || effect_a.is_some() != effect_b.is_some()
                    || identity_a.is_some() != identity_b.is_some()
                {
                    return Err(heads);
                }
                let type_args = type_args_a
                    .iter()
                    .zip(type_args_b.iter())
                    .map(|(x, y)| self.relate(*x, *y, Relation::Eq))
                    .collect::<Result<_, _>>()?;
                let effect = effect_a
                    .zip(effect_b)
                    .map(|(effect_a, effect_b)| match how {
                        Relation::Eq => self
                            .effect_vars
                            .unify(&effect_a, &effect_b)
                            .map_err(|Unequal| self.effects_mismatch(&effect_a, &effect_b)),
                        Relation::Join(_) => self
                            .effect_vars
                            .join(&effect_a, &effect_b, Effect::join)
                            .map_err(|_| self.effects_mismatch(&effect_a, &effect_b)),
                    })
                    .transpose()?;
                let identity = identity_a
                    .zip(identity_b)
                    .map(|(x, y)| {
                        self.unify_identities(&x, &y).map_err(|apart| match apart {
                            IdentitiesApart::Two => heads,
                            IdentitiesApart::Escapes(Side::A) => Mismatch::IdentityEscapes { term: a },
                            IdentitiesApart::Escapes(Side::B) => Mismatch::IdentityEscapes { term: b },
                        })
                    })
                    .transpose()?;
                // Marks join as effects do (types §8): `eq` makes them one
                // and fixes them, a join raises both to the higher.
                let marked = match how {
                    Relation::Eq => self.mark_vars.unify(&marked_a, &marked_b).map_err(|Unequal| Mismatch::Mark { a, b })?,
                    Relation::Join(_) => self
                        .mark_vars
                        .join(&marked_a, &marked_b, Mark::join)
                        .map_err(|_| Mismatch::Mark { a, b })?,
                };
                TyKind::UserDefined {
                    id: id_a,
                    type_args,
                    effect,
                    identity,
                    marked,
                }
            }
            (TyKind::Ref(mutability_a, x), TyKind::Ref(mutability_b, y)) => {
                if mutability_a != mutability_b {
                    return Err(heads);
                }
                TyKind::Ref(mutability_a, self.relate(x, y, Relation::Eq)?)
            }
            (TyKind::Var(_), _) | (_, TyKind::Var(_)) => {
                unreachable!("relate_nodes: an end is never a variable")
            }
            _ => return Err(heads),
        };
        Ok(self.intern(kind))
    }

    /// Two effects that did not relate, each with what it is.
    ///
    /// # Panics
    /// Either is unknown: two effects fail to relate only where both are
    /// known, as two unequal values or a fixed value the other would grow.
    fn effects_mismatch(&self, a: &EffectTerm<InferPhase>, b: &EffectTerm<InferPhase>) -> Mismatch {
        let is = |term: &EffectTerm<InferPhase>| match self.effect_vars.end(term) {
            CellEnd::Known(value) | CellEnd::Var { value: Some(value), .. } => value,
            CellEnd::Var { value: None, .. } => panic!("two effects fail to relate only where both are known"),
        };
        Mismatch::Effects {
            a: a.clone(),
            b: b.clone(),
            a_is: is(a),
            b_is: is(b),
        }
    }

    fn relate_objects(
        &mut self,
        a: Term,
        b: Term,
        (fields_a, label_a): (Box<[(Astr, Term)]>, Label),
        (fields_b, label_b): (Box<[(Astr, Term)]>, Label),
        how: Relation,
    ) -> Result<TyKind<InferPhase>, Mismatch> {
        let mut fields = Vec::new();
        for member in members(&fields_a, &fields_b) {
            let field = match member {
                Member::Both(name, x, y) => (name, self.relate(x, y, Relation::Eq)?),
                Member::OnlyA(name, x) => {
                    Self::may_grow(how.reach_of(Side::B), (a, b), (b, label_b), |object| Mismatch::OpaqueLacksField {
                        object,
                        field: name,
                    })?;
                    (name, x)
                }
                Member::OnlyB(name, y) => {
                    Self::may_grow(how.reach_of(Side::A), (a, b), (a, label_a), |object| Mismatch::OpaqueLacksField {
                        object,
                        field: name,
                    })?;
                    (name, y)
                }
            };
            fields.push(field);
        }
        Ok(TyKind::Object { fields: fields.into(), label: either_elsewhere(label_a, label_b) })
    }

    /// Whether `lacking`, labelled `label` and reached by `reach`, may grow
    /// by a member the other side has: only under `join`, where `reach` is
    /// `Some`, and only where transparent (types §1-2). Under `eq` the two
    /// are one shape or not one head.
    fn may_grow(
        reach: Option<Reach>,
        (a, b): (Term, Term),
        (lacking, label): (Term, Label),
        opaque_lacks: impl FnOnce(Term) -> Mismatch,
    ) -> Result<(), Mismatch> {
        match reach {
            None => Err(Mismatch::Heads { a, b }),
            Some(reach) if is_transparent(label, reach) => Ok(()),
            Some(_) => Err(opaque_lacks(lacking)),
        }
    }

    /// Which function values two function types related by `how` are made
    /// (types §1-2). A side of `Any`, which a declaration or the host
    /// wrote, takes the other as a value entering it, coerced and its own
    /// values forgotten. Otherwise, under `eq` the two are one; under
    /// `join` they are made one at the union of their values, which a side
    /// may grow to only where it is transparent. The values a function
    /// type states are the program's own, written where each value is, so
    /// only how the side was reached decides.
    fn relate_values(&mut self, how: Relation, a: ValuesVarId, b: ValuesVarId) -> Result<ValuesVarId, Mismatch> {
        let (term_a, term_b) = (ValuesTerm::Var(a), ValuesTerm::Var(b));
        let (values_a, values_b) = (self.values(a), self.values(b));
        match (&values_a, &values_b) {
            (Some(FnValues::Any), _) => return Ok(a),
            (_, Some(FnValues::Any)) => return Ok(b),
            _ => {}
        }
        let (Some(reach_a), Some(reach_b)) = (how.reach_of(Side::A), how.reach_of(Side::B)) else {
            let one = self
                .values_vars
                .unify(&term_a, &term_b)
                .map_err(|Unequal| Mismatch::Values { a: values_a, b: values_b })?;
            return Ok(Self::values_var(one));
        };
        let joined = match (&values_a, &values_b) {
            (Some(x), Some(y)) => Some(x.join(y)),
            (Some(one), None) | (None, Some(one)) => Some(one.clone()),
            (None, None) => None,
        };
        if let Some(FnValues::Of(joined)) = &joined {
            for (values, reach) in [(&values_a, reach_a), (&values_b, reach_b)] {
                let Some(FnValues::Of(own)) = values else {
                    continue;
                };
                if let Some(value) = joined.difference(own).next()
                    && !is_transparent(Label::Here, reach)
                {
                    return Err(Mismatch::NotOneOf {
                        value: *value,
                        of: FnValues::Of(own.clone()),
                    });
                }
            }
        }
        let one = self
            .values_vars
            .join(&term_a, &term_b, FnValues::join)
            .expect("relate_values: a fixed side that would grow was refused above");
        Ok(Self::values_var(one))
    }

    /// The variable a values term made of variables is.
    ///
    /// # Panics
    /// `term` is known: `relate_values` relates variables, which stay
    /// variables.
    fn values_var(term: ValuesTerm) -> ValuesVarId {
        match term {
            ValuesTerm::Var(var) => var,
            ValuesTerm::Known(values) => panic!("values_var: two variables related are a variable, not {values:?}"),
        }
    }

    fn relate_variants(
        &mut self,
        (a, b): (Term, Term),
        (variants_a, label_a): (&[(Astr, Option<Term>)], Label),
        (variants_b, label_b): (&[(Astr, Option<Term>)], Label),
        how: Relation,
    ) -> Result<Box<[(Astr, Option<Term>)]>, Mismatch> {
        let mut variants = Vec::new();
        for member in members(variants_a, variants_b) {
            let variant = match member {
                Member::Both(tag, Some(x), Some(y)) => (tag, Some(self.relate(x, y, Relation::Eq)?)),
                Member::Both(tag, None, None) => (tag, None),
                Member::Both(_, Some(_), None) | Member::Both(_, None, Some(_)) => {
                    return Err(Mismatch::Heads { a, b });
                }
                Member::OnlyA(tag, x) => {
                    Self::may_grow(how.reach_of(Side::B), (a, b), (b, label_b), |r#enum| Mismatch::OpaqueLacksVariant { r#enum, tag })?;
                    (tag, x)
                }
                Member::OnlyB(tag, y) => {
                    Self::may_grow(how.reach_of(Side::A), (a, b), (a, label_a), |r#enum| Mismatch::OpaqueLacksVariant { r#enum, tag })?;
                    (tag, y)
                }
            };
            variants.push(variant);
        }
        Ok(variants.into())
    }
}
