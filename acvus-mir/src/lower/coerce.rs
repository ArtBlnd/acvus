//! A transparent structured value coerced to its slot by the tree typeck
//! recorded (types §3): a new value of the slot's type, built part by part.
//! An object keeps the fields its slot names and ends the others where it
//! is narrowed (ownership §6); an enum is the same variant of the wider
//! enum; an array is `Intrinsic::ArrayMap` over its elements, each coerced
//! alike; a tuple, an `Option` and a `Result` rebuild their parts in place.
//! A part is coerced by its own coercion, `call::coerce`.
//!
//! Through a reference (`through`), nothing is taken out and nothing
//! rebuilt: each kept part is a reference into the original storage
//! (`Op::Ref` through the value), coerced by its own coercion, and the
//! structure of those references is the reference to the result,
//! `Intrinsic::Gather` (ownership §5).

use acvus_ast::Span;
use acvus_utils::Astr;

use super::call::coerce;
use super::control::diamond;
use super::ctxt::{Declared, Entry, Lcx, Lowered};
use super::place::extract;
use crate::ir::cfg::Op;
use crate::ir::parts::step;
use crate::ir::{Intrinsic, PathSeg, RefTarget, ValueId, fn_effect, known_len};
use crate::ty::structural::{PartKey, Structure, Variant, structure};
use crate::ty::{Calls, Concrete, Effect, EffectTerm, FlowTerm, Flows, Holds, LenTerm, Mutability, Ty, TyKind};
use crate::typeck::Coercion;

/// `value`, a transparent structured value, or where `through` is `Some` a
/// reference of that mutability to one, coerced into a value of `to`, each
/// of `parts` by its own coercion. `to` states the shape the value takes:
/// the fields it is narrowed to, the variants it is widened to.
pub(super) fn structured(
    cx: &mut Lcx<'_, '_>,
    through: Option<Mutability>,
    to: Ty,
    parts: &[(PartKey, Option<Coercion>)],
    value: ValueId,
    span: Span,
) -> Lowered<ValueId> {
    let (from, target) = match through {
        None => (cx.value_ty(value), to),
        Some(_) => (referent(cx, cx.value_ty(value)), referent(cx, to)),
    };
    let made = match through {
        None => to,
        Some(mutability) => gathered(cx, mutability, target),
    };
    let assembled = assemble(cx, Assembly { through, from, target, made, parts, value, span })?;
    Ok(match through {
        None => assembled,
        Some(_) => cx.made(Op::Intrinsic { intrinsic: Intrinsic::Gather, args: vec![assembled] }, to, span),
    })
}

/// `as_ref` or `as_mut` of `value`, a reference of `mutability` to a
/// transparent structured value (operators §2): the structure of `made`,
/// each part a reference into the original storage, one level. Nothing is
/// taken out or copied.
pub(super) fn lend(cx: &mut Lcx<'_, '_>, mutability: Mutability, value: ValueId, made: Ty, span: Span) -> Lowered<ValueId> {
    let target = referent(cx, cx.value_ty(value));
    assemble(cx, Assembly { through: Some(mutability), from: target, target, made, parts: &[], value, span })
}

/// A structure assembled from `value`'s parts: `value` itself, or, where
/// `through` is `Some`, a reference of that mutability to it, of type
/// `from`, into the shape of `target`, the result of type `made`, each
/// part by its own coercion among `parts`.
struct Assembly<'p> {
    through: Option<Mutability>,
    from: Ty,
    target: Ty,
    made: Ty,
    parts: &'p [(PartKey, Option<Coercion>)],
    value: ValueId,
    span: Span,
}

fn assemble(cx: &mut Lcx<'_, '_>, assembly: Assembly<'_>) -> Lowered<ValueId> {
    let Assembly { through, from, target, made, parts, value, span } = assembly;
    let coerced = Coerced { parts, span };
    let access = Access { value, through, span };
    let types = cx.types();
    match (structure(types, from), structure(types, target)) {
        (Some(of_from @ Structure::Tuple(..)), Some(Structure::Tuple(..))) => {
            let fields = of_from.fields().expect("a tuple is a product");
            let elements = fields
                .iter()
                .map(|field| {
                    let seg = step(field.key).expect("a field has one step");
                    let part = access.part(cx, seg, field.ty)?;
                    coerced.part(cx, field.key, part)
                })
                .collect::<Lowered<_>>()?;
            Ok(cx.made(Op::MakeTuple { elements }, made, span))
        }
        (Some(Structure::Object(from_fields, _)), Some(Structure::Object(slot_fields, _))) => {
            let (from_fields, slot_fields): (Vec<(Astr, Ty)>, Vec<(Astr, Ty)>) = (from_fields.to_vec(), slot_fields.to_vec());
            let fields = slot_fields
                .into_iter()
                .map(|(name, _)| {
                    let (_, field_ty) = *from_fields
                        .iter()
                        .find(|(field, _)| *field == name)
                        .unwrap_or_else(|| panic!("typeck narrows an object only where it holds every field its slot names"));
                    let field = access.part(cx, PathSeg::Field(name), field_ty)?;
                    Ok((name, coerced.part(cx, PartKey::Field(name), field)?))
                })
                .collect::<Lowered<_>>()?;
            Ok(cx.made(Op::MakeObject { fields }, made, span))
        }
        (Some(of_from @ Structure::Option(..)), Some(Structure::Option(..)))
        | (Some(of_from @ Structure::Result { .. }), Some(Structure::Result { .. }))
        | (Some(of_from @ Structure::Enum(..)), Some(Structure::Enum(..))) => {
            let variants = of_from.variants().expect("an enum, an `Option` and a `Result` are variant types");
            same_variant(cx, &coerced, &access, &variants, made)
        }
        (Some(Structure::Array { element, len, .. }), Some(Structure::Array { .. })) => {
            let len = known_len(&len);
            let param = match through {
                None => element,
                Some(mutability) => part_reference(cx, mutability, element),
            };
            let mapped = match cx.kind(made) {
                TyKind::Array(mapped, ..) => *mapped,
                other => panic!("an array is coerced to an array, not {other:?}"),
            };
            let effect = coerced.effect(cx, PartKey::Element);
            let map = ArrayMap { arrays: vec![value], params: vec![param], mapped, len, effect, span };
            array_map(cx, map, &|cx, element| coerced.part(cx, PartKey::Element, element[0]))
        }
        _ => panic!("typeck coerces a structured value only to a slot of its own kind, not {from:?} to {target:?}"),
    }
}

/// The target type of the reference type `ty`.
fn referent(cx: &Lcx<'_, '_>, ty: Ty) -> Ty {
    match cx.kind(ty) {
        TyKind::Ref(_, target) => *target,
        other => panic!("a coercion through a reference coerces a reference, not {other:?}"),
    }
}

/// The type of a reference of `through` to a part of type `part`: a part
/// that is a reference is reborrowed at the weaker of the two
/// mutabilities, since a reference's target is never a reference (types
/// §4), as typeck pairs the parts.
pub(super) fn part_reference(cx: &mut Lcx<'_, '_>, through: Mutability, part: Ty) -> Ty {
    match cx.kind(part).clone() {
        TyKind::Ref(Mutability::Mut, target) if through == Mutability::Mut => cx.intern(TyKind::Ref(Mutability::Mut, target)),
        TyKind::Ref(_, target) => cx.intern(TyKind::Ref(Mutability::Shared, target)),
        _ => cx.ref_ty(through, part),
    }
}

/// The structure of references of `through` to the parts of `target`,
/// which `Intrinsic::Gather` makes the reference to a `target` (ownership
/// §5).
fn gathered(cx: &mut Lcx<'_, '_>, through: Mutability, target: Ty) -> Ty {
    let part = |cx: &mut Lcx<'_, '_>, ty: Ty| part_reference(cx, through, ty);
    let kind = match cx.kind(target).clone() {
        TyKind::Tuple(items, ()) => TyKind::Tuple(items.iter().map(|item| part(cx, *item)).collect(), ()),
        TyKind::Object { fields, label: () } => {
            TyKind::Object { fields: fields.iter().map(|(name, field)| (*name, part(cx, *field))).collect(), label: () }
        }
        TyKind::Enum { name, variants, label: () } => TyKind::Enum {
            name,
            variants: variants.iter().map(|(tag, payload)| (*tag, payload.map(|payload| part(cx, payload)))).collect(),
            label: (),
        },
        TyKind::Array(element, len, ()) => TyKind::Array(part(cx, element), LenTerm::Known(known_len(&len)), ()),
        TyKind::Option(payload, ()) => TyKind::Option(part(cx, payload), ()),
        TyKind::Result(ok, err, ()) => {
            let ok = part(cx, ok);
            TyKind::Result(ok, part(cx, err), ())
        }
        other => panic!("a reference is gathered only of a transparent structured type's parts, not {other:?}"),
    };
    cx.intern(kind)
}

/// How a part of the coerced value is reached: moved out of the value, or,
/// through a reference, as a reference into the original storage.
struct Access {
    value: ValueId,
    through: Option<Mutability>,
    span: Span,
}

impl Access {
    /// The part at `seg`, of type `ty` in the value's type, taken out of the
    /// value (`place::extract`) or lent through the reference to it
    /// (`Lcx::emit`).
    fn part(&self, cx: &mut Lcx<'_, '_>, seg: PathSeg, ty: Ty) -> Lowered<ValueId> {
        let span = self.span;
        let Some(through) = self.through else {
            return extract(cx, self.value, seg, span);
        };
        let target = RefTarget::Through(self.value);
        let path = vec![seg];
        let lent = part_reference(cx, through, ty);
        match cx.kind(ty).clone() {
            // A reference part is taken as it is where the mutabilities
            // agree, and read shared through the one taken otherwise.
            TyKind::Ref(held, _) => {
                let taken = cx.emit(Op::Take { target, path, taken_out: false }, ty, span)?;
                Ok(match held == through || held == Mutability::Shared {
                    true => taken,
                    false => {
                        let op = Op::Ref { target: RefTarget::Through(taken), path: Vec::new(), mutability: Mutability::Shared };
                        cx.made(op, lent, span)
                    }
                })
            }
            _ => cx.emit(Op::Ref { target, path, mutability: through }, lent, span),
        }
    }
}

/// An `Intrinsic::ArrayMap` to be made: `arrays`, `k` arrays of `len`
/// elements or references to them, mapped by a function of `k` parameters
/// of the types `params` to `mapped`, whose effect is `effect`. For `k` =
/// 0, `len` is what makes the result's length.
pub(super) struct ArrayMap {
    pub arrays: Vec<ValueId>,
    pub params: Vec<Ty>,
    pub mapped: Ty,
    pub len: usize,
    pub effect: Effect,
    pub span: Span,
}

/// The array `map` makes, its function a body of its own (`Lcx::make_body`)
/// that `element` lowers from its parameters; it stands on the order chain
/// where that function's effect is not pure, as a call of it does
/// (`Lcx::emit`).
pub(super) fn array_map(
    cx: &mut Lcx<'_, '_>,
    map: ArrayMap,
    element: &dyn Fn(&mut Lcx<'_, '_>, &[ValueId]) -> Lowered<ValueId>,
) -> Lowered<ValueId> {
    let ArrayMap { arrays, params, mapped, len, effect, span } = map;
    let fn_ty = cx.intern(TyKind::Fn {
        params: params.clone().into(),
        ret: mapped,
        effect: EffectTerm::Known(effect.clone()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let result_ty = cx.intern(TyKind::Array(mapped, LenTerm::Known(len), ()));
    let name = cx.decls.core().receiver_param();
    let entry = Entry { captures: Vec::new(), params: params.iter().map(|ty| Declared { name, ty: *ty, span }).collect(), contexts: &[] };
    let body = cx.nested(cx.decided, super::ctxt::ordered(&effect), mapped).make_body(entry, |inner, _, at| element(inner, at), span);
    let f = cx.made(Op::MakeClosure { bodies: vec![body], captures: Vec::new() }, fn_ty, span);
    let mut args = arrays;
    args.push(f);
    cx.emit(Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args }, result_ty, span)
}

/// Each kept part's coercion, by its key.
struct Coerced<'p> {
    parts: &'p [(PartKey, Option<Coercion>)],
    span: Span,
}

impl Coerced<'_> {
    /// `part`, the value's part at `key`, coerced by its own coercion where
    /// it has one.
    fn part(&self, cx: &mut Lcx<'_, '_>, key: PartKey, part: ValueId) -> Lowered<ValueId> {
        match self.parts.iter().find(|(at, _)| *at == key) {
            Some((_, Some(coercion))) => coerce(cx, coercion, part, self.span),
            Some((_, None)) | None => Ok(part),
        }
    }

    /// The effect of coercing the part at `key`: its steps' functions'
    /// joined.
    fn effect(&self, cx: &Lcx<'_, '_>, key: PartKey) -> Effect {
        match self.parts.iter().find(|(at, _)| *at == key) {
            Some((_, Some(coercion))) => coercion_effect(cx, coercion),
            Some((_, None)) | None => Effect::pure(),
        }
    }
}

/// The effect of `coercion`: every step's function's, joined.
fn coercion_effect(cx: &Lcx<'_, '_>, coercion: &Coercion) -> Effect {
    match coercion {
        Coercion::Steps(steps) => steps.iter().fold(Effect::pure(), |joined, step| {
            let effect = fn_effect(cx.types(), step.via_ty).expect("a step's function type has an effect");
            joined.join(effect)
        }),
        Coercion::Structured { parts, .. } => parts
            .iter()
            .filter_map(|(_, part)| part.as_ref())
            .fold(Effect::pure(), |joined, part| joined.join(&coercion_effect(cx, part))),
    }
}

/// A variant value made the same variant of `made`, its payload coerced.
fn same_variant(cx: &mut Lcx<'_, '_>, coerced: &Coerced<'_>, access: &Access, variants: &[Variant<Concrete>], made: Ty) -> Lowered<ValueId> {
    let span = coerced.span;
    by_variant(cx, variants, made, access.value, span, &|cx, variant| {
        let payload = variant
            .payload
            .map(|ty| {
                let key = PartKey::Payload(variant.key);
                let part = access.part(cx, step(key).expect("a payload has one step"), ty)?;
                coerced.part(cx, key, part)
            })
            .transpose()?;
        Ok(cx.made(Op::MakeVariant { variant: variant.key, payload }, made, span))
    })
}

/// `arm` of the variant `value` is, a value of `ty`: each of `variants` of
/// its type but the last is tested in turn, and the last is what the value
/// is where none of them matched.
///
/// # Panics
/// `variants` is empty: no value has a type of no variant, so no run gets
/// past making or lending one (`Lcx::emit`), and none dispatches on it.
pub(super) fn by_variant(
    cx: &mut Lcx<'_, '_>,
    variants: &[Variant<Concrete>],
    ty: Ty,
    value: ValueId,
    span: Span,
    arm: &dyn Fn(&mut Lcx<'_, '_>, &Variant<Concrete>) -> Lowered<ValueId>,
) -> Lowered<ValueId> {
    let Some((variant, rest)) = variants.split_first() else {
        unreachable!("no value has a type of no variant, so no run dispatches on one")
    };
    if rest.is_empty() {
        return arm(cx, variant);
    }
    let bool_ty = cx.bool_ty();
    let is = cx.made(Op::TestVariant { src: value, variant: variant.key }, bool_ty, span);
    diamond(cx, is, ty, span, |cx| arm(cx, variant), |cx| by_variant(cx, rest, ty, value, span, arm))
}
