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
use super::ctxt::Lcx;
use crate::ir::ValueId;
use crate::ir::cfg::Op;
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{Ty, TyKind};
use crate::typeck::Coercion;

/// `value`, a transparent structured value, coerced into a value of `to`,
/// each of `parts` by its own coercion. `to` states the shape the value
/// takes: the fields it is narrowed to, the variants it is widened to.
pub(super) fn structured(
    cx: &mut Lcx<'_, '_>,
    to: Ty,
    parts: &[(PartKey, Option<Coercion>)],
    value: ValueId,
    span: Span,
) -> ValueId {
    let coerced = Coerced { parts, span };
    let from = cx.value_ty(value);
    let types = cx.builder.types();
    match (structure(types, from), structure(types, to)) {
        (Some(Structure::Tuple(items, _)), Some(Structure::Tuple(..))) => {
            let items = items.to_vec();
            let elements = items
                .into_iter()
                .enumerate()
                .map(|(at, item)| {
                    let part = cx.builder.push(Op::TupleIndex { tuple: value, index: at }, item, span);
                    coerced.part(cx, PartKey::Index(at), part)
                })
                .collect();
            cx.builder.push(Op::MakeTuple { elements }, to, span)
        }
        (Some(Structure::Object(..)), Some(Structure::Object(slot_fields, _))) => {
            let slot_fields: Vec<(Astr, Ty)> = slot_fields.to_vec();
            let from_fields: Vec<(Astr, Ty)> = match cx.kind(from) {
                TyKind::Object(object) => object.fields().to_vec(),
                other => panic!("an object's structure is an object's, not {other:?}"),
            };
            let fields = slot_fields
                .into_iter()
                .map(|(name, _)| {
                    let (_, field_ty) = *from_fields
                        .iter()
                        .find(|(field, _)| *field == name)
                        .unwrap_or_else(|| panic!("typeck narrows an object only where it holds every field its slot names"));
                    let field = cx.builder.push(Op::FieldGet { object: value, field: name, rest: Vec::new() }, field_ty, span);
                    (name, coerced.part(cx, PartKey::Field(name), field))
                })
                .collect();
            cx.builder.push(Op::MakeObject { fields }, to, span)
        }
        (Some(Structure::Option(payload, _)), Some(Structure::Option(..))) => {
            let variants = Variant::of_option(cx, payload);
            same_variant(cx, &coerced, &variants, to, value)
        }
        (Some(Structure::Result { ok, err, .. }), Some(Structure::Result { .. })) => {
            let variants = Variant::of_result(cx, ok, err);
            same_variant(cx, &coerced, &variants, to, value)
        }
        (Some(Structure::Enum(variants, _)), Some(Structure::Enum(..))) => {
            let variants = Variant::of_enum(variants);
            same_variant(cx, &coerced, &variants, to, value)
        }
        (Some(Structure::Array { .. }), Some(Structure::Array { .. })) => {
            // NOTE: an open question, returned to the coordinator. An array's
            // elements coerced in one `For` must be pushed onto a new array
            // whose type grows by one element each iteration, which no block
            // parameter of the loop can carry; unrolling is what the rule
            // excludes. Kept here as a stop, not as a decision.
            todo!("an array coerced element by element in one `For`: no IR form builds a new array in a loop")
        }
        _ => panic!("typeck coerces a structured value only to a slot of its own kind, not {from:?} to {to:?}"),
    }
}

/// Each kept part's coercion, by its key.
struct Coerced<'p> {
    parts: &'p [(PartKey, Option<Coercion>)],
    span: Span,
}

impl Coerced<'_> {
    /// `part`, the value's part at `key`, coerced by its own coercion where
    /// it has one.
    fn part(&self, cx: &mut Lcx<'_, '_>, key: PartKey, part: ValueId) -> ValueId {
        match self.parts.iter().find(|(at, _)| *at == key) {
            Some((_, Some(coercion))) => coerce(cx, coercion, part, self.span),
            Some((_, None)) | None => part,
        }
    }
}

/// One variant of a variant type: its tag, and its payload's type and key
/// where it has one.
pub(super) struct Variant {
    pub tag: Astr,
    pub payload: Option<(PartKey, Ty)>,
}

impl Variant {
    pub fn of_option(cx: &Lcx<'_, '_>, payload: Ty) -> [Variant; 2] {
        let tags = cx.decls.core().tags;
        [Variant { tag: tags.some, payload: Some((PartKey::Some, payload)) }, Variant { tag: tags.none, payload: None }]
    }

    pub fn of_result(cx: &Lcx<'_, '_>, ok: Ty, err: Ty) -> [Variant; 2] {
        let tags = cx.decls.core().tags;
        [Variant { tag: tags.ok, payload: Some((PartKey::Ok, ok)) }, Variant { tag: tags.err, payload: Some((PartKey::Err, err)) }]
    }

    pub fn of_enum(variants: &[(Astr, Option<Ty>)]) -> Vec<Variant> {
        variants
            .iter()
            .map(|(tag, payload)| Variant { tag: *tag, payload: payload.map(|ty| (PartKey::Variant(*tag), ty)) })
            .collect()
    }
}

/// A variant value made the same variant of `to`, its payload coerced.
fn same_variant(cx: &mut Lcx<'_, '_>, coerced: &Coerced<'_>, variants: &[Variant], to: Ty, value: ValueId) -> ValueId {
    let span = coerced.span;
    by_variant(cx, variants, to, value, span, &|cx, variant| {
        let payload = variant.payload.map(|(key, ty)| {
            let unwrapped = cx.builder.push(Op::UnwrapVariant { src: value }, ty, span);
            coerced.part(cx, key, unwrapped)
        });
        cx.builder.push(Op::MakeVariant { tag: variant.tag, payload }, to, span)
    })
}

/// `arm` of the variant `value` is, a value of `ty`: each of `variants` of
/// its type but the last is tested in turn, and the last is what the value
/// is where none of them matched.
pub(super) fn by_variant(
    cx: &mut Lcx<'_, '_>,
    variants: &[Variant],
    ty: Ty,
    value: ValueId,
    span: Span,
    arm: &dyn Fn(&mut Lcx<'_, '_>, &Variant) -> ValueId,
) -> ValueId {
    let Some((variant, rest)) = variants.split_first() else {
        panic!("a value of an enum of no variant is never made")
    };
    if rest.is_empty() {
        return arm(cx, variant);
    }
    let bool_ty = cx.bool_ty();
    let is = cx.builder.push(Op::TestVariant { src: value, tag: variant.tag }, bool_ty, span);
    diamond(cx, is, ty, span, |cx| arm(cx, variant), |cx| by_variant(cx, rest, ty, value, span, arm))
}
