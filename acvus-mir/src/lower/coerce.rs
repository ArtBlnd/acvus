//! A transparent structured value coerced to its slot by the tree typeck
//! recorded (types §3): a new value of the slot's type, built part by part.
//! An object keeps the fields its slot names and ends the others where it
//! is narrowed (ownership §6); an enum is the same variant of the wider
//! enum; an array is one `For` over its elements, each coerced alike; a
//! tuple, an `Option` and a `Result` rebuild their parts in place. A part
//! is coerced by its own coercion, `call::coerce`.

use acvus_ast::Span;

use super::ctxt::Lcx;
use crate::infer::ShapeCoercion;
use crate::ir::ValueId;
use crate::ty::structural::PartKey;
use crate::typeck::Coercion;

/// `value`, a transparent structured value, coerced by `shape` and each of
/// `parts`.
pub(super) fn structured(
    cx: &mut Lcx<'_, '_>,
    shape: &ShapeCoercion,
    parts: &[(PartKey, Option<Coercion>)],
    value: ValueId,
    span: Span,
) -> ValueId {
    todo!("a new value of the slot's type, each kept part coerced in turn")
}
