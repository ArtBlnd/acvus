//! Places: a binding, a context held in a slot, a field, `a[i]` through the
//! recorded `as_slice(_mut)` or the language's own view of an array
//! (operators §3, extern-fn §1), and `*r`. A place is read by a `Take`,
//! lent by a `Ref`, and written by an `Assign`; `&` of a reference is a
//! reborrow, since a reference's target is never a reference (types §4);
//! `*r` is a take (operators §6). A value that is not a place and is lent
//! is put in a slot of its own first (D67 item 9).

use acvus_ast::{AstId, Expr, Place as AstPlace, PlaceBase, RefKind, Root, Slot, Span};
use acvus_utils::QualifiedRef;

use super::call::convert;
use super::context;
use super::ctxt::{ContextHome, Home, Lcx};
use super::expr::lower_expr;
use crate::ir::cfg::Op;
use crate::ir::{IndexBound, IndexMode, PathSeg, RefTarget, ValueId, is_word};
use crate::ty::{Mutability, Ty, TyKind};
use crate::typeck::Name;

/// A storage and a path under it, holding a `ty`.
#[derive(Debug, Clone)]
pub(super) struct Place {
    pub target: RefTarget,
    pub path: Vec<PathSeg>,
    pub ty: Ty,
}

impl Place {
    pub fn with(&self, seg: PathSeg, ty: Ty) -> Place {
        let mut path = self.path.clone();
        path.push(seg);
        Place { target: self.target, path, ty }
    }
}

/// An expression as a place where it names one, and as a value otherwise.
#[derive(Debug, Clone)]
pub(super) enum Operand {
    Place(Place),
    Value(ValueId),
}

/// The place a binding lives in, holding a `ty`.
/// A slot holds its content at the slot's own type: a `&mut` binding a
/// lambda captured by `&` holds a shared reborrow of it there, though the
/// lambda's uses are typed as the binding (D85).
pub(super) fn binding_place(cx: &Lcx<'_, '_>, binder: AstId, ty: Ty) -> Place {
    match cx.home(binder) {
        Home::Slot(slot) => Place { target: RefTarget::Var(slot), path: Vec::new(), ty: cx.value_ty(slot) },
        Home::Through(reference) => Place { target: RefTarget::Through(reference), path: Vec::new(), ty },
    }
}

/// `expr` as a place where it names one; `mutability` is what an index in
/// it lends its container for.
pub(super) fn operand<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>, mutability: Mutability) -> Operand {
    match expr {
        Expr::Ident { id, ref_kind: RefKind::Value, .. } => match cx.typeck.names.get(id) {
            Some(Name::Binding(binder)) => Operand::Place(binding_place(cx, *binder, cx.ty(*id))),
            _ => Operand::Value(lower_expr(cx, expr)),
        },
        Expr::Ident { id, name, ref_kind: RefKind::ExternParam, .. } => {
            Operand::Place(input_place(cx, name.name, cx.ty(*id)))
        }
        Expr::ContextRef { id, name, span } => context_operand(cx, *name, cx.ty(*id), *span),
        Expr::FieldAccess { id, object, field, span } => {
            let ty = cx.ty(*id);
            let object = operand(cx, object, mutability);
            field_of(cx, object, *field, ty, *span)
        }
        Expr::Index { id, callee_id, object, index, span } => {
            let element = cx.ty(*id);
            let reference = element_ref(cx, object, *callee_id, index, mutability, element, *span);
            Operand::Place(Place { target: RefTarget::Through(reference), path: Vec::new(), ty: element })
        }
        Expr::UnaryOp { id, op: acvus_ast::UnaryOp::Deref, operand: reference, .. } => {
            let reference = lower_expr(cx, reference);
            Operand::Place(Place { target: RefTarget::Through(reference), path: Vec::new(), ty: cx.ty(*id) })
        }
        Expr::Paren { inner, .. } => operand(cx, inner, mutability),
        _ => Operand::Value(lower_expr(cx, expr)),
    }
}

/// The type of the input `name`, as its home holds it.
pub(super) fn input_ty(cx: &Lcx<'_, '_>, name: acvus_utils::Astr) -> Ty {
    let home = *cx
        .inputs
        .get(&name)
        .unwrap_or_else(|| panic!("the input {name:?} the body names is given its home at entry"));
    match home {
        Home::Slot(slot) => cx.value_ty(slot),
        Home::Through(reference) => match cx.kind(cx.value_ty(reference)) {
            TyKind::Ref(_, target) => target.ty(),
            other => panic!("an input captured by reference is reached through a reference, not {other:?}"),
        },
    }
}

/// The place the input `name` lives in, holding a `ty`.
pub(super) fn input_place(cx: &Lcx<'_, '_>, name: acvus_utils::Astr, ty: Ty) -> Place {
    let home = *cx
        .inputs
        .get(&name)
        .unwrap_or_else(|| panic!("the input {name:?} the body names is given its home at entry"));
    let target = match home {
        Home::Slot(slot) => RefTarget::Var(slot),
        Home::Through(reference) => RefTarget::Through(reference),
    };
    Place { target, path: Vec::new(), ty }
}

fn context_operand(cx: &mut Lcx<'_, '_>, context: QualifiedRef, ty: Ty, span: Span) -> Operand {
    match cx.context_home(context) {
        ContextHome::Slot(slot) => Operand::Place(Place { target: RefTarget::Var(slot), path: Vec::new(), ty }),
        ContextHome::Volatile => Operand::Value(context::load(cx, context, ty, span)),
    }
}

/// Field `field` of `object`, through a reference as Rust auto-derefs.
fn field_of(cx: &mut Lcx<'_, '_>, object: Operand, field: acvus_utils::Astr, ty: Ty, span: Span) -> Operand {
    match object {
        Operand::Place(place) if is_ref(cx, place.ty) => {
            let reference = read(cx, Operand::Place(place), span);
            Operand::Place(Place { target: RefTarget::Through(reference), path: vec![PathSeg::Field(field)], ty })
        }
        Operand::Place(place) => Operand::Place(place.with(PathSeg::Field(field), ty)),
        Operand::Value(value) if is_ref(cx, cx.value_ty(value)) => {
            Operand::Place(Place { target: RefTarget::Through(value), path: vec![PathSeg::Field(field)], ty })
        }
        Operand::Value(value) => {
            Operand::Value(cx.builder.push(Op::FieldGet { object: value, field, rest: Vec::new() }, ty, span))
        }
    }
}

fn is_ref(cx: &Lcx<'_, '_>, ty: Ty) -> bool {
    matches!(cx.kind(ty), TyKind::Ref(..))
}

/// The value of `operand`: a place's is taken.
pub(super) fn read(cx: &mut Lcx<'_, '_>, operand: Operand, span: Span) -> ValueId {
    match operand {
        Operand::Place(place) => {
            let op = Op::Take { target: place.target, path: place.path, taken_out: false };
            cx.builder.push(op, place.ty, span)
        }
        Operand::Value(value) => value,
    }
}

/// A `mutability` reference to `operand`: a place is lent where it stands,
/// a reference is reborrowed, and any other value is put in a slot of its
/// own and lent there.
pub(super) fn lend(cx: &mut Lcx<'_, '_>, operand: Operand, mutability: Mutability, span: Span) -> ValueId {
    let held = match &operand {
        Operand::Place(place) => place.ty,
        Operand::Value(value) => cx.value_ty(*value),
    };
    if let TyKind::Ref(made, target) = cx.kind(held) {
        let (made, target) = (*made, target.ty());
        let reference = read(cx, operand, span);
        return reborrow(cx, reference, made, target, mutability, span);
    }
    let place = match operand {
        Operand::Place(place) => place,
        Operand::Value(value) => {
            let slot = cx.builder.slot(held, span);
            cx.builder.push_unit(Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }, span);
            Place { target: RefTarget::Var(slot), path: Vec::new(), ty: held }
        }
    };
    let ty = cx.ref_ty(mutability, place.ty);
    cx.builder.push(Op::Ref { target: place.target, path: place.path, mutability }, ty, span)
}

/// `reference`, a `made` reference to `target`, as a `wanted` one: itself
/// where the two agree, and a shared reborrow of a `&mut` otherwise.
///
/// # Panics
/// A `&mut` is wanted of a `&`: typeck refuses that lend.
pub(super) fn reborrow(
    cx: &mut Lcx<'_, '_>,
    reference: ValueId,
    made: Mutability,
    target: Ty,
    wanted: Mutability,
    span: Span,
) -> ValueId {
    match (made, wanted) {
        (Mutability::Shared, Mutability::Mut) => {
            panic!("a `&mut` of a `&` is refused by typeck, so an admitted program lends none")
        }
        (made, wanted) if made == wanted => reference,
        _ => {
            let ty = cx.ref_ty(Mutability::Shared, target);
            let op = Op::Ref { target: RefTarget::Through(reference), path: Vec::new(), mutability: Mutability::Shared };
            cx.builder.push(op, ty, span)
        }
    }
}

/// `&place` or `&mut place`.
pub(super) fn lower_borrow<S: Slot>(cx: &mut Lcx<'_, '_>, mutable: bool, place: &Expr<S>, span: Span) -> ValueId {
    let mutability = match mutable {
        true => Mutability::Mut,
        false => Mutability::Shared,
    };
    // A generalized binding is lent as the value its name reads, kept at
    // the use's signature; its storage, holding every signature, is never
    // lent (functions §2).
    if let Expr::Ident { id, .. } = place
        && let Some(crate::typeck::Name::Binding(binder)) = cx.typeck.names.get(id).copied()
        && super::closure::is_generalized(cx, binder)
    {
        let value = super::expr::lower_expr(cx, place);
        return lend(cx, Operand::Value(value), mutability, span);
    }
    let operand = operand(cx, place, mutability);
    lend(cx, operand, mutability, span)
}

/// `a[i]` read: a word element is copied out, any other is taken through a
/// reference to it.
pub(super) fn lower_index<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    callee_site: AstId,
    object: &Expr<S>,
    index: &Expr<S>,
    span: Span,
) -> ValueId {
    let element = cx.ty(site);
    if !is_word(cx.builder.types(), element) {
        let reference = element_ref(cx, object, callee_site, index, Mutability::Shared, element, span);
        let place = Place { target: RefTarget::Through(reference), path: Vec::new(), ty: element };
        return read(cx, Operand::Place(place), span);
    }
    let slice = slice_of(cx, object, callee_site, Mutability::Shared, element, span);
    let at = lower_expr(cx, index);
    let op = Op::Index { slice, index: at, mode: IndexMode::Copy, bound: IndexBound::Checked };
    cx.builder.push(op, element, span)
}

/// A reference to element `index` of `object`, lent for `mutability` where
/// the language's own view decides it.
fn element_ref<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    object: &Expr<S>,
    callee_site: AstId,
    index: &Expr<S>,
    mutability: Mutability,
    element: Ty,
    span: Span,
) -> ValueId {
    let slice = slice_of(cx, object, callee_site, mutability, element, span);
    let at = lower_expr(cx, index);
    let held = match cx.kind(cx.value_ty(slice)) {
        TyKind::Ref(held, _) => *held,
        other => panic!("a slice is reached through a reference, not {other:?}"),
    };
    let ty = cx.ref_ty(held, element);
    let op = Op::Index { slice, index: at, mode: IndexMode::Ref, bound: IndexBound::Checked };
    cx.builder.push(op, ty, span)
}

/// How a container gives up the slice of its elements (operators §3).
enum Sequence {
    /// An array: lent, then viewed by the language (extern-fn §1).
    Array,
    /// A reference to an array: viewed by the language.
    ArrayReference,
    /// A slice reference, as it is.
    Slice,
    /// Any other container: through the `as_slice(_mut)` reached for it.
    Container,
}

fn sequence(cx: &Lcx<'_, '_>, held: Ty) -> Sequence {
    match cx.kind(held) {
        TyKind::Array(..) => Sequence::Array,
        TyKind::Ref(_, target) => match cx.kind(target.ty()) {
            TyKind::Array(..) => Sequence::ArrayReference,
            TyKind::Slice(_) => Sequence::Slice,
            _ => Sequence::Container,
        },
        _ => Sequence::Container,
    }
}

/// The slice of `container`'s elements, of type `&[element]` or
/// `&mut [element]`: an array's through the language's own view; any other
/// container's through the `as_slice(_mut)` typeck reached at
/// `callee_site`, at the mutability that impl names.
pub(super) fn slice_of<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    container: &Expr<S>,
    callee_site: AstId,
    mutability: Mutability,
    element: Ty,
    span: Span,
) -> ValueId {
    match sequence(cx, cx.ty(container.id())) {
        Sequence::Array => {
            let operand = operand(cx, container, mutability);
            let array = lend(cx, operand, mutability, span);
            view_array(cx, array, mutability, element, span)
        }
        Sequence::ArrayReference => {
            let reference = lower_expr(cx, container);
            let array = as_mutability(cx, reference, mutability, span);
            view_array(cx, array, mutability, element, span)
        }
        Sequence::Slice => {
            let reference = lower_expr(cx, container);
            as_mutability(cx, reference, mutability, span)
        }
        Sequence::Container => {
            let chosen = cx.reached(callee_site);
            let mutability = match chosen.declaration == cx.decls.core().as_slice_mut {
                true => Mutability::Mut,
                false => Mutability::Shared,
            };
            let operand = operand(cx, container, mutability);
            let lent = lend(cx, operand, mutability, span);
            let lent = convert(cx, callee_site, lent, span);
            let slice = cx.builder.intern(TyKind::Slice(element));
            let ty = cx.ref_ty(mutability, slice);
            let op = Op::AsSlice { container: lent, mutability, r#impl: cx.extern_impl(chosen) };
            cx.builder.push(op, ty, span)
        }
    }
}

/// `reference` reborrowed at `mutability`.
fn as_mutability(cx: &mut Lcx<'_, '_>, reference: ValueId, mutability: Mutability, span: Span) -> ValueId {
    let TyKind::Ref(made, target) = cx.kind(cx.value_ty(reference)) else {
        panic!("a reference-typed value is a reference")
    };
    let (made, target) = (*made, target.ty());
    reborrow(cx, reference, made, target, mutability, span)
}

/// The language's own view of the array `array` references as a slice of
/// its `element`s.
pub(super) fn view_array(cx: &mut Lcx<'_, '_>, array: ValueId, mutability: Mutability, element: Ty, span: Span) -> ValueId {
    let slice = cx.builder.intern(TyKind::Slice(element));
    let ty = cx.ref_ty(mutability, slice);
    cx.builder.push(Op::ArrayView { src: array, mutability }, ty, span)
}

/// `*r` read: a take of what `r` names (operators §6).
pub(super) fn lower_deref<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, reference: &Expr<S>, span: Span) -> ValueId {
    let reference = lower_expr(cx, reference);
    let place = Place { target: RefTarget::Through(reference), path: Vec::new(), ty: cx.ty(site) };
    read(cx, Operand::Place(place), span)
}

/// Where a store writes: a place, or a part of a volatile context, which
/// is loaded, written and committed whole (context §3).
enum StoreTarget {
    Place(Place),
    Volatile { context: QualifiedRef, path: Vec<PathSeg>, whole: Ty },
}

/// `place = value`: the value is evaluated first, as Rust does, and enters
/// the place's type at its own site.
pub(super) fn lower_store<S: Slot>(cx: &mut Lcx<'_, '_>, place: &AstPlace<S>, value: &Expr<S>, span: Span) {
    let stored = lower_expr(cx, value);
    let stored = convert(cx, value.id(), stored, value.span());
    match store_target(cx, place) {
        StoreTarget::Place(place) => assign(cx, place, stored, span),
        StoreTarget::Volatile { context, path, .. } if path.is_empty() => context::commit(cx, context, stored, span),
        StoreTarget::Volatile { context, path, whole } => {
            let loaded = context::load(cx, context, whole, span);
            let slot = cx.builder.slot(whole, span);
            let whole_place = Place { target: RefTarget::Var(slot), path: Vec::new(), ty: whole };
            assign(cx, whole_place.clone(), loaded, span);
            let part = Place { target: RefTarget::Var(slot), path, ty: cx.ty(place.id()) };
            assign(cx, part, stored, span);
            let written = read(cx, Operand::Place(whole_place), span);
            context::commit(cx, context, written, span);
        }
    }
}

pub(super) fn assign(cx: &mut Lcx<'_, '_>, place: Place, value: ValueId, span: Span) {
    let op = Op::Assign { target: place.target, path: place.path, value, restores: false };
    cx.builder.push_unit(op, span);
}

fn store_target<S: Slot>(cx: &mut Lcx<'_, '_>, place: &AstPlace<S>) -> StoreTarget {
    let ty = cx.ty(place.id());
    match place {
        AstPlace::Base(PlaceBase::Root { id, root, .. }) => match root {
            Root::Local(_) => match cx.typeck.names.get(id) {
                Some(Name::Binding(binder)) => StoreTarget::Place(binding_place(cx, *binder, ty)),
                other => panic!("a store's root names a binding in an admitted program, not {other:?}"),
            },
            Root::Context(name) => {
                let context = QualifiedRef::root(*name);
                match cx.context_home(context) {
                    ContextHome::Slot(slot) => {
                        StoreTarget::Place(Place { target: RefTarget::Var(slot), path: Vec::new(), ty })
                    }
                    ContextHome::Volatile => StoreTarget::Volatile { context, path: Vec::new(), whole: ty },
                }
            }
            Root::ExternParam(_) => panic!("an input is never assigned (programs §4); typeck refuses the store"),
        },
        AstPlace::Base(PlaceBase::Element { callee_id, container, index, span, .. }) => {
            let reference = element_ref(cx, container.expr(), *callee_id, index, Mutability::Mut, ty, *span);
            StoreTarget::Place(Place { target: RefTarget::Through(reference), path: Vec::new(), ty })
        }
        AstPlace::Field { object, field, span, .. } => match store_target(cx, object) {
            StoreTarget::Place(object) if is_ref(cx, object.ty) => {
                let reference = read(cx, Operand::Place(object), *span);
                StoreTarget::Place(Place { target: RefTarget::Through(reference), path: vec![PathSeg::Field(*field)], ty })
            }
            StoreTarget::Place(object) => StoreTarget::Place(object.with(PathSeg::Field(*field), ty)),
            StoreTarget::Volatile { context, mut path, whole } => {
                path.push(PathSeg::Field(*field));
                StoreTarget::Volatile { context, path, whole }
            }
        },
    }
}

/// `*r = v`: a store through the `&mut` `r` (operators §6).
pub(super) fn lower_deref_store<S: Slot>(cx: &mut Lcx<'_, '_>, target: &Expr<S>, value: &Expr<S>, span: Span) {
    let reference = lower_expr(cx, target);
    let stored = lower_expr(cx, value);
    let stored = convert(cx, value.id(), stored, value.span());
    let TyKind::Ref(_, referent) = cx.kind(cx.value_ty(reference)) else {
        panic!("typeck admits `*r = v` only where `r` is a `&mut`")
    };
    let ty = referent.ty();
    assign(cx, Place { target: RefTarget::Through(reference), path: Vec::new(), ty }, stored, span);
}
