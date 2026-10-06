//! Control forms, each one region with one entry, and one exit where a run
//! gets past it (control-flow).
//! `if`/`if let` and short-circuit `&&`/`||` are a `Diamond` the Builder
//! makes with its join known, a value being the join's parameter; `match`
//! is `pat`'s; `for` is a `For` by typeck's `Iteration` (`&`/`&mut` through
//! the recorded `as_slice(_mut)` or the language's view of an array, arrays
//! by value, ranges); `while`/`while let` a `While`; `break`/`continue` go
//! to the innermost loop's exit/header; `return` is the body's `Return`;
//! `?` tests the tag and returns the `Err`/`None` early.

use acvus_ast::{AstId, BinOp, Binder, ElseBranch, Expr, ForHead, Pattern, Slot, Span, Stmt};

use super::call::convert;
use super::ctxt::{Home, Lcx, LoopTargets, Lowered, Unreached};
use super::expr::lower_expr;
use super::pat::{Matched, bind, test};
use super::place::{Place, assign, view_array};
use super::stmt::{lower_body, lower_stmts};
use crate::decl::unwrapped;
use crate::ir::cfg::{Op, Terminator};
use crate::ir::{ForSource, RefTarget, ValueId};
use crate::ty::structural::{PartKey, VariantKey, structure};
use crate::ty::{Mutability, Ty, TyKind};
use crate::typeck::{Iteration, TryExit};

/// A two-way region: a `Diamond` on `cond` into two arms that meet at a
/// join taking one value of `ty`. Each arm lowers itself in its own block
/// and yields its value; an arm a run gets past enters the join with it.
/// Where neither is, the region is `Unreached`, and its join, a block of the
/// region all the same, ends in `Diverge`.
pub(super) fn diamond(
    cx: &mut Lcx<'_, '_>,
    cond: ValueId,
    ty: Ty,
    span: Span,
    then_arm: impl FnOnce(&mut Lcx<'_, '_>) -> Lowered<ValueId>,
    else_arm: impl FnOnce(&mut Lcx<'_, '_>) -> Lowered<ValueId>,
) -> Lowered<ValueId> {
    let then_to = cx.new_block(&[]);
    let else_to = cx.new_block(&[]);
    let join = cx.new_block(&[ty]);
    let terminator =
        Terminator::Diamond { cond, then_to, then_args: Vec::new(), else_to, else_args: Vec::new(), join };
    cx.terminate(terminator, span);
    let mut joined = false;
    if let Ok(value) = cx.enter_block(then_to, span).and_then(|_| then_arm(cx)) {
        cx.jump(join, vec![value], span);
        joined = true;
    }
    if let Ok(value) = cx.enter_block(else_to, span).and_then(|_| else_arm(cx)) {
        cx.jump(join, vec![value], span);
        joined = true;
    }
    let params = cx.enter_block(join, span)?;
    match joined {
        true => Ok(params[0]),
        false => Err(cx.end_block(Terminator::Diverge, span)),
    }
}

pub(super) fn lower_if<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    cond: &Expr<S>,
    then: &[Stmt<S>],
    then_tail: Option<&Expr<S>>,
    otherwise: Option<&ElseBranch<S>>,
    span: Span,
) -> Lowered<ValueId> {
    let cond = lower_expr(cx, cond)?;
    let ty = cx.ty(site);
    diamond(cx, cond, ty, span, |cx| branch(cx, site, then, then_tail, span), |cx| else_branch(cx, site, otherwise, span))
}

/// `if let pattern = source`: the pattern's test is the condition, and its
/// names are bound where it holds.
pub(super) fn lower_if_let<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    pattern: &Pattern<S>,
    source: &Expr<S>,
    then: &[Stmt<S>],
    then_tail: Option<&Expr<S>>,
    otherwise: Option<&ElseBranch<S>>,
    span: Span,
) -> Lowered<ValueId> {
    let scrutinee = lower_expr(cx, source)?;
    let matched = Matched::scrutinee(cx, scrutinee, source.span());
    let cond = match test(cx, pattern, &matched)? {
        Some(cond) => cond,
        None => cx.bool_const(true, span),
    };
    let ty = cx.ty(site);
    diamond(
        cx,
        cond,
        ty,
        span,
        |cx| {
            bind(cx, pattern, &matched)?;
            branch(cx, site, then, then_tail, span)
        },
        |cx| else_branch(cx, site, otherwise, span),
    )
}

/// A branch's block; its value enters the construct's one slot at its
/// tail's site, or at the construct's where it has none.
fn branch<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, stmts: &[Stmt<S>], tail: Option<&Expr<S>>, span: Span) -> Lowered<ValueId> {
    let value = lower_body(cx, stmts, tail, span)?;
    let at = tail.map_or(site, Expr::id);
    convert(cx, at, value, span)
}

/// The `else`: without one, the missing branch is `()` (control-flow).
fn else_branch<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, otherwise: Option<&ElseBranch<S>>, span: Span) -> Lowered<ValueId> {
    match otherwise {
        None => {
            let unit = cx.unit(span);
            convert(cx, site, unit, span)
        }
        Some(ElseBranch::ElseIf(expr)) => {
            let value = lower_expr(cx, expr)?;
            convert(cx, expr.id(), value, expr.span())
        }
        Some(ElseBranch::Else { body, tail, span }) => branch(cx, site, body, tail.as_deref(), *span),
    }
}

/// `a && b`, `a || b`: `b` runs only where `a` does not decide.
pub(super) fn lower_short_circuit<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    op: BinOp,
    left: &Expr<S>,
    right: &Expr<S>,
    span: Span,
) -> Lowered<ValueId> {
    let decided = lower_expr(cx, left)?;
    let ty = cx.ty(site);
    let rest = |cx: &mut Lcx<'_, '_>| lower_expr(cx, right);
    match op {
        BinOp::And => diamond(cx, decided, ty, span, rest, |cx| Ok(cx.bool_const(false, span))),
        BinOp::Or => diamond(cx, decided, ty, span, |cx| Ok(cx.bool_const(true, span)), rest),
        other => panic!("{other:?} does not short-circuit"),
    }
}

/// A loop's body between its header and its exit: `break` and `continue`
/// inside go to them, and where a run gets past the body, it goes back to
/// the header.
fn loop_body<S: Slot>(cx: &mut Lcx<'_, '_>, targets: LoopTargets, body: &[Stmt<S>], span: Span) {
    cx.in_loop(targets, |cx| {
        if lower_stmts(cx, body).is_ok() {
            cx.jump(targets.header, Vec::new(), span);
        }
    });
}

/// `while cond { body }`: its exit is reached where its condition is. Its
/// condition leaves no loop: typeck refuses a `break` or a `continue` there
/// (`InLoopCondition`).
pub(super) fn lower_while<S: Slot>(cx: &mut Lcx<'_, '_>, cond: &Expr<S>, body: &[Stmt<S>], span: Span) -> Lowered<()> {
    let header = cx.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.enter_block(header, span)?;
    let cond = lower_expr(cx, cond)?;
    let body_block = cx.new_block(&[]);
    let exit = cx.new_block(&[]);
    let terminator = Terminator::While { cond, body: body_block, body_args: Vec::new(), exit, exit_args: Vec::new() };
    cx.terminate(terminator, span);
    if cx.enter_block(body_block, span).is_ok() {
        loop_body(cx, LoopTargets { exit, header }, body, span);
    }
    cx.enter_block(exit, span)?;
    Ok(())
}

/// `while let pattern = source`: the pattern's test is the loop's
/// condition, and its names are bound in the body. Its source leaves no
/// loop, as a `while`'s condition does not.
pub(super) fn lower_while_let<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    pattern: &Pattern<S>,
    source: &Expr<S>,
    body: &[Stmt<S>],
    span: Span,
) -> Lowered<()> {
    let header = cx.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.enter_block(header, span)?;
    let scrutinee = lower_expr(cx, source)?;
    let matched = Matched::scrutinee(cx, scrutinee, source.span());
    let cond = match test(cx, pattern, &matched)? {
        Some(cond) => cond,
        None => cx.bool_const(true, span),
    };
    let body_block = cx.new_block(&[]);
    let exit = cx.new_block(&[]);
    let terminator = Terminator::While { cond, body: body_block, body_args: Vec::new(), exit, exit_args: Vec::new() };
    cx.terminate(terminator, span);
    if cx.enter_block(body_block, span).and_then(|_| bind(cx, pattern, &matched)).is_ok() {
        loop_body(cx, LoopTargets { exit, header }, body, span);
    }
    cx.enter_block(exit, span)?;
    Ok(())
}

/// What a `for` traverses, and the parameters its body block takes.
struct Traversal {
    source: ForSource,
    params: Vec<Ty>,
}

pub(super) fn lower_for<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    callee_site: AstId,
    binder: &Binder,
    head: &ForHead<S>,
    body: &[Stmt<S>],
    span: Span,
) -> Lowered<()> {
    let element = cx.ty(binder.id);
    let traversal = traversal(cx, site, callee_site, head, element, span)?;
    let header = cx.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.enter_block(header, span)?;
    let body_block = cx.new_block(&traversal.params);
    let exit = cx.new_block(&[]);
    let terminator = Terminator::For {
        source: traversal.source,
        body: body_block,
        body_args: Vec::new(),
        exit,
        exit_args: Vec::new(),
    };
    cx.terminate(terminator, span);
    if let Ok(params) = cx.enter_block(body_block, binder.span) {
        let slot = cx.slot(element, binder.span);
        assign(cx, Place { target: RefTarget::Var(slot), path: Vec::new(), ty: element }, params[0], binder.span);
        cx.bind(binder.id, Home::Slot(slot), binder.span);
        loop_body(cx, LoopTargets { exit, header }, body, span);
    }
    cx.enter_block(exit, span)?;
    Ok(())
}

/// The source of a `for` by what typeck decided it iterates. typeck
/// decides none for a head of type `!`, which no run gets past.
fn traversal<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    callee_site: AstId,
    head: &ForHead<S>,
    element: Ty,
    span: Span,
) -> Lowered<Traversal> {
    let counter = cx.u64_ty();
    let iteration = cx.decided.iteration(site);
    match (head, iteration) {
        (ForHead::Range { lo, hi }, _) => {
            let at = lower_expr(cx, lo)?;
            let hi = lower_expr(cx, hi)?;
            Ok(Traversal { source: ForSource::Range { at, hi }, params: vec![element] })
        }
        (ForHead::Value(expr), Some(Iteration::ByValue)) => {
            let array = lower_expr(cx, expr)?;
            Ok(Traversal { source: ForSource::Array(array), params: vec![element, counter] })
        }
        (ForHead::Value(expr), Some(Iteration::ByRef(mutability))) => {
            let reference = lower_expr(cx, expr)?;
            let TyKind::Ref(_, lent) = cx.kind(element) else {
                panic!("a `for` by reference binds a reference to each element")
            };
            let item = *lent;
            let slice = match cx.kind(cx.value_ty(reference)) {
                TyKind::Ref(_, target) => match cx.kind(*target) {
                    TyKind::Array(..) => view_array(cx, reference, mutability, item, span),
                    TyKind::Slice(_) => reference,
                    _ => {
                        let chosen = cx.reached(callee_site);
                        let slice = cx.intern(TyKind::Slice(item));
                        let ty = cx.ref_ty(mutability, slice);
                        let op = Op::AsSlice { container: reference, mutability, r#impl: cx.reached_function(chosen) };
                        cx.made(op, ty, span)
                    }
                },
                other => panic!("a `for` by reference has a reference head, not {other:?}"),
            };
            let source = match mutability {
                Mutability::Shared => ForSource::Slice(slice),
                Mutability::Mut => ForSource::SliceMut(slice),
            };
            Ok(Traversal { source, params: vec![element, counter] })
        }
        (ForHead::Value(expr), Some(Iteration::Range)) => {
            panic!("typeck iterates by range only a range head, not {:?}", expr.id())
        }
        (ForHead::Value(expr), None) => {
            let Err(unreached) = lower_expr(cx, expr) else {
                panic!("typeck decides how a `for` iterates every head a run gets past, and not {:?}", expr.id())
            };
            Err(unreached)
        }
    }
}

/// `break` or `continue`: to the innermost loop's exit or header.
pub(super) fn lower_loop_exit(cx: &mut Lcx<'_, '_>, to_exit: bool, span: Span) -> Unreached {
    let targets = *cx.loops.last().expect("typeck refuses `break` and `continue` outside a loop (OutsideLoop)");
    let to = match to_exit {
        true => targets.exit,
        false => targets.header,
    };
    cx.end_block(Terminator::Jump { to, args: Vec::new() }, span)
}

/// `return value`: the value enters the body's result at its own site.
pub(super) fn lower_return<S: Slot>(cx: &mut Lcx<'_, '_>, value: &Expr<S>, span: Span) -> Lowered<ValueId> {
    let returned = lower_expr(cx, value)?;
    let returned = convert(cx, value.id(), returned, value.span())?;
    Err(cx.return_value(returned, span))
}

/// `inner?`: the payload of the variant it goes on with
/// (`decl::unwrapped`) goes on; the other variant leaves the body, coerced
/// into its result at the `?`'s site.
pub(super) fn lower_try<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, inner: &Expr<S>, span: Span) -> Lowered<ValueId> {
    let tried = lower_expr(cx, inner)?;
    let value_ty = cx.ty(site);
    let exit = cx
        .decided
        .try_exit(site)
        .unwrap_or_else(|| panic!("typeck decides the exit of every `?` whose operand a run gets past, and not {site:?}"));
    let tried_ty = cx.value_ty(tried);
    let goes_on = structure(cx.types(), tried_ty)
        .and_then(|structured| unwrapped(&structured))
        .unwrap_or_else(|| panic!("typeck admits a `?` only of an `Option` or a `Result`, not {:?}", cx.kind(tried_ty)));
    let exit_ty = match exit {
        TryExit::Option(ty) | TryExit::Result(ty) => ty,
    };
    let bool_ty = cx.bool_ty();
    let holds = cx.made(Op::TestVariant { src: tried, variant: goes_on }, bool_ty, span);
    let on = cx.new_block(&[]);
    let leave = cx.new_block(&[]);
    let terminator =
        Terminator::JumpIf { cond: holds, then_to: on, then_args: Vec::new(), else_to: leave, else_args: Vec::new() };
    cx.terminate(terminator, span);
    let left = cx.enter_block(leave, span).and_then(|_| match exit {
        TryExit::Option(_) => Ok(cx.made(Op::MakeVariant { variant: VariantKey::None, payload: None }, exit_ty, span)),
        TryExit::Result(_) => {
            let error_ty = structure(cx.types(), tried_ty)
                .and_then(|structured| structured.part(PartKey::Payload(VariantKey::Err)))
                .unwrap_or_else(|| panic!("a `?` that leaves by `Err` tries a `Result`, not {:?}", cx.kind(tried_ty)));
            let error = cx.emit(Op::UnwrapVariant { src: tried, variant: VariantKey::Err }, error_ty, span)?;
            Ok(cx.made(Op::MakeVariant { variant: VariantKey::Err, payload: Some(error) }, exit_ty, span))
        }
    });
    if let Ok(left) = left.and_then(|left| convert(cx, site, left, span)) {
        cx.return_value(left, span);
    }
    cx.enter_block(on, span)?;
    cx.emit(Op::UnwrapVariant { src: tried, variant: goes_on }, value_ty, span)
}
