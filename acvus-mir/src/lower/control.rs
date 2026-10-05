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
use crate::ir::cfg::{Op, Terminator};
use crate::ir::{ForSource, RefTarget, ValueId};
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
    let then_to = cx.builder.new_block(&[]);
    let else_to = cx.builder.new_block(&[]);
    let join = cx.builder.new_block(&[ty]);
    let terminator =
        Terminator::Diamond { cond, then_to, then_args: Vec::new(), else_to, else_args: Vec::new(), join };
    cx.builder.terminate(terminator, span);
    let mut joined = false;
    cx.builder.switch_to(then_to);
    if let Ok(value) = then_arm(cx) {
        cx.jump(join, vec![value], span);
        joined = true;
    }
    cx.builder.switch_to(else_to);
    if let Ok(value) = else_arm(cx) {
        cx.jump(join, vec![value], span);
        joined = true;
    }
    cx.builder.switch_to(join);
    match joined {
        true => Ok(cx.builder.block_params(join)[0]),
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

/// `while cond { body }`: its exit is reached where its condition is.
pub(super) fn lower_while<S: Slot>(cx: &mut Lcx<'_, '_>, cond: &Expr<S>, body: &[Stmt<S>], span: Span) -> Lowered<()> {
    let header = cx.builder.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.builder.switch_to(header);
    let cond = lower_expr(cx, cond)?;
    let body_block = cx.builder.new_block(&[]);
    let exit = cx.builder.new_block(&[]);
    let terminator = Terminator::While { cond, body: body_block, body_args: Vec::new(), exit, exit_args: Vec::new() };
    cx.builder.terminate(terminator, span);
    cx.builder.switch_to(body_block);
    loop_body(cx, LoopTargets { exit, header }, body, span);
    cx.builder.switch_to(exit);
    Ok(())
}

/// `while let pattern = source`: the pattern's test is the loop's
/// condition, and its names are bound in the body.
pub(super) fn lower_while_let<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    pattern: &Pattern<S>,
    source: &Expr<S>,
    body: &[Stmt<S>],
    span: Span,
) -> Lowered<()> {
    let header = cx.builder.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.builder.switch_to(header);
    let scrutinee = lower_expr(cx, source)?;
    let matched = Matched::scrutinee(cx, scrutinee, source.span());
    let cond = match test(cx, pattern, &matched)? {
        Some(cond) => cond,
        None => cx.bool_const(true, span),
    };
    let body_block = cx.builder.new_block(&[]);
    let exit = cx.builder.new_block(&[]);
    let terminator = Terminator::While { cond, body: body_block, body_args: Vec::new(), exit, exit_args: Vec::new() };
    cx.builder.terminate(terminator, span);
    cx.builder.switch_to(body_block);
    if bind(cx, pattern, &matched).is_ok() {
        loop_body(cx, LoopTargets { exit, header }, body, span);
    }
    cx.builder.switch_to(exit);
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
    let header = cx.builder.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.builder.switch_to(header);
    let body_block = cx.builder.new_block(&traversal.params);
    let exit = cx.builder.new_block(&[]);
    let terminator = Terminator::For {
        source: traversal.source,
        body: body_block,
        body_args: Vec::new(),
        exit,
        exit_args: Vec::new(),
    };
    cx.builder.terminate(terminator, span);
    cx.builder.switch_to(body_block);
    let item = cx.builder.block_params(body_block)[0];
    let slot = cx.builder.slot(element, binder.span);
    assign(cx, Place { target: RefTarget::Var(slot), path: Vec::new(), ty: element }, item, binder.span);
    cx.bind(binder.id, Home::Slot(slot), binder.span);
    loop_body(cx, LoopTargets { exit, header }, body, span);
    cx.builder.switch_to(exit);
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
                        let slice = cx.builder.intern(TyKind::Slice(item));
                        let ty = cx.ref_ty(mutability, slice);
                        let op = Op::AsSlice { container: reference, mutability, r#impl: cx.reached_function(chosen) };
                        cx.builder.push(op, ty, span)
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

/// The body's return of `value`.
pub(super) fn return_value(cx: &mut Lcx<'_, '_>, value: ValueId, span: Span) -> Unreached {
    cx.end_block(Terminator::Return { value, order: None, span }, span)
}

/// `return value`: the value enters the body's result at its own site.
pub(super) fn lower_return<S: Slot>(cx: &mut Lcx<'_, '_>, value: &Expr<S>, span: Span) -> Lowered<ValueId> {
    let returned = lower_expr(cx, value)?;
    let returned = convert(cx, value.id(), returned, value.span())?;
    Err(return_value(cx, returned, span))
}

/// `inner?`: `Some`/`Ok`'s payload goes on; `None`/`Err` leaves the body,
/// coerced into its result at the `?`'s site.
pub(super) fn lower_try<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, inner: &Expr<S>, span: Span) -> Lowered<ValueId> {
    let tried = lower_expr(cx, inner)?;
    let value_ty = cx.ty(site);
    let exit = cx
        .decided
        .try_exit(site)
        .unwrap_or_else(|| panic!("typeck decides the exit of every `?` whose operand a run gets past, and not {site:?}"));
    let tags = cx.decls.core().tags;
    let (goes_on, exit_ty) = match exit {
        TryExit::Option(ty) => (tags.some, ty),
        TryExit::Result(ty) => (tags.ok, ty),
    };
    let bool_ty = cx.bool_ty();
    let holds = cx.builder.push(Op::TestVariant { src: tried, tag: goes_on }, bool_ty, span);
    let on = cx.builder.new_block(&[]);
    let leave = cx.builder.new_block(&[]);
    let terminator =
        Terminator::JumpIf { cond: holds, then_to: on, then_args: Vec::new(), else_to: leave, else_args: Vec::new() };
    cx.builder.terminate(terminator, span);
    cx.builder.switch_to(leave);
    let left = match exit {
        TryExit::Option(_) => cx.builder.push(Op::MakeVariant { tag: tags.none, payload: None }, exit_ty, span),
        TryExit::Result(_) => {
            let error_ty = match cx.kind(cx.value_ty(tried)) {
                TyKind::Result(_, error, _) => *error,
                other => panic!("a `?` that leaves by `Err` tries a `Result`, not {other:?}"),
            };
            let error = cx.builder.push(Op::UnwrapVariant { src: tried }, error_ty, span);
            cx.builder.push(Op::MakeVariant { tag: tags.err, payload: Some(error) }, exit_ty, span)
        }
    };
    if let Ok(left) = convert(cx, site, left, span) {
        return_value(cx, left, span);
    }
    cx.builder.switch_to(on);
    Ok(cx.builder.push(Op::UnwrapVariant { src: tried }, value_ty, span))
}
