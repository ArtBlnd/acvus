use acvus_ast::{
    AstId, BinOp, ElseBranch, Expr, ForHead, Literal, MatchExprArm, ObjectExprField,
    ObjectPatternField, Pattern, RefKind, Script, Span, Stmt, Template, TupleElem,
    TuplePatternElem, UnaryOp,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};

use crate::error::OperatorSignature;
use crate::graph::QualifiedRef;
use crate::ir::{
    Callee, CastKind, ExternCast, ExternInstance, ForKind, ForSource, IndexAccess, IndexMode, Inst,
    InstKind, Label, MirBody, MirModule, OrderEdge, PathSeg, RefTarget, SwitchKey, ValOrigin,
    ValueId, reaches,
};
use crate::place::{Element, PlaceBase, Projected, Storage, projected, projected_store};
use crate::solver::{CaptureRead, MatchMode};
use crate::structural::StructuralSignature;
use crate::ty::{CastTy, Effect, Mutability, Task, Ty, TypeArg};
use crate::typeck::{CallTarget, CapturedName, Passing, StructuralCall, TypeResolution};

/// The name of a template's accumulator. A source name cannot collide with
/// it: the lexer admits no `<` in an identifier.
const TEMPLATE_RESULT: &str = "<template>";

pub struct Lowerer<'a> {
    body: MirBody,
    /// Interner for string interning.
    interner: &'a Interner,
    /// Stack of scopes: variable name -> its binding. A binding introduced
    /// while a name is already bound shadows it: a fresh slot, and the outer
    /// binding is untouched and visible again when the scope ends.
    scopes: Vec<FxHashMap<Astr, Local>>,
    /// Frozen type resolution from typeck.
    resolution: Freeze<TypeResolution>,
    /// The `ret` of the graph `Function` whose body this is (RFC-0054).
    ret: Ty,
    /// Coercion map from type checker (expr AstId -> CastKind).
    coercion_lookup: FxHashMap<AstId, CastKind>,
    /// Closures produced during lowering.
    closures: FxHashMap<Label, MirBody>,
    /// Global closure label counter - shared across nesting levels to prevent
    /// label collisions when nested closures each allocate from a sub-body.
    closure_label_count: u32,
    /// The storage slot holding the current `Order` of the body being lowered
    /// (RFC-0007). `None` for a body whose effect is Pure. The SSA pass
    /// promotes the slot, so branches and loops join orders through phis.
    order_slot: Option<ValueId>,
    /// The innermost `anyorder` block being lowered, if any.
    anyorder: Option<AnyorderScope>,
    /// The slot holding a template's accumulated text, which every
    /// `Stmt::Append` writes through (RFC-0071). `None` in a script.
    result: Option<ValueId>,
    context_slots: BTreeMap<QualifiedRef, ValueId>,
    /// The places the calls being lowered have taken out for their
    /// callees, innermost call last (RFC-0041).
    taken_out: Vec<PlaceRestore>,
    /// The loops enclosing the statement being lowered, innermost last: a
    /// `break` jumps to the last one's exit and a `continue` to its header
    /// (RFC-0057 rule 4).
    loops: Vec<Loop>,
}

/// Where the innermost loop's two jumps go.
struct Loop {
    header: Label,
    exit: Label,
}

/// One capture register of a closure body: the type the caller handed
/// `MakeClosure`, and the reference to it the runtime binds the register to.
struct CaptureRegister {
    given: Ty,
    cap_ty: Ty,
}

#[derive(Clone, Copy)]
struct DiamondLabels {
    then_label: Label,
    else_label: Label,
    join: Label,
}

struct PendingDiamond {
    at: usize,
    labels: DiamondLabels,
}

/// Which of the two jumps out of a loop body this is.
#[derive(Clone, Copy)]
enum Leave {
    Break,
    Continue,
}

/// An `anyorder` block while its body is lowered (RFC-0007): every
/// effectful call inside takes `entry`, and `acc` accumulates a merge of
/// what they yielded. A loop inside stores to `acc` on every iteration, so
/// the SSA pass gives it a loop phi.
#[derive(Clone, Copy)]
struct AnyorderScope {
    entry: ValueId,
    acc: ValueId,
}

/// A place a program names: where it is, and the type of what it holds.
#[derive(Clone)]
struct Place {
    target: RefTarget,
    path: Vec<PathSeg>,
    ty: Ty,
}

/// The arguments of one call, and where the places it takes out begin in
/// `Lowerer::taken_out`.
struct CallArgs {
    values: Vec<ValueId>,
    taken_from: usize,
}

/// A place taken out of its slot for the call and lent as `lent`, a
/// temporary `cast` put in the callee's representation: `back` runs on
/// what `lent` holds afterwards, and the result is assigned to the place.
struct PlaceRestore {
    span: Span,
    place: Place,
    lent: Place,
    mutability: Mutability,
    cast: ExternCast,
    back: ExternCast,
}

impl PlaceRestore {
    /// Whether a shared lend of `place` through `cast` lends this
    /// temporary: the same place, taken out for a shared lend through the
    /// same cast.
    fn shared_by(&self, place: &Place, cast: &ExternCast) -> bool {
        self.mutability == Mutability::Shared
            && self.place.target == place.target
            && self.place.path == place.path
            && self.cast == *cast
    }
}

/// A place lent as the call argument at `id`.
struct Lent {
    id: AstId,
    span: Span,
    place: Place,
    from: LentFrom,
    mutability: Mutability,
}

/// What a lent place is: a place the program names, or the temporary an
/// argument that is no place is bound to, which nothing reads after the
/// call.
enum LentFrom {
    Place,
    Temporary,
}

/// What a pattern is applied to (RFC-0024).
enum PatSrc {
    Placed(Placed),
    Value { value: ValueId, ty: Ty },
}

impl PatSrc {
    fn ty(&self) -> &Ty {
        match self {
            PatSrc::Placed(placed) => placed.ty(),
            PatSrc::Value { ty, .. } => ty,
        }
    }
}

#[derive(Clone)]
enum Placed {
    Place {
        target: RefTarget,
        path: Vec<PathSeg>,
        ty: Ty,
    },
    Through {
        reference: ValueId,
        ty: Ty,
    },
}

impl Placed {
    fn ty(&self) -> &Ty {
        match self {
            Placed::Place { ty, .. } | Placed::Through { ty, .. } => ty,
        }
    }
}

struct PatternPart<'p> {
    seg: PathSeg,
    ty: Ty,
    pattern: &'p Pattern,
}

fn fields(path: &[Astr]) -> Vec<PathSeg> {
    path.iter().copied().map(PathSeg::Field).collect()
}

/// A context place lent to a call through a temporary local.
/// A local variable binding: its type and the storage slot it owns.
/// What a call's summary says it may touch; a type with no effect says
/// everything.
enum TouchedContexts {
    Every,
    These(BTreeSet<QualifiedRef>),
}

struct Local {
    ty: Ty,
    slot: ValueId,
}

/// A connective whose left operand can decide the result alone (RFC-0020).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShortCircuit {
    And,
    Or,
}

impl ShortCircuit {
    /// The result the left operand decides alone, which is also the value
    /// of the left operand that decides it.
    fn decided(self) -> bool {
        match self {
            ShortCircuit::And => false,
            ShortCircuit::Or => true,
        }
    }
}

/// The connective an operator is, for the two operators that short-circuit.
fn short_circuit_of(op: BinOp) -> Option<ShortCircuit> {
    match op {
        BinOp::And => Some(ShortCircuit::And),
        BinOp::Or => Some(ShortCircuit::Or),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Eq
        | BinOp::Neq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte
        | BinOp::Xor
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::Shr => None,
    }
}

/// A pattern that matches every value of its type: a binding, a context
/// bind, a tuple of such, or an object of such. The type checker has
/// already required every key an object pattern names to exist on the
/// source type, so the pattern cannot fail at run time.
/// A `match` whose arms are one dispatch: every arm is a variant pattern
/// whose payload asks nothing (RFC-0051 rule 3 -- a nested refutable payload is
/// more than a tag test) or a literal pattern that is a [`SwitchKey`], with
/// at most one catch-all, last, and every key of one kind. `keys` runs in
/// arm order and skips the catch-all; `catch_all` is that arm's index.
pub(crate) struct Dispatch {
    keys: Vec<SwitchKey>,
    catch_all: Option<usize>,
}

impl Dispatch {
    pub(crate) fn of(arms: &[MatchExprArm], interner: &Interner) -> Option<Self> {
        Self::plan(arms, interner)
    }

    /// Whether the arms can hold one dispatch at all: an arm that is a
    /// catch-all answers yes on its own, because `_` is always the way
    /// through (RFC-0051 rule 3).
    pub(crate) fn is_decidable(arms: &[MatchExprArm], interner: &Interner) -> bool {
        Self::plan(arms, interner).is_some() || arms.iter().any(Self::is_catch_all)
    }

    /// The first key two arms of this dispatch both name, which is an arm no
    /// value can reach because the one before it holds first.
    pub(crate) fn repeated_key(arms: &[MatchExprArm], interner: &Interner) -> Option<SwitchKey> {
        let plan = Self::plan(arms, interner)?;
        plan.keys
            .iter()
            .enumerate()
            .find(|(at, key)| plan.keys[..*at].contains(key))
            .map(|(_, key)| *key)
    }

    fn is_catch_all(arm: &MatchExprArm) -> bool {
        matches!(
            arm.pattern,
            Pattern::Wildcard { .. }
                | Pattern::Binding {
                    ref_kind: RefKind::Value,
                    ..
                }
        )
    }

    fn plan(arms: &[MatchExprArm], interner: &Interner) -> Option<Self> {
        let mut keys: Vec<SwitchKey> = Vec::with_capacity(arms.len());
        let mut catch_all = None;
        for (index, arm) in arms.iter().enumerate() {
            let key = match &arm.pattern {
                Pattern::Variant { tag, payload, .. }
                    if payload.as_deref().is_none_or(pattern_is_irrefutable) =>
                {
                    SwitchKey::Tag(*tag)
                }
                Pattern::Literal { value, .. } => SwitchKey::of_literal(value, interner)?,
                Pattern::Wildcard { .. }
                | Pattern::Binding {
                    ref_kind: RefKind::Value,
                    ..
                } if catch_all.is_none() => {
                    catch_all = Some(index);
                    continue;
                }
                _ => return None,
            };
            if catch_all.is_some() {
                // An arm after the catch-all can never be taken; the chain
                // keeps the arm order the source wrote.
                return None;
            }
            if keys.first().is_some_and(|first| !first.same_kind(key)) {
                return None;
            }
            keys.push(key);
        }
        (!keys.is_empty()).then_some(Self { keys, catch_all })
    }
}

/// `String` or `str`, or a reference to either: what `StringConcat` and
/// `StringEq` read (RFC-0062 rule 3).
fn holds_text(ty: &Ty) -> bool {
    match ty {
        Ty::String | Ty::Str => true,
        Ty::Ref(_, inner) => holds_text(&inner.ty()),
        _ => false,
    }
}

fn pattern_is_irrefutable(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Binding { .. } | Pattern::Wildcard { .. } | Pattern::ContextBind { .. } => true,
        Pattern::Tuple { elements, .. } => elements.iter().all(|e| match e {
            TuplePatternElem::Pattern(p) => pattern_is_irrefutable(p),
            TuplePatternElem::Wildcard(_) => true,
        }),
        Pattern::Object { fields, .. } => fields
            .iter()
            .all(|field| pattern_is_irrefutable(&field.pattern)),
        Pattern::Literal { .. } | Pattern::List { .. } | Pattern::Variant { .. } => false,
    }
}

/// Whether testing this pattern reads a part out of the value it is applied
/// to, rather than only reading the value itself. The interpreter's
/// `read_slot` and `unwrap_*` ops move a `Large` out of the slot they read,
/// so on a register a part read to test and read again to bind is a value
/// taken twice; a place is read through a reference instead.
fn test_reads_a_part(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Binding { .. }
        | Pattern::Wildcard { .. }
        | Pattern::ContextBind { .. }
        | Pattern::Literal { .. } => false,
        Pattern::List { .. } | Pattern::Object { .. } | Pattern::Tuple { .. } => true,
        Pattern::Variant { payload, .. } => payload
            .as_deref()
            .is_some_and(|p| !pattern_is_irrefutable(p)),
    }
}

impl<'a> Lowerer<'a> {
    pub fn new(interner: &'a Interner, resolution: Freeze<TypeResolution>, ret: Ty) -> Self {
        let coercion_lookup: FxHashMap<AstId, CastKind> =
            resolution.coercion_map.iter().cloned().collect();
        let initial_scope = FxHashMap::default();
        let mut body = MirBody::new();

        // Allocate param_regs for extern params (LLVM-style: params are SSA values).
        // param_regs[i] holds the initial value of extern_params[i].
        // SSA will use these as entry definitions instead of Ref+Load.
        for (name, ty) in &resolution.extern_params {
            let reg = body.val_factory.next();
            body.val_types.insert(reg, ty.clone());
            body.params.push((*name, reg));
        }
        body.task = resolution.effect.task;

        Self {
            body,
            interner,
            scopes: vec![initial_scope],
            resolution,
            ret,
            coercion_lookup,
            closures: FxHashMap::default(),
            closure_label_count: 0,
            order_slot: None,
            anyorder: None,
            result: None,
            context_slots: BTreeMap::new(),
            taken_out: Vec::new(),
            loops: Vec::new(),
        }
    }

    /// A template is its statements over one accumulator: a `String` the
    /// appends write through and the body returns (RFC-0071).
    pub fn lower_template(mut self, template: &Template) -> MirModule {
        let effect = self.resolution.effect.clone();
        self.enter_body_order(effect, template.span);
        self.enter_contexts(
            acvus_ast::direct_template_context_refs(template),
            template.span,
        );
        let empty = self.emit_empty_string(template.span);
        let slot = self.define_var(self.interner.intern(TEMPLATE_RESULT), Ty::String);
        self.emit_assign(template.span, RefTarget::Var(slot), vec![], empty);
        self.result = Some(slot);
        for stmt in &template.body {
            self.lower_stmt(stmt);
        }
        let result = self.emit_take(template.span, RefTarget::Var(slot), vec![], Ty::String);
        self.emit_return(template.span, result);
        self.build_module()
    }

    pub fn lower_script(mut self, script: &Script) -> MirModule {
        let effect = self.resolution.effect.clone();
        self.enter_body_order(effect, script.span);
        self.enter_contexts(acvus_ast::direct_script_context_refs(script), script.span);
        for stmt in &script.stmts {
            self.lower_stmt(stmt);
        }
        let val = match &script.tail {
            Some(tail) => self.lower_expr(tail),
            None => self.emit_unit(script.span),
        };
        self.emit_return(script.span, val);
        self.build_module()
    }

    /// Give the body being lowered its entry `Order` when its effect is not
    /// Pure: an order parameter, stored into a fresh slot that every
    /// effectful call reads and advances.
    fn enter_body_order(&mut self, effect: Effect, span: Span) {
        self.anyorder = None;
        if effect.is_pure() {
            self.order_slot = None;
            return;
        }
        let order_param = self.alloc_val();
        self.set_val_type(order_param, Ty::Order);
        self.body.order_param = Some(order_param);
        let slot = self.alloc_slot(Ty::Order, ValOrigin::Named(self.interner.intern("$order")));
        self.emit_assign(span, RefTarget::Var(slot), vec![], order_param);
        self.order_slot = Some(slot);
    }

    fn enter_contexts(&mut self, named: FxHashSet<QualifiedRef>, span: Span) {
        let named: BTreeSet<QualifiedRef> = named.into_iter().collect();
        for qref in named {
            let ty = self
                .resolution
                .context_types
                .get(&qref)
                .cloned()
                .unwrap_or_else(|| panic!("context {qref:?} is named but was not typed"));
            let slot = self.alloc_slot(ty.clone(), ValOrigin::Context(qref.name));
            self.fetch_into(span, qref, slot, ty);
            self.context_slots.insert(qref, slot);
        }
    }

    fn context_slot(&self, qref: QualifiedRef) -> ValueId {
        *self
            .context_slots
            .get(&qref)
            .unwrap_or_else(|| panic!("context {qref:?} has no variable in this body"))
    }

    fn slot_type(&self, slot: ValueId) -> Ty {
        self.body
            .val_types
            .get(&slot)
            .cloned()
            .unwrap_or_else(|| panic!("slot {slot:?} has no type"))
    }

    fn fetch_into(&mut self, span: Span, context: QualifiedRef, slot: ValueId, ty: Ty) {
        let fetched = self.alloc_val();
        self.set_val_type(fetched, ty);
        self.set_origin(fetched, ValOrigin::Context(context.name));
        self.emit_inst(
            span,
            InstKind::Fetch {
                dst: fetched,
                context,
            },
        );
        self.emit_assign(span, RefTarget::Var(slot), vec![], fetched);
    }

    fn commit_from(&mut self, span: Span, context: QualifiedRef, slot: ValueId) {
        let ty = self.slot_type(slot);
        let value = self.emit_take(span, RefTarget::Var(slot), vec![], ty);
        self.emit_inst(span, InstKind::Commit { context, value });
    }

    /// The summary of a call (RFC-0025 rule 5): the callee's, joined with that of
    /// every function value it is passed.
    fn touched_contexts(&self, callee_ty: &Ty, args: &[ValueId]) -> TouchedContexts {
        let mut touched = BTreeSet::new();
        let mut join = |ty: &Ty| -> bool {
            let Some(effect) = ty.effect() else {
                return false;
            };
            touched.extend(effect.reads.iter().copied());
            touched.extend(effect.writes.iter().copied());
            true
        };
        if !join(callee_ty) {
            return TouchedContexts::Every;
        }
        for arg in args {
            if let Some(ty) = self.body.val_types.get(arg)
                && matches!(ty, Ty::Fn { .. })
                && !join(ty)
            {
                return TouchedContexts::Every;
            }
        }
        TouchedContexts::These(touched)
    }

    fn bracketed_contexts(&self, callee_ty: &Ty, args: &[ValueId]) -> Vec<(QualifiedRef, ValueId)> {
        match self.touched_contexts(callee_ty, args) {
            TouchedContexts::Every => self.context_slots.iter().map(|(q, s)| (*q, *s)).collect(),
            TouchedContexts::These(touched) => self
                .context_slots
                .iter()
                .filter(|(q, _)| touched.contains(q))
                .map(|(q, s)| (*q, *s))
                .collect(),
        }
    }

    /// `inner?` (RFC-0038): test the operand's tag; on `Ok`/`Some` the
    /// payload is the value, on `Err`/`None` the function returns the
    /// operand's failure rebuilt at the function's return type.
    fn lower_try(&mut self, id: AstId, inner: &Expr, span: Span) -> ValueId {
        let src = self.lower_expr(inner);
        let operand_ty = self
            .body
            .val_types
            .get(&src)
            .cloned()
            .unwrap_or(Ty::error());
        let return_ty = self
            .resolution
            .try_returns
            .get(&id)
            .cloned()
            .unwrap_or(Ty::error());
        let (ok_tag, fail_tag, fail_carries) = match &operand_ty {
            Ty::Result(..) => ("Ok", "Err", true),
            _ => ("Some", "None", false),
        };
        let ok_tag = self.interner.intern(ok_tag);
        let fail_tag = self.interner.intern(fail_tag);

        let is_ok = self.alloc_val();
        self.set_val_type(is_ok, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::TestVariant {
                dst: is_ok,
                src,
                tag: ok_tag,
            },
        );
        let ok_label = self.alloc_label();
        let fail_label = self.alloc_label();
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: is_ok,
                then_label: ok_label,
                then_args: vec![],
                else_label: fail_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, fail_label);
        let failure = self.alloc_val();
        self.set_val_type(failure, return_ty.clone());
        let payload = if fail_carries {
            let err = self.alloc_val();
            self.set_val_type(err, self.payload_type(&operand_ty, fail_tag));
            self.emit_inst(span, InstKind::UnwrapVariant { dst: err, src });
            Some(err)
        } else {
            None
        };
        self.emit_inst(
            span,
            InstKind::MakeVariant {
                dst: failure,
                tag: fail_tag,
                payload,
            },
        );
        self.emit_return(span, failure);

        self.emit_label(span, ok_label);
        let dst = self.alloc_expr(id);
        self.emit_inst(span, InstKind::UnwrapVariant { dst, src });
        dst
    }

    /// Leave the body with `value`, yielding the current `Order` last.
    fn emit_return(&mut self, span: Span, value: ValueId) {
        let contexts: Vec<(QualifiedRef, ValueId)> =
            self.context_slots.iter().map(|(q, s)| (*q, *s)).collect();
        for (qref, slot) in contexts {
            self.commit_from(span, qref, slot);
        }
        let order = self.current_order(span);
        self.emit_inst(span, InstKind::Return { value, order });
    }

    /// Read the current `Order` of the body, if it has one.
    fn current_order(&mut self, span: Span) -> Option<ValueId> {
        self.order_slot
            .map(|slot| self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order))
    }

    /// Emit a call. A callee whose effect is not Pure takes the current
    /// `Order` and the call advances it.
    fn emit_call(
        &mut self,
        span: Span,
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
    ) {
        let bracketed = self.bracketed_contexts(&callee_ty, &args);
        for (qref, slot) in &bracketed {
            self.commit_from(span, *qref, *slot);
        }
        let effectful = callee_ty.effect().is_some_and(|e| !e.is_pure());
        let order = if effectful {
            let slot = self.order_slot.unwrap_or_else(|| {
                panic!("effectful call lowered inside a body whose effect is Pure")
            });
            let before = match self.anyorder {
                Some(scope) => scope.entry,
                None => self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order),
            };
            let after = self.alloc_val();
            self.set_val_type(after, Ty::Order);
            Some(OrderEdge { before, after })
        } else {
            None
        };
        self.emit_inst(
            span,
            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                order,
            },
        );
        for (qref, slot) in bracketed {
            let ty = self.slot_type(slot);
            self.fetch_into(span, qref, slot, ty);
        }
        let Some(edge) = order else {
            return;
        };
        match self.anyorder {
            Some(scope) => {
                let acc = self.emit_take(span, RefTarget::Var(scope.acc), vec![], Ty::Order);
                let merged = self.alloc_val();
                self.set_val_type(merged, Ty::Order);
                self.emit_inst(
                    span,
                    InstKind::Merge {
                        dst: merged,
                        orders: vec![acc, edge.after],
                    },
                );
                self.emit_assign(span, RefTarget::Var(scope.acc), vec![], merged);
            }
            None => {
                let slot = self.order_slot.expect("an order edge needs the slot");
                self.emit_assign(span, RefTarget::Var(slot), vec![], edge.after);
            }
        }
    }

    /// Lower `anyorder { body }`. The block takes the current `Order` once;
    /// every effectful call inside takes it, and the block yields a merge of
    /// everything they yielded. Inside a block already open, nothing changes.
    fn lower_anyorder(&mut self, body: &[Stmt], span: Span) {
        let Some(slot) = self.order_slot else {
            // A Pure body issues no effect; the block declares nothing.
            self.push_scope();
            for s in body {
                self.lower_stmt(s);
            }
            self.pop_scope();
            return;
        };
        let outer = self.anyorder;
        if outer.is_none() {
            let entry = self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order);
            let acc = self.alloc_slot(
                Ty::Order,
                ValOrigin::Named(self.interner.intern("$anyorder")),
            );
            self.emit_assign(span, RefTarget::Var(acc), vec![], entry);
            self.anyorder = Some(AnyorderScope { entry, acc });
        }
        self.push_scope();
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();
        if outer.is_none() {
            let scope = self.anyorder.take().expect("the block opened above");
            let exit = self.emit_take(span, RefTarget::Var(scope.acc), vec![], Ty::Order);
            self.emit_assign(span, RefTarget::Var(slot), vec![], exit);
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Store {
                place, expr, span, ..
            } => {
                self.lower_store(place, expr, *span);
            }
            Stmt::DerefStore {
                target, expr, span, ..
            } => {
                self.lower_deref_store(target, expr, *span);
            }
            Stmt::Expr(expr) => {
                self.lower_expr(expr);
            }

            // -- Script mode statements ------------------------------
            Stmt::LetBind {
                binder, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let ty = self
                    .body
                    .val_types
                    .get(&val)
                    .cloned()
                    .unwrap_or(Ty::error());
                let slot = self.define_var(binder.name, ty);
                self.emit_assign(*span, RefTarget::Var(slot), vec![], val);
            }
            Stmt::LetUninit { id, binder, .. } => {
                // Type from typeck (fresh variable, unified later).
                let ty = self.type_of_id(*id);
                let slot = self.define_var(binder.name, ty.clone());
                self.set_val_type(slot, ty);
                // No store - init_check tracks this as uninit.
            }
            Stmt::Assign {
                name, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let slot = self
                    .lookup_var_slot(*name)
                    .expect("Assign to undefined variable - should have been caught by typeck");
                self.emit_assign(*span, RefTarget::Var(slot), vec![], val);
            }
            Stmt::While {
                cond, body, span, ..
            } => {
                self.lower_while(cond, body, *span);
            }
            Stmt::Anyorder { body, span, .. } => {
                self.lower_anyorder(body, *span);
            }
            Stmt::WhileLet {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                self.lower_while_let(pattern, source, body, *span);
            }
            Stmt::For {
                id,
                callee_id,
                binder,
                head,
                body,
                span,
            } => {
                self.lower_for(*id, *callee_id, binder.name, head, body, *span);
            }
            Stmt::Break { span, .. } => self.leave_loop(Leave::Break, *span),
            Stmt::Continue { span, .. } => self.leave_loop(Leave::Continue, *span),
            Stmt::Append { expr, span, .. } => self.lower_append(expr, *span),
        }
    }

    /// One append onto the template's accumulator. The `&mut` is taken per
    /// append rather than once above the block: a borrow held across a
    /// loop's back edge would refuse every other name of the accumulator
    /// in the body.
    fn lower_append(&mut self, expr: &Expr, span: Span) {
        let part = self.lower_expr(expr);
        let Some(slot) = self.result else {
            return;
        };
        let target = self.emit_ref(
            span,
            RefTarget::Var(slot),
            vec![],
            Mutability::Mut,
            Ty::String,
        );
        self.emit_inst(span, InstKind::StringAppend { target, part });
    }

    /// Lower `match e { P1 => e1, .., Pn => en }` (RFC-0051).
    ///
    /// The scrutinee is evaluated once, into one `PatSrc` every arm reads.
    /// The dispatch this half emits is the chain the tag form already had:
    /// a `lower_pattern_test` (a `TestVariant` over the tag, for a variant
    /// pattern) and a `JumpIf` per arm, with a shared merge block carrying
    /// the result. The **last arm is the chain's else** -- it is not
    /// tested, because `validate`'s exhaustiveness pass is what makes the
    /// match exhaustive (a catch-all is written last, and a `Closed` set
    /// covered by the arms leaves the last arm the only one that can hold).
    ///
    /// The second half replaces the whole loop below with one
    /// `Terminator::Switch { tag, arms, default }`.
    fn lower_match_expr(
        &mut self,
        id: AstId,
        scrutinee: &Expr,
        arms: &[MatchExprArm],
        span: Span,
    ) -> ValueId {
        let result_ty = self.type_of_id(id);
        let Some((last, tested)) = arms.split_last() else {
            // `match e { }`: the scrutinee still runs; the match has no
            // value of its own.
            self.lower_expr(scrutinee);
            return self.emit_unit(span);
        };
        if let Some(dispatch) = Dispatch::of(arms, self.interner) {
            return self.lower_match_switch(&dispatch, scrutinee, arms, result_ty, span);
        }
        let src = self.pattern_source(scrutinee, arms.iter().map(|arm| &arm.pattern));
        let merge_label = self.alloc_label();
        let first_arm_label = self.alloc_label();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: first_arm_label,
                args: vec![],
            },
        );
        self.emit_label(span, first_arm_label);

        for arm in tested {
            let matched = self.lower_pattern_test(&arm.pattern, &src, arm.span);
            let body_label = self.alloc_label();
            let next_label = self.alloc_label();
            self.emit_inst(
                arm.span,
                InstKind::JumpIf {
                    cond: matched,
                    then_label: body_label,
                    then_args: vec![],
                    else_label: next_label,
                    else_args: vec![],
                },
            );
            self.emit_label(arm.span, body_label);
            let value = self.lower_arm_body(arm, &src);
            self.emit_inst(
                arm.span,
                InstKind::Jump {
                    label: merge_label,
                    args: vec![value],
                },
            );
            self.emit_label(arm.span, next_label);
        }

        let value = self.lower_arm_body(last, &src);
        self.emit_inst(
            last.span,
            InstKind::Jump {
                label: merge_label,
                args: vec![value],
            },
        );

        let result = self.alloc_val();
        self.set_val_type(result, result_ty);
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: merge_label,
                params: vec![result],
            },
        );
        result
    }

    /// Lower a `match` whose arms are one dispatch over a tag: the
    /// scrutinee is read once into the value whose tag the `Switch` reads,
    /// and each arm is its own block. `Terminator::Switch` is the shape
    /// that names a `match` and nothing else, which is what
    /// `validate::exhaustive` reads (RFC-0051 rules 3, 4 and 5).
    fn lower_match_switch(
        &mut self,
        dispatch: &Dispatch,
        scrutinee: &Expr,
        arms: &[MatchExprArm],
        result_ty: Ty,
        span: Span,
    ) -> ValueId {
        let src = self.pattern_source(scrutinee, arms.iter().map(|arm| &arm.pattern));
        let tag = self.dispatch_source(dispatch.keys[0], &src, span);
        let merge_label = self.alloc_label();
        let arm_labels: Vec<Label> = arms.iter().map(|_| self.alloc_label()).collect();
        let switch_arms: Vec<(SwitchKey, Label, Vec<ValueId>)> = dispatch
            .keys
            .iter()
            .zip(&arm_labels)
            .map(|(key, label)| (*key, *label, Vec::new()))
            .collect();
        let default = dispatch
            .catch_all
            .map(|index| (arm_labels[index], Vec::new()));
        self.emit_inst(
            span,
            InstKind::Switch {
                tag,
                arms: switch_arms,
                default,
            },
        );

        for (arm, label) in arms.iter().zip(&arm_labels) {
            self.emit_label(arm.span, *label);
            let value = self.lower_arm_body(arm, &src);
            self.emit_inst(
                arm.span,
                InstKind::Jump {
                    label: merge_label,
                    args: vec![value],
                },
            );
        }

        let result = self.alloc_val();
        self.set_val_type(result, result_ty);
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: merge_label,
                params: vec![result],
            },
        );
        result
    }

    fn dispatch_source(&mut self, key: SwitchKey, src: &PatSrc, span: Span) -> ValueId {
        match key {
            SwitchKey::Tag(_) => self.tag_reference(src, span),
            SwitchKey::Int(_) | SwitchKey::Bool(_) | SwitchKey::Char(_) | SwitchKey::Str(_) => {
                self.read_leaf(src, span)
            }
        }
    }

    /// One arm's bindings, statements and tail, in a scope of their own.
    /// An arm with no tail has the value `Unit`, as an `if` branch with no
    /// tail has.
    fn lower_arm_body(&mut self, arm: &MatchExprArm, src: &PatSrc) -> ValueId {
        self.push_scope();
        self.lower_pattern_bind(&arm.pattern, src, arm.span);
        for s in &arm.body {
            self.lower_stmt(s);
        }
        let value = match &arm.tail {
            Some(tail) => self.lower_expr(tail),
            None => self.emit_unit(arm.span),
        };
        self.pop_scope();
        value
    }

    /// One arm matched for its effect: test the pattern, and on a match bind
    /// it and run the body, joining where the failed test goes. An irrefutable
    /// pattern has no test and no branch. No value leaves the join.
    fn lower_match_bind_arm(
        &mut self,
        pattern: &Pattern,
        src: &PatSrc,
        body: &[Stmt],
        tail: Option<&Expr>,
        span: Span,
    ) {
        if pattern_is_irrefutable(pattern) {
            self.lower_match_bind_body(pattern, src, body, tail, span);
            return;
        }

        let matched = self.lower_pattern_test(pattern, src, span);
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();
        let pending = self.open_diamond(
            span,
            matched,
            DiamondLabels {
                then_label: body_label,
                else_label: end_label,
                join: end_label,
            },
        );

        self.emit_label(span, body_label);
        self.lower_match_bind_body(pattern, src, body, tail, span);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: end_label,
                args: vec![],
            },
        );
        self.close_diamond(pending);

        self.emit_label(span, end_label);
    }

    /// The arm's bindings and its statements, in a scope of their own. A tail
    /// expression is run for its effect and its value discarded.
    fn lower_match_bind_body(
        &mut self,
        pattern: &Pattern,
        src: &PatSrc,
        body: &[Stmt],
        tail: Option<&Expr>,
        span: Span,
    ) {
        self.push_scope();
        self.lower_pattern_bind(pattern, src, span);
        for s in body {
            self.lower_stmt(s);
        }
        if let Some(tail) = tail {
            self.lower_expr(tail);
        }
        self.pop_scope();
    }

    /// One traversal (RFC-0057). The header block holds nothing but the
    /// terminator: the comparison, the element read and the advance are the
    /// terminator's, and the body takes the element and the counter as its
    /// leading parameters.
    fn lower_for(
        &mut self,
        id: AstId,
        callee_id: AstId,
        binding: Astr,
        head: &ForHead,
        body: &[Stmt],
        span: Span,
    ) {
        let kind = self.for_kind(id);
        let source = self.lower_for_source(id, callee_id, kind, head);

        let header = self.alloc_label();
        let body_label = self.alloc_label();
        let exit = self.alloc_label();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: header,
                args: vec![],
            },
        );

        let element = self.type_of_id(id);
        let binding_ty = match kind {
            ForKind::Slice(mutability) => {
                Ty::Ref(mutability, Box::new(TypeArg::uniform(element.clone())))
            }
            ForKind::Array | ForKind::Range => element.clone(),
        };
        let elem = self.alloc_val();
        self.set_val_type(elem, binding_ty.clone());
        self.set_origin(elem, ValOrigin::Named(binding));
        let mut params = vec![elem];
        if kind != ForKind::Range {
            let index = self.alloc_val();
            self.set_val_type(index, Ty::U64);
            params.push(index);
        }

        self.emit_label(span, header);
        self.emit_inst(
            span,
            InstKind::For {
                source,
                body: body_label,
                body_args: vec![],
                exit,
                exit_args: vec![],
            },
        );

        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: body_label,
                params,
            },
        );
        self.push_scope();
        let slot = self.define_var(binding, binding_ty);
        self.emit_assign(span, RefTarget::Var(slot), vec![], elem);
        self.loops.push(Loop { header, exit });
        for s in body {
            self.lower_stmt(s);
        }
        self.loops.pop();
        self.pop_scope();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: header,
                args: vec![],
            },
        );

        self.emit_label(span, exit);
    }

    /// The source, taken once before the header: a container's `as_slice` at
    /// the instance the checker settled on the statement's `callee_id`, an
    /// array moved into the loop, or the range's two bounds.
    fn lower_for_source(
        &mut self,
        id: AstId,
        callee_id: AstId,
        kind: ForKind,
        head: &ForHead,
    ) -> ForSource {
        match (kind, head) {
            (
                ForKind::Slice(mutability),
                ForHead::Value(Expr::Borrow {
                    place,
                    span: borrow_span,
                    ..
                }),
            ) => {
                let slice = self.take_slice(place, callee_id, id, mutability, *borrow_span);
                match mutability {
                    Mutability::Shared => ForSource::Slice(slice),
                    Mutability::Mut => ForSource::SliceMut(slice),
                }
            }
            (ForKind::Array, ForHead::Value(array)) => ForSource::Array(self.lower_expr(array)),
            (ForKind::Range, ForHead::Range { lo, hi }) => ForSource::Range {
                at: self.lower_expr(lo),
                hi: self.lower_expr(hi),
            },
            (kind, _) => panic!("the checker settled {kind:?} on a head of another shape"),
        }
    }

    fn for_kind(&self, id: AstId) -> ForKind {
        self.resolution
            .for_kinds
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("type checking settles every `for` head"))
    }

    /// `break` and `continue` name the innermost loop (RFC-0057 rule 4):
    /// the jump to its exit and the jump to its header, which is its latch.
    /// What the source wrote after one lands in a block no jump reaches, as
    /// it does after an expression typed `!`. The drops of the scopes the
    /// jump leaves are `optimize::drop_insertion`'s edge drops, as they are
    /// on every other edge.
    fn leave_loop(&mut self, leave: Leave, span: Span) {
        let Some(loop_) = self.loops.last() else {
            panic!("type checking refuses a `break` or a `continue` outside a loop")
        };
        let label = match leave {
            Leave::Break => loop_.exit,
            Leave::Continue => loop_.header,
        };
        self.emit_inst(
            span,
            InstKind::Jump {
                label,
                args: vec![],
            },
        );
        let unreachable = self.alloc_label();
        self.emit_label(span, unreachable);
    }

    fn lower_while(&mut self, cond: &Expr, body: &[Stmt], span: Span) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );
        self.emit_label(span, loop_label);

        let cond_val = self.lower_expr(cond);
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: cond_val,
                then_label: body_label,
                then_args: vec![],
                else_label: end_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, body_label);
        self.push_scope();
        self.loops.push(Loop {
            header: loop_label,
            exit: end_label,
        });
        for s in body {
            self.lower_stmt(s);
        }
        self.loops.pop();
        self.pop_scope();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );

        self.emit_label(span, end_label);
    }

    /// Lower `while let pattern = source { body }`.
    ///
    /// loop_label:
    ///   src = lower(source)
    ///   matched = test(pattern, src)
    ///   JumpIf matched -> body_label, end_label
    /// body_label:
    ///   bind(pattern, src)
    ///   body...
    ///   Jump loop_label
    /// end_label:
    fn lower_while_let(&mut self, pattern: &Pattern, source: &Expr, body: &[Stmt], span: Span) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );
        self.emit_label(span, loop_label);

        let src = self.pattern_source(source, [pattern]);
        let matched = self.lower_pattern_test(pattern, &src, span);
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: matched,
                then_label: body_label,
                then_args: vec![],
                else_label: end_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, body_label);
        self.push_scope();
        self.loops.push(Loop {
            header: loop_label,
            exit: end_label,
        });
        self.lower_pattern_bind(pattern, &src, span);
        for s in body {
            self.lower_stmt(s);
        }
        self.loops.pop();
        self.pop_scope();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );

        self.emit_label(span, end_label);
    }

    /// Lower `if cond { body; tail } else { ... }` as an expression.
    fn lower_if_expr(
        &mut self,
        id: AstId,
        cond: &Expr,
        then_body: &[Stmt],
        then_tail: &Option<Box<Expr>>,
        else_branch: &Option<Box<ElseBranch>>,
        span: Span,
    ) -> ValueId {
        let result_ty = self.type_of_id(id);
        let then_label = self.alloc_label();
        let merge_label = self.alloc_label();

        let cond_val = self.lower_expr(cond);

        match else_branch {
            Some(eb) => {
                let else_label = self.alloc_label();
                let pending = self.open_diamond(
                    span,
                    cond_val,
                    DiamondLabels {
                        then_label,
                        else_label,
                        join: merge_label,
                    },
                );

                // Then branch.
                self.emit_label(span, then_label);
                self.push_scope();
                for s in then_body {
                    self.lower_stmt(s);
                }
                let then_val = match then_tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![then_val],
                    },
                );

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![else_val],
                    },
                );
                self.close_diamond(pending);

                // Merge.
                let result = self.alloc_val();
                self.set_val_type(result, result_ty);
                self.emit_inst(
                    span,
                    InstKind::BlockLabel {
                        label: merge_label,
                        params: vec![result],
                    },
                );
                result
            }
            None => {
                // No else: then branch stores to a var slot, else path skips.
                // The result is Unit (no value merge needed).
                let pending = self.open_diamond(
                    span,
                    cond_val,
                    DiamondLabels {
                        then_label,
                        else_label: merge_label,
                        join: merge_label,
                    },
                );

                self.emit_label(span, then_label);
                self.push_scope();
                for s in then_body {
                    self.lower_stmt(s);
                }
                if let Some(tail) = then_tail {
                    self.lower_expr(tail); // value discarded
                }
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![],
                    },
                );
                self.close_diamond(pending);

                self.emit_label(span, merge_label);
                self.emit_unit(span)
            }
        }
    }

    /// Lower `if let pattern = source { body; tail } else { ... }` as an expression.
    fn lower_if_let_expr(
        &mut self,
        id: AstId,
        pattern: &Pattern,
        source: &Expr,
        then_body: &[Stmt],
        then_tail: &Option<Box<Expr>>,
        else_branch: &Option<Box<ElseBranch>>,
        span: Span,
    ) -> ValueId {
        let result_ty = self.type_of_id(id);
        let src = self.pattern_source(source, [pattern]);

        match else_branch {
            Some(eb) => {
                let matched = self.lower_pattern_test(pattern, &src, span);
                let then_label = self.alloc_label();
                let merge_label = self.alloc_label();
                let else_label = self.alloc_label();
                let pending = self.open_diamond(
                    span,
                    matched,
                    DiamondLabels {
                        then_label,
                        else_label,
                        join: merge_label,
                    },
                );

                // Then branch.
                self.emit_label(span, then_label);
                self.push_scope();
                self.lower_pattern_bind(pattern, &src, span);
                for s in then_body {
                    self.lower_stmt(s);
                }
                let then_val = match then_tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![then_val],
                    },
                );

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![else_val],
                    },
                );
                self.close_diamond(pending);

                // Merge.
                let result = self.alloc_val();
                self.set_val_type(result, result_ty);
                self.emit_inst(
                    span,
                    InstKind::BlockLabel {
                        label: merge_label,
                        params: vec![result],
                    },
                );
                result
            }
            None => {
                // The same match the tag form `pattern = source { body };`
                // writes: one arm, run for its effect.
                self.lower_match_bind_arm(pattern, &src, then_body, then_tail.as_deref(), span);
                self.emit_unit(span)
            }
        }
    }

    /// Lower an else branch, returning the value it produces.
    fn lower_else_branch(&mut self, eb: &ElseBranch, span: Span, _merge_label: Label) -> ValueId {
        match eb {
            ElseBranch::ElseIf(expr) => self.lower_expr(expr),
            ElseBranch::Else { body, tail, .. } => {
                self.push_scope();
                for s in body {
                    self.lower_stmt(s);
                }
                let val = match tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                val
            }
        }
    }

    fn build_module(self) -> MirModule {
        MirModule {
            main: self.body,
            closures: self.closures,
            ret: self.ret,
        }
    }

    fn place_or_temporary(&mut self, place: &Expr) -> Place {
        let Projected { base, fields } = projected(place);
        let target = match self.base_target(base) {
            Some(target) => target,
            None => {
                let value = self.lower_before_coercion(base);
                self.temporary(base.span(), value, self.type_of_id(base.id()))
            }
        };
        let ty = self.place_ty(place.id(), base.id(), &fields);
        Place {
            target,
            path: self::fields(&fields),
            ty,
        }
    }

    fn place_ty(&self, place: AstId, base: AstId, fields: &[Astr]) -> Ty {
        let ty = self.type_of_id(place);
        match (self.place_base(base), fields) {
            (PlaceBase::ThroughReferenceIn(_) | PlaceBase::ThroughReference, []) => {
                let Ty::Ref(_, referent) = ty else {
                    panic!("the checker reads a base through a reference only where it is one")
                };
                referent.into_ty()
            }
            _ => ty,
        }
    }

    fn store_target(&mut self, place: &acvus_ast::Place) -> Place {
        let (base, fields) = projected_store(place);
        let target = match self.place_base(base.id()) {
            PlaceBase::Storage(storage) => self.storage(storage),
            PlaceBase::ThroughReferenceIn(storage) => {
                self.through_stored_reference(storage, base.id(), base.span())
            }
            PlaceBase::Element(access) => {
                let element = Element::of_store(base)
                    .expect("the checker records an element base only on an element store");
                self.element_reference(access, element)
            }
            other @ (PlaceBase::ThroughReference | PlaceBase::Temporary) => {
                panic!("a store target's base is a storage or an element, not {other:?}")
            }
        };
        let ty = self.place_ty(place.id(), base.id(), &fields);
        Place {
            target,
            path: self::fields(&fields),
            ty,
        }
    }

    /// An operator operand borrowed for the expression (RFC-0020).
    fn lend_operand(&mut self, operand: &Expr) -> ValueId {
        if let Passing::AsIs = self.passing(operand) {
            return self.lower_expr(operand);
        }
        let span = operand.span();
        let Projected { base, fields } = projected(operand);
        if let Some(target) = self.base_target(base) {
            // A place a `&mut` holds is lent as the shared reborrow of what
            // it names, as `&r` is (RFC-0029 rule 3).
            let ty = self.place_ty(operand.id(), base.id(), &fields);
            return self.emit_ref(span, target, self::fields(&fields), Mutability::Shared, ty);
        }
        let ty = self.type_of_id(operand.id());
        let value = self.lower_expr(operand);
        let owned = self.temporary(span, value, ty.clone());
        self.emit_ref(span, owned, vec![], Mutability::Shared, ty)
    }

    /// The slot a value with no storage of its own is bound to. It is an
    /// ordinary slot, so `optimize::drop_insertion` releases it where it
    /// dies, as it releases the slot a `let` binds.
    fn temporary(&mut self, span: Span, value: ValueId, ty: Ty) -> RefTarget {
        let slot = self.alloc_val();
        self.set_val_type(slot, ty);
        self.emit_assign(span, RefTarget::Var(slot), vec![], value);
        RefTarget::Var(slot)
    }

    /// `a[i]` (RFC-0047): the container's slice, then the element. The
    /// mode is the checker's, from `index_modes`; nothing here decides how
    /// the element comes back.
    fn lower_index(
        &mut self,
        id: AstId,
        callee_id: AstId,
        object: &Expr,
        index: &Expr,
        span: Span,
    ) -> ValueId {
        let access = self.index_access(id);
        self.lower_index_as(access, id, callee_id, object, index, span)
    }

    fn index_access(&self, id: AstId) -> IndexAccess {
        self.resolution
            .index_access
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("type checking settles every index expression"))
    }

    /// The element as a reference into the slice, whatever its type: what
    /// a place below an `a[i]` names — `&a[i]`, `a[i][j]`, `a[i].f`, and a
    /// method receiver.
    fn lower_index_as(
        &mut self,
        access: IndexAccess,
        id: AstId,
        callee_id: AstId,
        object: &Expr,
        index: &Expr,
        span: Span,
    ) -> ValueId {
        let IndexAccess { mutability, mode } = access;
        let slice = self.take_slice(object, callee_id, id, mutability, span);
        let index = self.lower_expr(index);
        let element = self.type_of_id(id);
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            match mode {
                IndexMode::Copy => element,
                IndexMode::Ref => Ty::Ref(mutability, Box::new(TypeArg::uniform(element))),
            },
        );
        self.emit_inst(
            span,
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            },
        );
        dst
    }

    /// The whole run of a container's elements, at the instance the checker
    /// settled on the index expression's callee id.
    fn take_slice(
        &mut self,
        object: &Expr,
        callee_id: AstId,
        id: AstId,
        mutability: Mutability,
        span: Span,
    ) -> ValueId {
        let Some(CallTarget::Declared(Callee::Extern {
            id: qref, instance, ..
        })) = self.resolution.calls.get(&callee_id).cloned()
        else {
            panic!("type checking settles an `as_slice` instance on every index expression")
        };
        let taken_from = self.taken_out.len();
        let container = self.receiver(object);
        debug_assert!(
            self.taken_out.len() == taken_from,
            "a container lent for a slice crosses no boundary, so it is not cast back"
        );
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            Ty::Ref(
                mutability,
                Box::new(TypeArg::uniform(Ty::Slice(Box::new(self.type_of_id(id))))),
            ),
        );
        self.emit_inst(
            span,
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                instance: ExternInstance { id: qref, instance },
            },
        );
        dst
    }

    fn place_base(&self, base: AstId) -> PlaceBase {
        *self
            .resolution
            .place_bases
            .get(&base)
            .expect("the checker settles the base of every place the lowering reads")
    }

    fn base_target(&mut self, base: &Expr) -> Option<RefTarget> {
        match self.place_base(base.id()) {
            PlaceBase::Storage(storage) => Some(self.storage(storage)),
            PlaceBase::ThroughReferenceIn(storage) => {
                Some(self.through_stored_reference(storage, base.id(), base.span()))
            }
            PlaceBase::ThroughReference => Some(RefTarget::Through(self.lower_expr(base))),
            PlaceBase::Element(access) => {
                let element = Element::of(base)
                    .expect("the checker records an element base only on an index into a place");
                Some(self.element_reference(access, element))
            }
            PlaceBase::Temporary => None,
        }
    }

    fn element_reference(&mut self, access: IndexAccess, element: Element<'_>) -> RefTarget {
        let Element {
            id,
            callee_id,
            container,
            index,
            span,
        } = element;
        RefTarget::Through(self.lower_index_as(access, id, callee_id, container, index, span))
    }

    fn through_stored_reference(&mut self, storage: Storage, base: AstId, span: Span) -> RefTarget {
        let target = self.storage(storage);
        let reference = self.emit_take(span, target, vec![], self.type_of_id(base));
        RefTarget::Through(reference)
    }

    fn storage(&mut self, storage: Storage) -> RefTarget {
        match storage {
            Storage::Local(name) => RefTarget::Var(self.var_slot(name)),
            Storage::Input(name) => match self.try_param_slot(name) {
                Some(param) => RefTarget::Param(param),
                None => RefTarget::Var(self.var_slot(name)),
            },
            Storage::Context(qref) => RefTarget::Var(self.context_slot(qref)),
        }
    }

    /// Move `value` into a storage.
    fn emit_assign(&mut self, span: Span, target: RefTarget, path: Vec<PathSeg>, value: ValueId) {
        self.emit_inst(
            span,
            InstKind::Assign {
                target,
                path,
                value,
                restores: false,
            },
        );
    }

    /// Move the value of a storage out into a fresh value of `ty`.
    fn emit_take(&mut self, span: Span, target: RefTarget, path: Vec<PathSeg>, ty: Ty) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, ty);
        self.emit_inst(
            span,
            InstKind::Take {
                dst,
                target,
                path,
                taken_out: false,
            },
        );
        dst
    }

    /// A register holding a `&T` read at `T` where `T` is a word: the word
    /// copy of RFC-0018, which is what an operator's operand and a closure's
    /// capture both take. A register of any other type is returned as it is.
    fn read_word_through(&mut self, span: Span, taken: ValueId) -> ValueId {
        let Some(Ty::Ref(_, arg)) = self.body.val_types.get(&taken).cloned() else {
            return taken;
        };
        if arg.ty().is_word() != Some(true) {
            return taken;
        }
        self.emit_take(span, RefTarget::Through(taken), vec![], arg.into_ty())
    }

    /// A reference to a storage: a fresh `&T` / `&mut T` value.
    fn emit_ref(
        &mut self,
        span: Span,
        target: RefTarget,
        path: Vec<PathSeg>,
        mutability: Mutability,
        inner_ty: Ty,
    ) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            Ty::Ref(mutability, Box::new(TypeArg::uniform(inner_ty))),
        );
        self.set_origin(dst, ValOrigin::RefField(target.clone(), path.clone()));
        self.emit_inst(
            span,
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            },
        );
        dst
    }

    fn alloc_val(&mut self) -> ValueId {
        self.body.val_factory.next()
    }

    fn alloc_label(&mut self) -> Label {
        let l = Label(self.body.label_count);
        self.body.label_count += 1;
        l
    }

    /// Allocate a closure label from the global counter (not body-local).
    /// Prevents label collisions when nested closures each run in a sub-body.
    fn alloc_closure_label(&mut self) -> Label {
        let l = Label(self.closure_label_count);
        self.closure_label_count += 1;
        l
    }

    fn emit(&mut self, inst: Inst) {
        self.body.insts.push(inst);
    }

    fn emit_inst(&mut self, span: Span, kind: InstKind) {
        self.emit(Inst { span, kind });
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop().expect("a scope to pop");
    }

    /// Introduce a binding in the current scope with a fresh slot. A name
    /// already bound, here or outside, is shadowed.
    fn alloc_slot(&mut self, ty: Ty, origin: ValOrigin) -> ValueId {
        let slot = self.alloc_val();
        self.set_val_type(slot, ty);
        self.set_origin(slot, origin);
        slot
    }

    fn define_var(&mut self, name: Astr, ty: Ty) -> ValueId {
        let slot = self.alloc_slot(ty.clone(), ValOrigin::Named(name));
        let scope = self.scopes.last_mut().expect("a scope to define in");
        scope.insert(name, Local { ty, slot });
        slot
    }

    fn is_defined(&self, name: Astr) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(&name))
    }

    fn var_type(&self, name: Astr) -> Ty {
        self.lookup_local(name)
            .map(|l| l.ty.clone())
            .unwrap_or_else(Ty::error)
    }

    /// The innermost binding of `name`.
    fn lookup_local(&self, name: Astr) -> Option<&Local> {
        self.scopes.iter().rev().find_map(|scope| scope.get(&name))
    }

    fn lookup_var_slot(&self, name: Astr) -> Option<ValueId> {
        self.lookup_local(name).map(|l| l.slot)
    }

    /// The slot of the innermost binding of `name`; the name must be bound.
    fn var_slot(&mut self, name: Astr) -> ValueId {
        self.lookup_var_slot(name)
            .unwrap_or_else(|| panic!("variable {:?} used before it is bound", name))
    }

    /// Look up the param_reg for an extern parameter by name.
    /// Returns None if not found (e.g., inside a lambda where the param was captured).
    fn try_param_slot(&self, name: Astr) -> Option<ValueId> {
        for (pname, preg) in &self.body.params {
            if *pname == name {
                return Some(*preg);
            }
        }
        None
    }

    fn set_val_type(&mut self, val: ValueId, ty: Ty) {
        self.body.val_types.insert(val, ty);
    }

    fn set_origin(&mut self, val: ValueId, origin: ValOrigin) {
        self.body.debug.set(val, origin);
    }

    fn emit_label(&mut self, span: Span, label: Label) {
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label,
                params: vec![],
            },
        );
    }

    /// Allocate a new value with type inferred from AST id.
    fn alloc_typed(&mut self, id: AstId) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, self.type_of_id(id));
        dst
    }

    /// Allocate a new value with type inferred from AST id and origin set to Expr.
    fn alloc_expr(&mut self, id: AstId) -> ValueId {
        let dst = self.alloc_typed(id);
        self.set_origin(dst, ValOrigin::Expr);
        dst
    }

    /// `left && right`, with `right` lowered only on the path that reaches
    /// it (RFC-0020).
    fn emit_and<Right>(&mut self, span: Span, left: ValueId, right: Right) -> ValueId
    where
        Right: FnOnce(&mut Self) -> ValueId,
    {
        let dst = self.alloc_val();
        self.lower_short_circuit(span, dst, left, ShortCircuit::And, right)
    }

    /// RFC-0020: `a && b` is `if a { b } else { false }` and `a || b` is
    /// `if a { true } else { b }`. `right` is lowered inside the block the
    /// other path enters, so the right operand's effects happen only where
    /// it is evaluated.
    fn lower_short_circuit<Right>(
        &mut self,
        span: Span,
        dst: ValueId,
        left: ValueId,
        connective: ShortCircuit,
        right: Right,
    ) -> ValueId
    where
        Right: FnOnce(&mut Self) -> ValueId,
    {
        let right_label = self.alloc_label();
        let decided_label = self.alloc_label();
        let merge_label = self.alloc_label();

        let labels = match connective {
            ShortCircuit::And => DiamondLabels {
                then_label: right_label,
                else_label: decided_label,
                join: merge_label,
            },
            ShortCircuit::Or => DiamondLabels {
                then_label: decided_label,
                else_label: right_label,
                join: merge_label,
            },
        };
        let pending = self.open_diamond(span, left, labels);

        self.emit_label(span, right_label);
        let right_val = right(self);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: merge_label,
                args: vec![right_val],
            },
        );

        self.emit_label(span, decided_label);
        let decided_val = self.emit_const_bool(span, connective.decided());
        self.emit_inst(
            span,
            InstKind::Jump {
                label: merge_label,
                args: vec![decided_val],
            },
        );

        self.close_diamond(pending);

        self.set_val_type(dst, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: merge_label,
                params: vec![dst],
            },
        );
        dst
    }

    #[must_use]
    fn open_diamond(&mut self, span: Span, cond: ValueId, labels: DiamondLabels) -> PendingDiamond {
        let DiamondLabels {
            then_label,
            else_label,
            join,
        } = labels;
        let at = self.body.insts.len();
        self.emit_inst(
            span,
            InstKind::Diamond {
                cond,
                then_label,
                then_args: vec![],
                else_label,
                else_args: vec![],
                join,
            },
        );
        PendingDiamond { at, labels }
    }

    /// RFC-0063 rule 1 admits a `Diamond` only where both arms rejoin. An
    /// arm that `break`s, `continue`s or ends in a call typed `!` leaves
    /// through an edge of its own, and the branch is a `JumpIf`. The arms are
    /// the instructions between the branch and here and nothing else, so that
    /// is the range the question is asked of.
    fn close_diamond(&mut self, pending: PendingDiamond) {
        let PendingDiamond { at, labels } = pending;
        let arms = &self.body.insts[at..];
        if reaches(arms, labels.then_label, labels.join)
            && reaches(arms, labels.else_label, labels.join)
        {
            return;
        }
        let InstKind::Diamond {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } = &mut self.body.insts[at].kind
        else {
            panic!("`open_diamond` wrote a `Diamond` at this index")
        };
        let demoted = InstKind::JumpIf {
            cond: *cond,
            then_label: *then_label,
            then_args: std::mem::take(then_args),
            else_label: *else_label,
            else_args: std::mem::take(else_args),
        };
        self.body.insts[at].kind = demoted;
    }

    /// Emit short-circuit merge: `ok_val` flows into the success path,
    /// `false` into the fail path, and a block param merges them.
    fn emit_fail_merge(&mut self, span: Span, ok_val: ValueId, labels: DiamondLabels) -> ValueId {
        let (fail_label, result_label) = (labels.else_label, labels.join);
        let result_param = self.alloc_val();
        self.set_val_type(result_param, Ty::Bool);

        self.emit_inst(
            span,
            InstKind::Jump {
                label: result_label,
                args: vec![ok_val],
            },
        );

        // Fail path.
        self.emit_label(span, fail_label);
        let false_val = self.emit_const_bool(span, false);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: result_label,
                args: vec![false_val],
            },
        );

        // Merge.
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: result_label,
                params: vec![result_param],
            },
        );
        result_param
    }

    fn emit_unit(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Unit);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::Unit,
            },
        );
        dst
    }

    fn emit_const_bool(&mut self, span: Span, value: bool) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::Bool(value),
            },
        );
        dst
    }

    fn type_of_id(&self, id: AstId) -> Ty {
        self.resolution
            .type_map
            .get(&id)
            .cloned()
            .unwrap_or(Ty::error())
    }

    // --- Node lowering ---

    fn emit_empty_string(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::String(String::new()),
            },
        );
        dst
    }

    /// If the coercion map indicates this span needs a cast, emit a Cast
    /// instruction and return the new ValueId. Otherwise return `val` as-is.
    fn maybe_cast(&mut self, id: AstId, span: Span, val: ValueId) -> ValueId {
        match self.coercion_lookup.get(&id).cloned() {
            Some(CastKind::Extern(cast)) => self.emit_extern_cast(span, &cast, val),
            Some(CastKind::Slice {
                mutability,
                as_slice,
            }) => self.emit_as_slice(span, val, mutability, &as_slice),
            Some(CastKind::Str { as_str }) => {
                self.emit_as_slice(span, val, Mutability::Shared, &as_str)
            }
            Some(CastKind::ThroughRef { .. }) => {
                unreachable!("a cast through a reference is lowered where the argument is lent")
            }
            Some(CastKind::Reborrow { shared }) => {
                let dst = self.alloc_val();
                self.set_val_type(dst, shared);
                self.emit_inst(
                    span,
                    InstKind::Ref {
                        dst,
                        target: RefTarget::Through(val),
                        path: vec![],
                        mutability: Mutability::Shared,
                    },
                );
                dst
            }
            None => val,
        }
    }

    /// The container's own `as_slice` of a `&C` the caller already holds,
    /// where the parameter is a `&[T]` (RFC-0047 rule 6). The slice's type is
    /// the instance's return type, as every other coercion's result is.
    fn emit_as_slice(
        &mut self,
        span: Span,
        container: ValueId,
        mutability: Mutability,
        as_slice: &ExternCast,
    ) -> ValueId {
        let Ty::Fn { ret, .. } = &as_slice.callee_ty else {
            panic!(
                "an as_slice coercion's callee_ty is not Fn: {:?}",
                as_slice.callee_ty
            )
        };
        let dst = self.alloc_val();
        self.set_val_type(dst, (**ret).clone());
        self.emit_inst(
            span,
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                instance: ExternInstance {
                    id: as_slice.fn_ref,
                    instance: as_slice.instance,
                },
            },
        );
        dst
    }

    /// A call of a cast function on `val`: pure, one argument, no context.
    fn emit_extern_cast(&mut self, span: Span, cast: &ExternCast, val: ValueId) -> ValueId {
        let Ty::Fn { ret, .. } = &cast.callee_ty else {
            panic!("a cast's callee_ty is not Fn: {:?}", cast.callee_ty)
        };
        let cast_dst = self.alloc_val();
        self.set_val_type(cast_dst, (**ret).clone());
        self.emit_call(
            span,
            cast_dst,
            Callee::Extern {
                id: cast.fn_ref,
                instance: cast.instance,
                required: cast.required.clone(),
            },
            cast.callee_ty.clone(),
            vec![val],
        );
        cast_dst
    }

    // --- Expression lowering ---

    /// Lower an expression to a value.
    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_before_coercion(expr);
        self.maybe_cast(expr.id(), expr.span(), val)
    }

    fn lower_before_coercion(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        if matches!(self.type_of_id(expr.id()), Ty::Never) {
            self.emit_diverge(expr.span());
        }
        val
    }

    /// The expression just lowered was typed `!`: nothing after it runs.
    /// The block ends here, and what the source wrote after it lands in a
    /// block no jump reaches (RFC-0038).
    fn emit_diverge(&mut self, span: Span) {
        self.emit_inst(span, InstKind::Diverge);
        let unreachable = self.alloc_label();
        self.emit_label(span, unreachable);
    }

    fn lower_expr_inner(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Borrow {
                place,
                mutable,
                span,
                ..
            } => {
                let mutability = if *mutable {
                    Mutability::Mut
                } else {
                    Mutability::Shared
                };
                let place = self.place_or_temporary(place);
                self.emit_ref(*span, place.target, place.path, mutability, place.ty)
            }
            Expr::Literal { id, value, span } => {
                let dst = self.alloc_expr(*id);
                let kind = match value.desugared() {
                    Literal::String(text) => InstKind::ConstStr { dst, text },
                    value => InstKind::Const { dst, value },
                };
                self.emit_inst(*span, kind);
                dst
            }

            Expr::ContextRef {
                id,
                name: qref,
                span,
            } => {
                let inner_ty = self.type_of_id(*id);
                let slot = self.context_slot(*qref);
                let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], inner_ty);
                self.set_origin(dst, ValOrigin::Context(qref.name));
                dst
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => match ref_kind {
                RefKind::ExternParam => {
                    let ty = self.type_of_id(*id);
                    // Try current body's params first. If not found (e.g., inside a
                    // lambda that captured this param as a regular variable), fall
                    // through to local variable lookup.
                    if let Some(param_reg) = self.try_param_slot(name.name) {
                        let dst = self.emit_take(*span, RefTarget::Param(param_reg), vec![], ty);
                        self.set_origin(dst, ValOrigin::ExternParam(name.name));
                        dst
                    } else if self.is_defined(name.name) {
                        let slot = self.var_slot(name.name);
                        let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                        self.set_origin(dst, ValOrigin::Named(name.name));
                        dst
                    } else {
                        // Not found - emit poison (should be caught by typeck).
                        let dst = self.alloc_val();
                        self.set_val_type(dst, ty);
                        self.emit_inst(*span, InstKind::Poison { dst });
                        dst
                    }
                }
                RefKind::Value => {
                    if !self.is_defined(name.name) {
                        // Undefined variable - emit Unit (dead code).
                        // TODO: this should be an error, not silent Unit.
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Unit);
                        return dst;
                    }
                    let ty = self.var_type(name.name);
                    let slot = self.var_slot(name.name);
                    let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                    self.set_origin(dst, ValOrigin::Named(name.name));
                    dst
                }
            },

            Expr::BinaryOp {
                id,
                left,
                op,
                right,
                span,
            } => {
                if let Some(connective) = short_circuit_of(*op) {
                    let l = self.lower_expr(left);
                    let l = self.read_word_through(*span, l);
                    let dst = self.alloc_expr(*id);
                    return self.lower_short_circuit(*span, dst, l, connective, |s| {
                        let r = s.lower_expr(right);
                        s.read_word_through(*span, r)
                    });
                }
                let on_text = holds_text(&self.type_of_id(left.id()));
                if on_text && matches!(op, BinOp::Eq | BinOp::Neq | BinOp::Add) {
                    let l = self.lend_operand(left);
                    let r = self.lend_operand(right);
                    let dst = self.alloc_expr(*id);
                    match op {
                        BinOp::Add => {
                            self.emit_inst(
                                *span,
                                InstKind::StringConcat {
                                    dst,
                                    parts: vec![l, r],
                                },
                            );
                        }
                        BinOp::Eq => {
                            self.emit_inst(*span, InstKind::StringEq { dst, a: l, b: r });
                        }
                        _ => {
                            let eq = self.alloc_val();
                            self.set_val_type(eq, Ty::Bool);
                            self.emit_inst(
                                *span,
                                InstKind::StringEq {
                                    dst: eq,
                                    a: l,
                                    b: r,
                                },
                            );
                            self.emit_inst(
                                *span,
                                InstKind::UnaryOp {
                                    dst,
                                    op: UnaryOp::Not,
                                    operand: eq,
                                },
                            );
                        }
                    }
                    return dst;
                }
                if let Some(CallTarget::Structural(call)) = self.resolution.calls.get(id).cloned() {
                    let l = self.lend_operand(left);
                    let r = self.lend_operand(right);
                    let dst = self.alloc_expr(*id);
                    let StructuralCall {
                        signature: StructuralSignature::Eq,
                        leaves,
                    } = call
                    else {
                        panic!(
                            "an operator offers the structural instance of `eq` alone, got {call:?}"
                        )
                    };
                    let eq = match op {
                        BinOp::Neq => {
                            let v = self.alloc_val();
                            self.set_val_type(v, Ty::Bool);
                            v
                        }
                        _ => dst,
                    };
                    self.emit_inst(
                        *span,
                        InstKind::StructuralEq {
                            dst: eq,
                            a: l,
                            b: r,
                            leaves,
                        },
                    );
                    if eq != dst {
                        self.emit_inst(
                            *span,
                            InstKind::UnaryOp {
                                dst,
                                op: UnaryOp::Not,
                                operand: eq,
                            },
                        );
                    }
                    return dst;
                }
                if let Some(CallTarget::Operator(call)) = self.resolution.calls.get(id).cloned() {
                    let (callee, fn_ty) = (call.callee, call.ty);
                    let l = self.lend_operand(left);
                    let r = self.lend_operand(right);
                    let dst = self.alloc_expr(*id);
                    return match call.signature {
                        OperatorSignature::Eq => {
                            let (call_dst, negate) = match op {
                                BinOp::Neq => {
                                    let v = self.alloc_val();
                                    self.set_val_type(v, Ty::Bool);
                                    (v, true)
                                }
                                _ => (dst, false),
                            };
                            self.emit_call(*span, call_dst, callee, fn_ty, vec![l, r]);
                            if negate {
                                self.emit_inst(
                                    *span,
                                    InstKind::UnaryOp {
                                        dst,
                                        op: UnaryOp::Not,
                                        operand: call_dst,
                                    },
                                );
                            }
                            dst
                        }
                        OperatorSignature::Cmp => {
                            let Ty::Fn { ret, .. } = &fn_ty else {
                                panic!("a shared signature's callee_ty is not Fn: {fn_ty:?}")
                            };
                            let ret = (**ret).clone();
                            let ordering = self.alloc_val();
                            self.set_val_type(ordering, ret.clone());
                            self.emit_call(*span, ordering, callee, fn_ty, vec![l, r]);
                            let zero = self.alloc_val();
                            self.set_val_type(zero, ret);
                            self.emit_inst(
                                *span,
                                InstKind::Const {
                                    dst: zero,
                                    value: Literal::Int(0),
                                },
                            );
                            self.emit_inst(
                                *span,
                                InstKind::BinOp {
                                    dst,
                                    op: *op,
                                    left: ordering,
                                    right: zero,
                                },
                            );
                            dst
                        }
                        OperatorSignature::Add
                        | OperatorSignature::Sub
                        | OperatorSignature::Mul
                        | OperatorSignature::Div
                        | OperatorSignature::Rem
                        | OperatorSignature::Neg => {
                            self.emit_call(*span, dst, callee, fn_ty, vec![l, r]);
                            dst
                        }
                    };
                }
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                let l = self.read_word_through(*span, l);
                let r = self.read_word_through(*span, r);
                let dst = self.alloc_expr(*id);
                self.emit_inst(
                    *span,
                    InstKind::BinOp {
                        dst,
                        op: *op,
                        left: l,
                        right: r,
                    },
                );
                dst
            }

            Expr::UnaryOp {
                id,
                op,
                operand,
                span,
            } => {
                if let Some(CallTarget::Operator(call)) = self.resolution.calls.get(id).cloned() {
                    let o = self.lend_operand(operand);
                    let dst = self.alloc_expr(*id);
                    self.emit_call(*span, dst, call.callee, call.ty, vec![o]);
                    return dst;
                }
                let o = self.lower_expr(operand);
                let o = match op {
                    UnaryOp::Neg => self.read_word_through(*span, o),
                    UnaryOp::Not | UnaryOp::Deref => o,
                };
                let dst = self.alloc_expr(*id);
                let kind = match op {
                    UnaryOp::Deref => InstKind::Take {
                        dst,
                        target: RefTarget::Through(o),
                        path: vec![],
                        taken_out: false,
                    },
                    UnaryOp::Neg | UnaryOp::Not => InstKind::UnaryOp {
                        dst,
                        op: *op,
                        operand: o,
                    },
                };
                self.emit_inst(*span, kind);
                dst
            }

            Expr::Index {
                id,
                callee_id,
                object,
                index,
                span,
            } => self.lower_index(*id, *callee_id, object, index, *span),

            Expr::FieldAccess {
                id,
                object,
                field,
                span,
            } => {
                let field_ty = self.type_of_id(*id);
                let Projected { base, fields } = projected(expr);
                match self.base_target(base) {
                    Some(target) => {
                        let path = self::fields(&fields);
                        let dst = self.emit_take(*span, target, path.clone(), field_ty);
                        self.set_origin(dst, ValOrigin::RefField(target, path));
                        dst
                    }
                    None => {
                        let obj = self.lower_expr(object);
                        let dst = self.alloc_val();
                        self.set_val_type(dst, field_ty);
                        self.set_origin(dst, ValOrigin::Field(obj, *field));
                        self.emit_inst(
                            *span,
                            InstKind::FieldGet {
                                dst,
                                object: obj,
                                field: *field,
                                rest: vec![],
                            },
                        );
                        dst
                    }
                }
            }

            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => self.lower_func_call(func, args, None, *id, *span),

            Expr::MethodCall {
                id,
                callee_id,
                receiver,
                name,
                args,
                span,
                ..
            } => self.lower_method_call(*callee_id, receiver, *name, args, *id, *span),

            Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
                // Desugar: `a | f(b, c)` -> `f(a, b, c)`, `a | f` -> `f(a)`
                match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.lower_func_call(func, args, Some(left), *id, *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    } => self.lower_func_call(right, &[], Some(left), *id, *span),
                    _ => {
                        // Fallback: evaluate both sides, call as indirect.
                        let l = self.lower_expr(left);
                        let r = self.lower_expr(right);
                        let fn_ty = self
                            .body
                            .val_types
                            .get(&r)
                            .expect("indirect callee must have val_type")
                            .clone();
                        let dst = self.alloc_typed(expr.id());
                        self.emit_call(*span, dst, Callee::Indirect(r), fn_ty, vec![l]);
                        dst
                    }
                }
            }

            Expr::Lambda {
                id,
                params,
                body,
                span,
            } => {
                let captured: Vec<CapturedName> = self
                    .resolution
                    .lambda_captures
                    .get(id)
                    .expect("the checker records a capture list for every lambda it checks")
                    .clone();

                // A closure owns what it captures (RFC-0018): each capture is
                // taken from the storage the name binds, at that storage's own
                // type, and not at the `&T` the name reads as inside an
                // enclosing closure body.
                let capture_regs: Vec<ValueId> = captured
                    .iter()
                    .map(|capture| {
                        let name = capture.name;
                        let taken = if let Some(param_reg) = self.try_param_slot(name) {
                            let ty = self.slot_type(param_reg);
                            let dst =
                                self.emit_take(*span, RefTarget::Param(param_reg), vec![], ty);
                            self.set_origin(dst, ValOrigin::ExternParam(name));
                            dst
                        } else {
                            let slot = self.var_slot(name);
                            let ty = self.slot_type(slot);
                            let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                            self.set_origin(dst, ValOrigin::Named(name));
                            dst
                        };
                        taken
                    })
                    .collect();
                // Create closure body.
                let closure_label = self.alloc_closure_label();

                // Build the closure body MIR in a sub-lowerer.
                let mut sub_body = MirBody::new();

                // Captures become the first registers. The closure owns the
                // value handed to `MakeClosure`, and `machine::bind_captures`
                // binds each of these registers to a reference into it,
                // whatever that value is — which is why every capture
                // register is a reference and the body reads one level
                // through it where it wants the value.
                let mut closure_capture_regs = Vec::new();
                let mut capture_tys = Vec::new();
                for (capture, capture_reg) in captured.iter().zip(capture_regs.iter()) {
                    let reg = sub_body.val_factory.next();
                    closure_capture_regs.push(reg);
                    sub_body
                        .debug
                        .val_origins
                        .insert(reg, ValOrigin::Named(capture.name));
                    let given = self
                        .body
                        .val_types
                        .get(capture_reg)
                        .expect("a capture register is typed where it is taken")
                        .clone();
                    let cap_ty = Ty::Ref(
                        Mutability::Shared,
                        Box::new(TypeArg::uniform(given.clone())),
                    );
                    sub_body.val_types.insert(reg, cap_ty.clone());
                    capture_tys.push(CaptureRegister { given, cap_ty });
                }

                // Params follow captures.
                let mut closure_param_regs = Vec::new();
                for p in params.iter() {
                    let reg = sub_body.val_factory.next();
                    closure_param_regs.push(reg);
                    let ty = self.type_of_id(p.id);
                    sub_body.val_types.insert(reg, ty);
                }

                // We need to lower the body in context of the sub-body.
                // Swap state.
                let saved_body = std::mem::replace(&mut self.body, sub_body);
                let saved_scopes = std::mem::replace(&mut self.scopes, vec![FxHashMap::default()]);
                let saved_order_slot = self.order_slot;
                let saved_anyorder = self.anyorder;
                let saved_context_slots = std::mem::take(&mut self.context_slots);
                let saved_taken_out = std::mem::take(&mut self.taken_out);
                let lambda_effect = self.type_of_id(*id).effect().unwrap_or(Effect::OPAQUE);
                self.enter_body_order(lambda_effect, *span);
                self.enter_contexts(acvus_ast::direct_expr_context_refs(body), *span);

                // Captures and params are the closure body's first bindings.
                for ((capture, capture_reg), register) in captured
                    .iter()
                    .zip(closure_capture_regs.iter())
                    .zip(capture_tys)
                {
                    let (bound, bound_ty) = match capture.read {
                        CaptureRead::Word => {
                            let word = self.emit_take(
                                *span,
                                RefTarget::Through(*capture_reg),
                                vec![],
                                register.given.clone(),
                            );
                            (word, register.given)
                        }
                        CaptureRead::Lent => (*capture_reg, register.cap_ty),
                    };
                    let slot = self.define_var(capture.name, bound_ty);
                    self.emit_assign(*span, RefTarget::Var(slot), vec![], bound);
                }
                for (p, param_reg) in params.iter().zip(closure_param_regs.iter()) {
                    let ty = self.type_of_id(p.id);
                    let slot = self.define_var(p.name, ty);
                    self.emit_assign(p.span, RefTarget::Var(slot), vec![], *param_reg);
                }

                // lower_expr calls maybe_cast(body.span(), val) which will
                // pick up any lambda return coercion registered by the typechecker.
                let result_reg = self.lower_expr(body);
                // Capture the actual return type (may differ from type_map if Cast was inserted).
                let actual_ret_ty = self
                    .body
                    .val_types
                    .get(&result_reg)
                    .cloned()
                    .unwrap_or(Ty::error());
                self.emit_return(*span, result_reg);

                let mut closure_body_mir = std::mem::replace(&mut self.body, saved_body);
                self.scopes = saved_scopes;
                self.order_slot = saved_order_slot;
                self.anyorder = saved_anyorder;
                self.context_slots = saved_context_slots;
                self.taken_out = saved_taken_out;

                closure_body_mir.captures = captured
                    .iter()
                    .map(|capture| capture.name)
                    .zip(closure_capture_regs)
                    .collect();
                closure_body_mir.params = params
                    .iter()
                    .map(|p| p.name)
                    .zip(closure_param_regs)
                    .collect();
                closure_body_mir.task = self
                    .type_of_id(*id)
                    .effect()
                    .map_or(Task::Sync, |effect| effect.task);
                self.closures.insert(closure_label, closure_body_mir);

                // Allocate dst with Fn type. If a return-site Cast was inserted,
                // update the Fn's ret to match the actual (cast) return type.
                let dst = self.alloc_val();
                let mut fn_ty = self.type_of_id(*id);
                if let Ty::Fn { ref mut ret, .. } = fn_ty {
                    **ret = actual_ret_ty;
                }
                self.set_val_type(dst, fn_ty);
                self.emit_inst(
                    *span,
                    InstKind::MakeClosure {
                        dst,
                        body: closure_label,
                        captures: capture_regs,
                    },
                );
                dst
            }

            Expr::Paren { inner, .. } => self.lower_expr(inner),

            Expr::Cast {
                id,
                expr,
                target,
                span,
                ..
            } => {
                let src = self.lower_expr(expr);
                let dst = self.alloc_expr(*id);
                let kind = match CastTy::of_name(self.interner.resolve(*target)) {
                    Some(to) => InstKind::Cast { dst, src, to },
                    None => InstKind::Poison { dst },
                };
                self.emit_inst(*span, kind);
                dst
            }

            Expr::Try { id, inner, span } => self.lower_try(*id, inner, *span),

            // No scope-stack drop belongs here: `optimize::drop_insertion`
            // owns the return edge, as it does the one `?` leaves by.
            Expr::Return { id, value, span } => {
                let val = self.lower_expr(value);
                self.emit_return(*span, val);
                let unreachable = self.alloc_label();
                self.emit_label(*span, unreachable);
                self.alloc_expr(*id)
            }

            Expr::List {
                id,
                head,
                rest: _,
                tail,
                span,
            } => {
                let elements: Vec<ValueId> = head
                    .iter()
                    .chain(tail.iter())
                    .map(|e| self.lower_expr(e))
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(*span, InstKind::MakeArray { dst, elements });
                dst
            }

            Expr::Object { id, fields, span } => {
                let field_regs = fields
                    .iter()
                    .map(|ObjectExprField { key, value, .. }| {
                        let r = self.lower_expr(value);
                        (*key, r)
                    })
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeObject {
                        dst,
                        fields: field_regs,
                    },
                );
                dst
            }

            Expr::Tuple { id, elements, span } => {
                let elem_vals: Vec<ValueId> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.lower_expr(e),
                        TupleElem::Wildcard(s) => {
                            let dst = self.alloc_val();
                            self.set_val_type(dst, Ty::Unit);
                            self.emit_inst(
                                *s,
                                InstKind::Const {
                                    dst,
                                    value: Literal::Bool(false),
                                },
                            );
                            dst
                        }
                    })
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeTuple {
                        dst,
                        elements: elem_vals,
                    },
                );
                dst
            }

            Expr::Group { elements, span, .. } => {
                // Should not appear outside lambda params. Lower last element.
                let Some(last) = elements.last() else {
                    let dst = self.alloc_val();
                    self.set_val_type(dst, Ty::Unit);
                    self.emit_inst(
                        *span,
                        InstKind::Const {
                            dst,
                            value: Literal::Bool(false),
                        },
                    );
                    return dst;
                };
                self.lower_expr(last)
            }

            Expr::Variant {
                id,
                tag,
                payload,
                span,
                ..
            } => {
                let payload_val = payload.as_ref().map(|e| self.lower_expr(e));
                let dst = self.alloc_expr(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeVariant {
                        dst,
                        tag: *tag,
                        payload: payload_val,
                    },
                );
                dst
            }

            Expr::Block { stmts, tail, .. } => {
                self.push_scope();
                for stmt in stmts {
                    self.lower_stmt(stmt);
                }
                let val = self.lower_expr(tail);
                self.pop_scope();
                val
            }

            Expr::If {
                id,
                cond,
                then_body,
                then_tail,
                else_branch,
                span,
            } => self.lower_if_expr(*id, cond, then_body, then_tail, else_branch, *span),

            Expr::Match {
                id,
                scrutinee,
                arms,
                span,
            } => self.lower_match_expr(*id, scrutinee, arms, *span),
            Expr::IfLet {
                id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                span,
            } => self.lower_if_let_expr(
                *id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                *span,
            ),
        }
    }

    /// A store into a place. An element write is the one place the IR has no
    /// `Assign` for: `PathSeg::Index` carries a constant step, and `a[i]`'s
    /// index is a value, so the element is written through the container's
    /// mutable slice (RFC-0047). Every other place is the storage its root
    /// names under a path of `PathSeg::Field` steps.
    fn lower_store(&mut self, place: &acvus_ast::Place, value_expr: &Expr, span: Span) {
        let value = self.lower_expr(value_expr);
        if let acvus_ast::Place::Base(acvus_ast::PlaceBase::Element {
            id,
            callee_id,
            container,
            index,
            span: index_span,
        }) = place
        {
            let taken = self.take_slice(
                container.expr(),
                *callee_id,
                *id,
                self.index_access(*id).mutability,
                *index_span,
            );
            let index = self.lower_expr(index);
            self.emit_inst(
                span,
                InstKind::IndexSet {
                    slice: taken,
                    index,
                    value,
                },
            );
            return;
        }
        let place = self.store_target(place);
        self.emit_assign(span, place.target, place.path, value);
    }

    fn lower_deref_store(&mut self, target: &Expr, value_expr: &Expr, span: Span) {
        let val = self.lower_expr(value_expr);
        let reference = self.lower_expr(target);
        self.emit_assign(span, RefTarget::Through(reference), vec![], val);
    }

    /// An intrinsic call over its arguments as every call lowers them.
    fn emit_intrinsic(
        &mut self,
        intrinsic: crate::typeck::Intrinsic,
        call: CallArgs,
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        assert!(
            self.taken_out.len() == call.taken_from,
            "an intrinsic takes no argument through a conversion"
        );
        let dst = self.alloc_expr(call_id);
        let kind = match (intrinsic, call.values.as_slice()) {
            (crate::typeck::Intrinsic::StringClone, &[src]) => InstKind::StringClone { dst, src },
            (crate::typeck::Intrinsic::StringClone, values) => panic!(
                "`clone` is checked at its signature's one parameter, and got {} arguments",
                values.len()
            ),
        };
        self.emit_inst(call_span, kind);
        dst
    }

    fn emit_structural(
        &mut self,
        structural: StructuralCall,
        call: CallArgs,
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        assert!(
            self.taken_out.len() == call.taken_from,
            "the structural instance takes no argument through a conversion"
        );
        let dst = self.alloc_expr(call_id);
        let StructuralCall { signature, leaves } = structural;
        let kind = match (signature, call.values.as_slice()) {
            (StructuralSignature::Eq, &[a, b]) => InstKind::StructuralEq { dst, a, b, leaves },
            (StructuralSignature::Clone, &[src]) => InstKind::StructuralClone { dst, src, leaves },
            (signature, values) => panic!(
                "{signature:?} is checked at its signature's parameters, and got {} arguments",
                values.len()
            ),
        };
        self.emit_inst(call_span, kind);
        dst
    }

    /// The arguments of a call: a `&place` argument is lent, any other is
    /// a value.
    fn lower_call_args<'e, I>(&mut self, args: I) -> CallArgs
    where
        I: Iterator<Item = &'e Expr>,
    {
        let mut call = CallArgs {
            values: Vec::new(),
            taken_from: self.taken_out.len(),
        };
        for a in args {
            let value = match a {
                Expr::Borrow {
                    id,
                    place,
                    mutable,
                    span,
                } => {
                    let mutability = if *mutable {
                        Mutability::Mut
                    } else {
                        Mutability::Shared
                    };
                    let lent = self.lent(*id, *span, place, mutability);
                    self.lend_place(lent)
                }
                other => self.lower_expr(other),
            };
            call.values.push(value);
        }
        call
    }

    fn lent(&mut self, id: AstId, span: Span, place: &Expr, mutability: Mutability) -> Lent {
        let from = match self.place_base(projected(place).base.id()) {
            PlaceBase::Storage(_)
            | PlaceBase::ThroughReferenceIn(_)
            | PlaceBase::ThroughReference
            | PlaceBase::Element(_) => LentFrom::Place,
            PlaceBase::Temporary => LentFrom::Temporary,
        };
        Lent {
            id,
            span,
            place: self.place_or_temporary(place),
            from,
            mutability,
        }
    }

    /// The reference a lent argument passes. A conversion the checker
    /// answered through the reference takes the place's value out of its
    /// slot, casts it into the callee's representation in a temporary, and
    /// lends the temporary, so no slot holds a value of another type. A
    /// place the program names is taken out, marked so, until the marked
    /// store that restores it after the call, and the move check refuses
    /// every other touch of it in between (RFC-0041); a shared lend of it
    /// through the same cast inside the call lends the same temporary. A
    /// conversion of the reference itself casts the reference.
    fn lend_place(&mut self, lent: Lent) -> ValueId {
        let Lent {
            id,
            span,
            place,
            from,
            mutability,
        } = lent;
        let coercion = self.coercion_lookup.get(&id).cloned();
        if let Some(CastKind::ThroughRef {
            mutability: Mutability::Shared,
            cast,
            ..
        }) = &coercion
            && let Some(taken) = self
                .taken_out
                .iter()
                .find(|taken| taken.shared_by(&place, cast))
        {
            let lent = taken.lent.clone();
            return self.emit_ref(span, lent.target, lent.path, mutability, lent.ty);
        }
        match coercion {
            Some(CastKind::ThroughRef {
                mutability: converted_as,
                cast,
                back,
            }) => {
                let value = self.alloc_val();
                self.set_val_type(value, place.ty.clone());
                self.emit_inst(
                    span,
                    InstKind::Take {
                        dst: value,
                        target: place.target,
                        path: place.path.clone(),
                        taken_out: matches!(from, LentFrom::Place),
                    },
                );
                let converted = self.emit_extern_cast(span, &cast, value);
                let ty = self.slot_type(converted);
                let lent = Place {
                    target: self.temporary(span, converted, ty.clone()),
                    path: Vec::new(),
                    ty,
                };
                let reference =
                    self.emit_ref(span, lent.target, Vec::new(), mutability, lent.ty.clone());
                match from {
                    LentFrom::Place => self.taken_out.push(PlaceRestore {
                        span,
                        place,
                        lent,
                        mutability: converted_as,
                        cast,
                        back,
                    }),
                    LentFrom::Temporary => {}
                }
                reference
            }
            Some(CastKind::Extern(cast)) => {
                let reference = self.emit_ref(span, place.target, place.path, mutability, place.ty);
                self.emit_extern_cast(span, &cast, reference)
            }
            Some(CastKind::Slice {
                mutability: sliced,
                as_slice,
            }) => {
                let reference = self.emit_ref(span, place.target, place.path, mutability, place.ty);
                self.emit_as_slice(span, reference, sliced, &as_slice)
            }
            Some(CastKind::Str { as_str }) => {
                let reference = self.emit_ref(span, place.target, place.path, mutability, place.ty);
                self.emit_as_slice(span, reference, Mutability::Shared, &as_str)
            }
            Some(CastKind::Reborrow { .. }) => {
                self.emit_ref(span, place.target, place.path, Mutability::Shared, place.ty)
            }
            None => self.emit_ref(span, place.target, place.path, mutability, place.ty),
        }
    }

    /// A call with its arguments; every place taken out for the callee is
    /// cast back from the temporary it was lent as and assigned after it.
    fn emit_call_with(
        &mut self,
        span: Span,
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: CallArgs,
    ) {
        self.emit_call(span, dst, callee, callee_ty, args.values);
        for restore in self.taken_out.split_off(args.taken_from) {
            let PlaceRestore {
                span,
                place,
                lent,
                back,
                ..
            } = restore;
            let value = self.emit_take(span, lent.target, lent.path, lent.ty);
            let restored = self.emit_extern_cast(span, &back, value);
            self.emit_inst(
                span,
                InstKind::Assign {
                    target: place.target,
                    path: place.path,
                    value: restored,
                    restores: true,
                },
            );
        }
    }

    /// A call lends the closure (RFC-0018). A slot already holding a
    /// `&Fn` — a capture — is reborrowed, since no `&&Fn` exists
    /// (RFC-0029).
    fn lent_closure(&mut self, name: Astr, span: Span) -> (ValueId, Ty) {
        let slot = self.var_slot(name);
        let (reg, ty) = match self.var_type(name) {
            Ty::Ref(_, lent) => {
                let reference = Ty::Ref(Mutability::Shared, lent.clone());
                let reg = self.emit_take(span, RefTarget::Var(slot), vec![], reference);
                (reg, lent.ty().into_owned())
            }
            owned => {
                let reg = self.emit_ref(
                    span,
                    RefTarget::Var(slot),
                    vec![],
                    Mutability::Shared,
                    owned.clone(),
                );
                (reg, owned)
            }
        };
        self.set_origin(reg, ValOrigin::Named(name));
        (reg, ty)
    }

    /// `recv.f(args)`: `f(recv', args)`, the receiver lent when `f`'s first
    /// parameter is a reference (RFC-0030).
    fn lower_method_call(
        &mut self,
        callee_id: AstId,
        receiver: &Expr,
        name: Astr,
        args: &[Expr],
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        let callee_ty = self.type_of_id(callee_id);
        let taken_from = self.taken_out.len();
        let first = self.receiver(receiver);
        let mut call = self.lower_call_args(args.iter());
        call.values.insert(0, first);
        call.taken_from = taken_from;
        let target = self.resolution.calls.get(&callee_id).cloned();
        if let Some(CallTarget::Intrinsic(intrinsic)) = target {
            return self.emit_intrinsic(intrinsic, call, call_id, call_span);
        }
        if let Some(CallTarget::Structural(structural)) = target {
            return self.emit_structural(structural, call, call_id, call_span);
        }
        let dst = self.alloc_typed(call_id);
        match target {
            Some(CallTarget::Declared(callee)) => {
                self.set_origin(dst, ValOrigin::Call(callee.id().name));
                self.emit_call_with(call_span, dst, callee, callee_ty, call);
            }
            Some(CallTarget::Binding) => {
                let (closure_reg, closure_ty) = self.lent_closure(name, call_span);
                self.emit_call_with(
                    call_span,
                    dst,
                    Callee::Indirect(closure_reg),
                    closure_ty,
                    call,
                );
            }
            Some(
                other @ (CallTarget::Intrinsic(_)
                | CallTarget::StructuralVariant
                | CallTarget::Operator(_)
                | CallTarget::Structural(_)),
            ) => panic!("a method call is checked as a named call, never {other:?}"),
            None => {
                self.emit_inst(call_span, InstKind::Poison { dst });
            }
        }
        dst
    }

    fn passing(&self, lent: &Expr) -> Passing {
        *self
            .resolution
            .passing
            .get(&lent.id())
            .expect("the checker records how every receiver and operand is passed")
    }

    /// A receiver as the checker decided it reaches its call.
    fn receiver(&mut self, receiver: &Expr) -> ValueId {
        match self.passing(receiver) {
            Passing::Value | Passing::AsIs => self.lower_expr(receiver),
            Passing::Lent(mutability) => {
                let lent = self.lent(receiver.id(), receiver.span(), receiver, mutability);
                self.lend_place(lent)
            }
        }
    }

    fn lower_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Box<Expr>>,
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        let written: Vec<&Expr> = pipe_left
            .map(|left| &**left)
            .into_iter()
            .chain(args)
            .collect();
        let target = self.resolution.calls.get(&func.id()).cloned();
        if let Some(CallTarget::Intrinsic(intrinsic)) = target {
            let call = self.lower_call_args(written.iter().copied());
            return self.emit_intrinsic(intrinsic, call, call_id, call_span);
        }
        if let Some(CallTarget::Structural(structural)) = target {
            let call = self.lower_call_args(written.iter().copied());
            return self.emit_structural(structural, call, call_id, call_span);
        }
        if let Some(CallTarget::StructuralVariant) = target
            && let [payload] = written.as_slice()
        {
            let Expr::Ident { name, .. } = func else {
                unreachable!("a structural variant call is a qualified name");
            };
            let payload_val = self.lower_expr(payload);
            let dst = self.alloc_expr(call_id);
            self.emit_inst(
                call_span,
                InstKind::MakeVariant {
                    dst,
                    tag: name.name,
                    payload: Some(payload_val),
                },
            );
            return dst;
        }
        let call = self.lower_call_args(written.iter().copied());
        let dst = self.alloc_typed(call_id);

        // Named function call (Ident).
        if let Expr::Ident {
            name,
            ref_kind,
            span: ident_span,
            ..
        } = func
        {
            match ref_kind {
                // fn_name(args) - named call
                RefKind::Value => {
                    self.set_origin(dst, ValOrigin::Call(name.name));

                    match target {
                        Some(CallTarget::Declared(callee)) => {
                            let callee_ty = self.type_of_id(func.id());
                            self.emit_call_with(call_span, dst, callee, callee_ty, call);
                        }
                        Some(CallTarget::Binding) => {
                            let (closure_reg, closure_ty) =
                                self.lent_closure(name.name, *ident_span);
                            self.emit_call_with(
                                call_span,
                                dst,
                                Callee::Indirect(closure_reg),
                                closure_ty,
                                call,
                            );
                        }
                        Some(
                            other @ (CallTarget::Intrinsic(_)
                            | CallTarget::StructuralVariant
                            | CallTarget::Operator(_)
                            | CallTarget::Structural(_)),
                        ) => panic!("a call of a name lowers as its target above, never {other:?}"),
                        None => {
                            self.emit_inst(call_span, InstKind::Poison { dst });
                        }
                    }
                    return dst;
                }
                _ => {}
            }
        }

        // Expression call (e.g., (|x| -> x)(42), or complex pipe)
        self.set_origin(dst, ValOrigin::Call(self.interner.intern("<closure>")));
        let func_reg = self.lower_expr(func);
        let fn_ty = self
            .body
            .val_types
            .get(&func_reg)
            .expect("indirect callee must have val_type")
            .clone();
        self.emit_call_with(call_span, dst, Callee::Indirect(func_reg), fn_ty, call);
        dst
    }

    // --- Patterns (RFC-0024) ---

    fn payload_type(&self, inner: &Ty, tag: Astr) -> Ty {
        match inner {
            Ty::Option(payload) => payload.as_ref().clone(),
            Ty::Result(ok, err) => {
                if self.interner.resolve(tag) == "Ok" {
                    ok.as_ref().clone()
                } else {
                    err.as_ref().clone()
                }
            }
            Ty::Enum { variants, .. } => variants
                .get(&tag)
                .cloned()
                .flatten()
                .map(|t| *t)
                .unwrap_or(Ty::Unit),
            _ => Ty::error(),
        }
    }

    fn pattern_source<'p, P>(&mut self, source: &Expr, patterns: P) -> PatSrc
    where
        P: IntoIterator<Item = &'p Pattern>,
    {
        let Projected { base, fields } = projected(source);
        if let PlaceBase::Storage(storage) = self.place_base(base.id()) {
            return PatSrc::Placed(Placed::Place {
                target: self.storage(storage),
                path: self::fields(&fields),
                ty: self.type_of_id(source.id()),
            });
        }
        let value = self.lower_expr(source);
        let ty = self
            .body
            .val_types
            .get(&value)
            .cloned()
            .expect("a lowered expression has a type");
        let mode = *self
            .resolution
            .pattern_modes
            .get(&source.id())
            .expect("the checker records how every pattern source is read");
        if mode == MatchMode::Through {
            let Ty::Ref(_, referent) = ty else {
                panic!("the checker reads a source through a reference only where it is one")
            };
            return PatSrc::Placed(Placed::Through {
                reference: value,
                ty: referent.into_ty(),
            });
        }
        if patterns.into_iter().any(test_reads_a_part) {
            let ty = self.type_of_id(source.id());
            return PatSrc::Placed(self.spill_pattern_source(source.span(), value, ty));
        }
        PatSrc::Value { value, ty }
    }

    fn spill_pattern_source(&mut self, span: Span, value: ValueId, ty: Ty) -> Placed {
        let slot = self.body.val_factory.next();
        self.set_origin(slot, ValOrigin::Named(self.interner.intern("$source")));
        self.set_val_type(slot, ty.clone());
        self.emit_assign(span, RefTarget::Var(slot), vec![], value);
        Placed::Place {
            target: RefTarget::Var(slot),
            path: vec![],
            ty,
        }
    }

    fn placed(&mut self, src: &PatSrc, span: Span) -> Placed {
        match src {
            PatSrc::Placed(placed) => placed.clone(),
            PatSrc::Value { value, ty } => self.spill_pattern_source(span, *value, ty.clone()),
        }
    }

    fn pattern_parts<'p>(&self, pattern: &'p Pattern, ty: &Ty) -> Vec<PatternPart<'p>> {
        match pattern {
            Pattern::List { head, tail, .. } => {
                let Ty::Array(elem, len) = ty else {
                    panic!("type checking gives a list pattern an array of known length")
                };
                let tail_start = len.get() - tail.len();
                head.iter()
                    .enumerate()
                    .chain(
                        tail.iter()
                            .enumerate()
                            .map(|(i, pattern)| (tail_start + i, pattern)),
                    )
                    .map(|(index, pattern)| PatternPart {
                        seg: PathSeg::Index(index),
                        ty: elem.as_ref().clone(),
                        pattern,
                    })
                    .collect()
            }
            Pattern::Object { fields, .. } => {
                let Ty::Object(field_tys) = ty else {
                    panic!("type checking gives an object pattern an object source")
                };
                fields
                    .iter()
                    .map(|ObjectPatternField { key, pattern, .. }| PatternPart {
                        seg: PathSeg::Field(*key),
                        ty: field_tys.get(key).cloned().unwrap_or_else(|| {
                            panic!("type checking settles every key an object pattern names")
                        }),
                        pattern,
                    })
                    .collect()
            }
            Pattern::Tuple { elements, .. } => {
                let Ty::Tuple(elem_tys) = ty else {
                    panic!("type checking gives a tuple pattern a tuple source")
                };
                elements
                    .iter()
                    .zip(elem_tys)
                    .enumerate()
                    .filter_map(|(index, (elem, ty))| match elem {
                        TuplePatternElem::Pattern(pattern) => Some(PatternPart {
                            seg: PathSeg::Index(index),
                            ty: ty.clone(),
                            pattern,
                        }),
                        TuplePatternElem::Wildcard(_) => None,
                    })
                    .collect()
            }
            Pattern::Variant { .. }
            | Pattern::Binding { .. }
            | Pattern::ContextBind { .. }
            | Pattern::Literal { .. }
            | Pattern::Wildcard { .. } => Vec::new(),
        }
    }

    fn project(&mut self, src: &Placed, seg: PathSeg, ty: Ty, span: Span) -> Placed {
        match src {
            Placed::Place { target, path, .. } => Placed::Place {
                target: *target,
                path: path.iter().copied().chain([seg]).collect(),
                ty,
            },
            Placed::Through { reference, .. } => Placed::Through {
                reference: self.emit_ref(
                    span,
                    RefTarget::Through(*reference),
                    vec![seg],
                    Mutability::Shared,
                    ty.clone(),
                ),
                ty,
            },
        }
    }

    fn project_parts<'p>(
        &mut self,
        pattern: &'p Pattern,
        src: &PatSrc,
        span: Span,
    ) -> Vec<(PatSrc, &'p Pattern)> {
        let placed = self.placed(src, span);
        self.pattern_parts(pattern, placed.ty())
            .into_iter()
            .map(|part| {
                let projected = self.project(&placed, part.seg, part.ty, span);
                (PatSrc::Placed(projected), part.pattern)
            })
            .collect()
    }

    fn project_payload(&mut self, src: &PatSrc, tag: Astr, span: Span) -> PatSrc {
        let ty = self.payload_type(src.ty(), tag);
        match src {
            PatSrc::Placed(placed) => {
                PatSrc::Placed(self.project(placed, PathSeg::Payload, ty, span))
            }
            PatSrc::Value { value, .. } => {
                let payload = self.alloc_val();
                self.set_val_type(payload, ty.clone());
                self.emit_inst(
                    span,
                    InstKind::UnwrapVariant {
                        dst: payload,
                        src: *value,
                    },
                );
                PatSrc::Value { value: payload, ty }
            }
        }
    }

    fn read_leaf(&mut self, src: &PatSrc, span: Span) -> ValueId {
        match src {
            PatSrc::Placed(Placed::Place { target, path, ty }) => {
                self.emit_take(span, *target, path.clone(), ty.clone())
            }
            PatSrc::Placed(Placed::Through { reference, .. }) => {
                self.read_word_through(span, *reference)
            }
            PatSrc::Value { value, .. } => *value,
        }
    }

    fn tag_reference(&mut self, src: &PatSrc, span: Span) -> ValueId {
        match src {
            PatSrc::Placed(Placed::Place { target, path, ty }) => {
                self.emit_ref(span, *target, path.clone(), Mutability::Shared, ty.clone())
            }
            PatSrc::Placed(Placed::Through { reference, .. }) => *reference,
            PatSrc::Value { value, .. } => *value,
        }
    }

    fn bind_part(&mut self, src: &PatSrc, name: Astr, span: Span) {
        let (value, ty) = match src {
            PatSrc::Placed(Placed::Through { reference, ty }) => (
                *reference,
                Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(ty.clone()))),
            ),
            PatSrc::Placed(Placed::Place { ty, .. }) | PatSrc::Value { ty, .. } => {
                (self.read_leaf(src, span), ty.clone())
            }
        };
        self.set_origin(value, ValOrigin::Named(name));
        let slot = self.define_var(name, ty);
        self.emit_assign(span, RefTarget::Var(slot), vec![], value);
    }

    /// The checker refuses a context bound through a reference
    /// (`ReferenceInData`), so that source lowers to poison.
    fn bind_context(&mut self, src: &PatSrc, context: QualifiedRef, span: Span) {
        match src {
            PatSrc::Placed(Placed::Through { .. }) => self.emit_poison(span),
            PatSrc::Placed(Placed::Place { .. }) | PatSrc::Value { .. } => {
                let value = self.read_leaf(src, span);
                let slot = self.context_slot(context);
                self.emit_assign(span, RefTarget::Var(slot), vec![], value);
            }
        }
    }

    fn emit_poison(&mut self, span: Span) {
        let dst = self.alloc_val();
        self.emit_inst(span, InstKind::Poison { dst });
    }

    fn lower_pattern_test(&mut self, pattern: &Pattern, src: &PatSrc, span: Span) -> ValueId {
        match pattern {
            Pattern::ContextBind { .. } | Pattern::Binding { .. } | Pattern::Wildcard { .. } => {
                self.emit_const_bool(span, true)
            }
            Pattern::Literal { value, .. } => {
                let leaf = self.read_leaf(src, span);
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestLiteral {
                        dst,
                        src: leaf,
                        value: value.desugared(),
                    },
                );
                dst
            }
            Pattern::List { .. } | Pattern::Object { .. } | Pattern::Tuple { .. } => {
                let mut all_ok = self.emit_const_bool(span, true);
                for (part, part_pattern) in self.project_parts(pattern, src, span) {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        s.lower_pattern_test(part_pattern, &part, span)
                    });
                }
                all_ok
            }
            Pattern::Variant { tag, payload, .. } => {
                let reference = self.tag_reference(src, span);
                let tag_ok = self.alloc_val();
                self.set_val_type(tag_ok, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestVariant {
                        dst: tag_ok,
                        src: reference,
                        tag: *tag,
                    },
                );
                let Some(payload) = payload.as_deref() else {
                    return tag_ok;
                };
                if pattern_is_irrefutable(payload) {
                    return tag_ok;
                }
                let labels = DiamondLabels {
                    then_label: self.alloc_label(),
                    else_label: self.alloc_label(),
                    join: self.alloc_label(),
                };
                let pending = self.open_diamond(span, tag_ok, labels);
                self.emit_label(span, labels.then_label);
                let payload_src = self.project_payload(src, *tag, span);
                let payload_ok = self.lower_pattern_test(payload, &payload_src, span);
                let result = self.emit_fail_merge(span, payload_ok, labels);
                self.close_diamond(pending);
                result
            }
        }
    }

    fn lower_pattern_bind(&mut self, pattern: &Pattern, src: &PatSrc, span: Span) {
        match pattern {
            Pattern::Wildcard { .. } | Pattern::Literal { .. } => {}
            Pattern::Binding {
                name,
                ref_kind: RefKind::Value,
                ..
            } => self.bind_part(src, *name, span),
            // The checker refuses an assignment to an extern parameter
            // (`ExternParamAssign`).
            Pattern::Binding {
                ref_kind: RefKind::ExternParam,
                ..
            } => self.emit_poison(span),
            Pattern::ContextBind { name, .. } => self.bind_context(src, *name, span),
            Pattern::List { .. } | Pattern::Object { .. } | Pattern::Tuple { .. } => {
                for (part, part_pattern) in self.project_parts(pattern, src, span) {
                    self.lower_pattern_bind(part_pattern, &part, span);
                }
            }
            Pattern::Variant { tag, payload, .. } => {
                let Some(payload) = payload.as_deref() else {
                    return;
                };
                let payload_src = self.project_payload(src, *tag, span);
                self.lower_pattern_bind(payload, &payload_src, span);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower(interner: &Interner, source: &str) -> MirModule {
        lower_with(interner, source, &FxHashMap::default())
    }

    fn lower_with(interner: &Interner, source: &str, context: &FxHashMap<Astr, Ty>) -> MirModule {
        let ctx: Vec<(&str, Ty)> = context
            .iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        let module = crate::test::compile_template(interner, source, &ctx).expect("compile failed");
        module
    }

    /// Each iteration appends onto the one accumulator; nothing rebuilds
    /// it, so a loop's cost is the text it writes (RFC-0071).
    #[test]
    fn a_loop_body_appends_once_per_piece_and_copies_nothing() {
        let interner = Interner::new();
        let module = lower(&interner, "% for i in 0..2\n- {{ \"x\" }}\n% end\n");
        let printed = crate::printer::dump(&interner, &module);
        let body = printed
            .split("L1(")
            .nth(1)
            .expect("the loop body block")
            .split("L2:")
            .next()
            .expect("the body ends at the exit block");
        assert_eq!(body.matches("append ").count(), 3, "{printed}");
        assert!(!printed.contains("string_concat"), "{printed}");
    }

    #[test]
    fn lower_text_node() {
        let interner = Interner::new();
        let module = lower(&interner, "hello world");
        // Template: the text as a `&str` constant, the concat, the return.
        let has_text =
            module.main.insts.iter().any(
                |i| matches!(&i.kind, InstKind::ConstStr { text, .. } if text == "hello world"),
            );
        let has_return = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::Return { .. }));
        assert!(has_text);
        assert!(has_return);
    }

    #[test]
    fn lower_string_emit() {
        let interner = Interner::new();
        let module = lower(&interner, r#"{{ "hello" }}"#);
        let has_text = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::ConstStr { text, .. } if text == "hello"));
        let appends = module
            .main
            .insts
            .iter()
            .filter(|i| matches!(&i.kind, InstKind::StringAppend { .. }))
            .count();
        assert!(has_text);
        assert_eq!(appends, 1);
    }

    #[test]
    fn extern_param_write_rejected() {
        let interner = Interner::new();
        let result = crate::test::compile_template(&interner, "% $count = 42", &[]);
        assert!(result.is_err());
    }

    /// A `%` line's block lowers through the script's own lowering: an
    /// `% if` is the `Diamond` an `if` statement is (RFC-0071).
    #[test]
    fn a_template_if_lowers_as_the_scripts_if() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let module = lower_with(&interner, "% if @n == 1\nmatched\n% end\n", &context);
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Diamond { .. } | InstKind::JumpIf { .. }))
        );
    }

    /// A text line and a tag each append to the accumulator; nothing joins
    /// them into a rebuilt `String`.
    #[test]
    fn every_part_of_a_template_is_one_append() {
        let interner = Interner::new();
        let module = lower(&interner, "a{{ \"b\" }}c");
        let appends = module
            .main
            .insts
            .iter()
            .filter(|i| matches!(&i.kind, InstKind::StringAppend { .. }))
            .count();
        assert_eq!(appends, 3);
        assert!(
            !module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::StringConcat { .. }))
        );
    }

    #[test]
    #[ignore = "requires Phase 2: builtin -> graph Function migration"]
    fn lower_builtin_call() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let module = lower_with(&interner, r#"{{ @n | to_string }}"#, &context);
        let has_call = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call);
    }
}
