//! MIR type-verification pass (error-collecting variant).
//!
//! Walks every instruction in a [`MirModule`] and checks that the types
//! recorded in `val_types` are exactly consistent with what each instruction
//! expects.  Any mismatch is collected as a [`ValidationError`] instead of
//! panicking.
//!
//! Design:
//! - `Ty::Error` unifies with anything (analysis mode may leave them
//!   unresolved).  `Ty::Var(Infallible)` is uninhabitable for concrete types.
//! - A cast is a pure `FunctionCall` of the cast ExternFn; no instruction
//!   changes a value's type in place.
//! - Generic variance is invariant: inner types must match recursively.

use crate::ir::{
    Callee, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValOrigin, ValueId,
};
use crate::ir::{ExternInstance, ForSource, IndexMode};
use crate::ty::{CastTy, Mutability, Ty, TypeArg};
use crate::validate::move_check::is_move_only;
use acvus_ast::{BinOp, Literal, Span, UnaryOp};
use acvus_utils::{Astr, LocalIdOps};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub scope: String,
    pub inst_index: usize,
    pub span: Span,
    pub kind: ValidationErrorKind,
}

impl ValidationError {
    /// The other places this refusal names, in the words of the site that
    /// raised it.
    pub fn labels(&self) -> &[acvus_ast::report::Label] {
        match &self.kind {
            ValidationErrorKind::UseAfterMove { labels, .. }
            | ValidationErrorKind::BorrowConflict { labels, .. } => labels,
            _ => &[],
        }
    }
}

/// A set of values no list of arms can name in full, so a `match` over one
/// closes only through a catch-all (RFC-0051 rule 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenSpace {
    /// The scrutinee's type names no variants at all.
    AType,
    Integers,
    Chars,
    Strings,
}

impl OpenSpace {
    pub fn shown(self) -> &'static str {
        match self {
            OpenSpace::AType => "the scrutinee's type names no variants",
            OpenSpace::Integers => "the integers are an open value space",
            OpenSpace::Chars => "the chars are an open value space",
            OpenSpace::Strings => "the strings are an open value space",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValidationErrorKind {
    TypeMismatch {
        inst_name: String,
        desc: String,
        expected: Ty,
        actual: Ty,
    },
    MissingType {
        value_id: u32,
    },
    /// A value the checker left with the poison type reached the machine.
    /// The type exists to suppress the cascade after a reported refusal, so
    /// one that arrives here says a refusal was not reported.
    ErrorType {
        value_id: u32,
        origin: Option<ValOrigin>,
    },
    /// A call's `Order` operand disagrees with its callee's effect: `pure`
    /// says which side the callee is on.
    OrderEdge {
        inst_name: String,
        pure: bool,
    },
    ArityMismatch {
        inst_name: String,
        expected: usize,
        got: usize,
    },
    /// A `match` with no catch-all whose arms do not close what the
    /// scrutinee can hold (RFC-0051 rule 3).
    NonExhaustiveMatch {
        over: OpenSpace,
    },
    /// The two bounds of a `for i in lo..hi` are not one integer width
    /// (RFC-0057 rule 1).
    ForRangeWidths {
        at: Ty,
        hi: Ty,
    },
    /// RFC-0063 rule 1.
    DiamondArmMissesJoin {
        side: &'static str,
        join: Label,
    },
    /// RFC-0063 rule 5: the arms of a branch `cfg::demote_diamond`
    /// demoted meet again, and the terminator was not restored.
    DemotedDiamondMeetsAgain {
        join: Label,
    },
    /// A `match` over a locally closed enum leaves a variant untaken.
    MatchMissesVariants {
        enum_name: Option<Astr>,
        missing: Vec<Astr>,
    },
    /// A `match` over `Bool` names one of the two values and no catch-all.
    MatchMissesBoolArm {
        missing: bool,
    },
    /// A `match` over an `Option` or a `Result` leaves a variant untaken.
    /// The arms are named by typeck, so the count is the whole check.
    MatchMissesBuiltinVariants {
        enum_name: &'static str,
        arity: usize,
        covered: usize,
    },
    InvalidConstructor {
        inst_name: String,
        expected_constructor: String,
        actual: Ty,
    },
    /// Use of a move-only value after it has been consumed. `labels` names
    /// where it was moved; it is empty where the move is a synthesized
    /// instruction, which carries no span.
    UseAfterMove {
        value_id: u32,
        ty: Ty,
        /// The subject's `DebugInfo` origin, which names it as the source
        /// wrote it.
        origin: Option<ValOrigin>,
        labels: Vec<acvus_ast::report::Label>,
    },
    /// A storage touched while a reference to it excludes that (RFC-0018).
    /// `labels` names the borrow and the use that keeps it live.
    BorrowConflict {
        storage: Option<ValOrigin>,
        touch: ConflictTouch,
        labels: Vec<acvus_ast::report::Label>,
    },
    /// A context moved out of and never assigned again, so the `Commit` that
    /// ends the run finds its place empty (RFC-0025). Stated at the move.
    ContextMovedOut {
        context: Astr,
    },
    /// The body's result holds a loan on a local, whose storage the run is
    /// about to leave (RFC-0064 rule 5).
    ReferenceToLocalLeavesBody {
        storage: Option<ValOrigin>,
    },
}

/// What the conflicting instruction does to the storage, in the word the
/// refusal uses for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictTouch {
    Read,
    Written,
    Moved,
}

impl ConflictTouch {
    pub fn word(self) -> &'static str {
        match self {
            ConflictTouch::Read => "read",
            ConflictTouch::Written => "written",
            ConflictTouch::Moved => "moved",
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Check type consistency of the entire module.  Returns all errors found.
pub fn check_types(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let declared = declared_closure_returns(module);

    let mut ctx = CheckCtx::new("main".to_string(), Some(module.ret.clone()));
    ctx.check_body(&module.main, &mut errors);

    for (label, closure) in &module.closures {
        let name = format!("closure({:?})", label);
        let mut ctx = CheckCtx::new(name, declared.get(label).cloned());
        ctx.check_body(closure, &mut errors);
    }

    errors
}

/// What each closure body declares it returns: the `ret` of the `Ty::Fn` that
/// the `MakeClosure` building it gives its value. A closure whose
/// `MakeClosure` is gone - inlined away, or dead - declares nothing.
fn declared_closure_returns(module: &MirModule) -> FxHashMap<Label, Ty> {
    std::iter::once(&module.main)
        .chain(module.closures.values())
        .flat_map(|maker| {
            maker.insts.iter().filter_map(|inst| {
                let InstKind::MakeClosure {
                    dst, body: made, ..
                } = &inst.kind
                else {
                    return None;
                };
                let Some(Ty::Fn { ret, .. }) = maker.val_types.get(dst) else {
                    return None;
                };
                Some((*made, (**ret).clone()))
            })
        })
        .collect()
}

struct BranchEdge<'a> {
    side: &'static str,
    label: Label,
    args: &'a [ValueId],
}

// ---------------------------------------------------------------------------
// Structural type equality (invariant, with Error escape)
// ---------------------------------------------------------------------------

/// Returns `true` if `expected` and `actual` are structurally equal under
/// invariant variance.  `Ty::Error` matches anything (poison).
/// `Ty::Var(Infallible)` is uninhabitable for concrete types.
fn types_match(expected: &Ty, actual: &Ty) -> bool {
    match (expected, actual) {
        // Poison - accept anything.
        (Ty::Error(_), _) | (_, Ty::Error(_)) => true,
        // A slot declared `!` states no type, so it accepts any value
        // (RFC-0054).
        (Ty::Never, _) => true,
        // Uninhabitable - concrete types never have Var.
        (Ty::Var(v), _) | (_, Ty::Var(v)) => match *v {},

        // Primitives
        (Ty::Int(a), Ty::Int(b)) => a == b,
        (Ty::Float, Ty::Float) => true,
        (Ty::Char, Ty::Char) => true,
        (Ty::String, Ty::String) => true,
        (Ty::Str, Ty::Str) => true,
        (Ty::Bool, Ty::Bool) => true,
        (Ty::Unit, Ty::Unit) => true,
        (Ty::Order, Ty::Order) => true,

        // Containers (invariant inner)
        (Ty::Array(a, la), Ty::Array(b, lb)) => la == lb && types_match(a, b),
        (Ty::Option(a), Ty::Option(b)) | (Ty::Slice(a), Ty::Slice(b)) => types_match(a, b),
        (Ty::Result(ta, ea), Ty::Result(tb, eb)) => types_match(ta, tb) && types_match(ea, eb),
        (Ty::Ref(ma, a), Ty::Ref(mb, b)) => {
            ma == mb && a.repr == b.repr && types_match(&a.ty, &b.ty)
        }
        (Ty::Tuple(a), Ty::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| types_match(x, y))
        }
        (Ty::Object(a), Ty::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|bv| types_match(v, bv)))
        }
        // Functions
        (
            Ty::Fn {
                params: p1,
                ret: r1,
                ..
            },
            Ty::Fn {
                params: p2,
                ret: r2,
                ..
            },
        ) => {
            p1.len() == p2.len()
                && p1.iter().zip(p2).all(|(a, b)| types_match(&a.ty, &b.ty))
                && types_match(r1, r2)
        }

        // Enum - same name is sufficient (variants are open/unified elsewhere)
        (Ty::Enum { name: n1, .. }, Ty::Enum { name: n2, .. }) => n1 == n2,

        // UserDefined - same id
        (Ty::UserDefined { id: a, .. }, Ty::UserDefined { id: b, .. }) => a == b,

        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn str_ty() -> Ty {
    Ty::Ref(
        crate::ty::Mutability::Shared,
        Box::new(crate::ty::TypeArg::uniform(Ty::Str)),
    )
}

fn literal_ty(lit: &Literal) -> Ty {
    match lit {
        Literal::String(_) => Ty::String,
        Literal::Int(_) => Ty::I64,
        Literal::IntOf(n) => Ty::Int(n.width.into()),
        Literal::Float(_) => Ty::Float,
        Literal::Char(_) => Ty::Char,
        Literal::Bytes(bytes) => {
            Ty::Array(Box::new(Ty::U8), crate::ty::LenTerm::Known(bytes.len()))
        }
        Literal::Bool(_) => Ty::Bool,
        Literal::List(_) => Ty::error(),
        Literal::Unit => Ty::Unit,
    }
}

/// When the lowerer reuses a ValueId for both a collection and its iterated
/// element (e.g. pattern-match iteration over a List), the val_types map may
/// record the collection type rather than the element type.  This helper
/// unwraps one level of Array so pattern-match checks don't produce
/// false positives.
fn unwrap_element_ty(ty: &Ty) -> &Ty {
    match ty {
        Ty::Array(inner, _) => inner,
        other => other,
    }
}

fn as_array_inner(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::Array(inner, _) => Some(inner),
        _ => None,
    }
}

/// Returns `true` if the BinOp is a comparison that returns Bool.
fn binop_returns_bool(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte
    )
}

/// Returns `true` if the BinOp is a logical op (Bool x Bool -> Bool).
fn binop_is_logical(op: BinOp) -> bool {
    matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
}

// ---------------------------------------------------------------------------
// Check context
// ---------------------------------------------------------------------------

struct CheckCtx {
    scope_name: String,
    /// label -> index in `insts` (for Jump target block param lookup)
    label_map: FxHashMap<Label, usize>,
    declared_ret: Option<Ty>,
}

impl CheckCtx {
    fn new(scope_name: String, declared_ret: Option<Ty>) -> Self {
        Self {
            scope_name,
            label_map: FxHashMap::default(),
            declared_ret,
        }
    }

    fn check_body(&mut self, body: &MirBody, errors: &mut Vec<ValidationError>) {
        // Build label map
        self.label_map.clear();
        for (i, inst) in body.insts.iter().enumerate() {
            if let InstKind::BlockLabel { label, .. } = &inst.kind {
                self.label_map.insert(*label, i);
            }
        }

        for (pc, inst) in body.insts.iter().enumerate() {
            self.check_inst(
                pc,
                inst.span,
                &inst.kind,
                &body.val_types,
                &body.insts,
                errors,
            );
            for value in crate::analysis::inst_info::defs(&inst.kind) {
                if matches!(body.val_types.get(&value), Some(Ty::Error(_))) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span: inst.span,
                        kind: ValidationErrorKind::ErrorType {
                            value_id: value.to_raw() as u32,
                            origin: body.debug.val_origins.get(&value).cloned(),
                        },
                    });
                }
            }
        }

        for branch in crate::ir::demoted_branches(&body.insts, &body.demoted_diamonds) {
            let Some(join) = crate::ir::meets_again(&body.insts, branch.at) else {
                continue;
            };
            errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: branch.at,
                span: body.insts[branch.at].span,
                kind: ValidationErrorKind::DemotedDiamondMeetsAgain { join },
            });
        }
    }

    /// Get the type of a ValueId.  Pushes `MissingType` if absent and returns
    /// a reference to a fallback `Ty::Error`.
    fn ty_of<'b>(
        &self,
        id: ValueId,
        val_types: &'b FxHashMap<ValueId, Ty>,
        span: Span,
        pc: usize,
        errors: &mut Vec<ValidationError>,
    ) -> Option<&'b Ty> {
        match val_types.get(&id) {
            Some(ty) => Some(ty),
            None => {
                errors.push(ValidationError {
                    scope: self.scope_name.clone(),
                    inst_index: pc,
                    span,
                    kind: ValidationErrorKind::MissingType {
                        value_id: id.to_raw() as u32,
                    },
                });
                None
            }
        }
    }

    /// Assert two types match.  Pushes a `TypeMismatch` error on failure.
    /// Returns `true` if they match.
    /// `id` must be typed `Order`.
    fn expect_order(
        &self,
        id: ValueId,
        val_types: &FxHashMap<ValueId, Ty>,
        span: Span,
        pc: usize,
        errors: &mut Vec<ValidationError>,
    ) {
        if let Some(ty) = self.ty_of(id, val_types, span, pc, errors) {
            self.assert_match(pc, span, "Order", "order", &Ty::Order, ty, errors);
        }
    }

    /// A call takes an `Order` exactly when its callee's effect is not Pure.
    fn expect_order_edge(
        &self,
        inst_name: &str,
        callee_ty: &Ty,
        before: Option<ValueId>,
        val_types: &FxHashMap<ValueId, Ty>,
        span: Span,
        pc: usize,
        errors: &mut Vec<ValidationError>,
    ) {
        let Some(effect) = callee_ty.effect() else {
            return;
        };
        match (effect.is_pure(), before) {
            (false, Some(o)) => self.expect_order(o, val_types, span, pc, errors),
            (true, None) => {}
            (pure, _) => errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: pc,
                span,
                kind: ValidationErrorKind::OrderEdge {
                    inst_name: inst_name.to_string(),
                    pure,
                },
            }),
        }
    }

    fn through(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        r: ValueId,
        path: &[PathSeg],
        vt: &FxHashMap<ValueId, Ty>,
        errors: &mut Vec<ValidationError>,
    ) -> Option<(Mutability, Ty)> {
        let r_ty = self.ty_of(r, vt, span, pc, errors)?;
        let Ty::Ref(m, inner) = r_ty else {
            if !r_ty.is_error() {
                errors.push(ValidationError {
                    scope: self.scope_name.clone(),
                    inst_index: pc,
                    span,
                    kind: ValidationErrorKind::InvalidConstructor {
                        inst_name: inst_name.to_string(),
                        expected_constructor: "Ref".to_string(),
                        actual: r_ty.clone(),
                    },
                });
            }
            return None;
        };
        let mut at: Ty = inner.ty.clone();
        for seg in path {
            let next = match (seg, &at) {
                (PathSeg::Field(field), Ty::Object(fields)) => fields.get(field).cloned(),
                (PathSeg::Index(i), Ty::Array(elem, _)) => {
                    Some(elem.as_ref().clone()).filter(|_| *i < usize::MAX)
                }
                (PathSeg::Index(i), Ty::Tuple(elems)) => elems.get(*i).cloned(),
                (PathSeg::Payload, Ty::Option(payload)) => Some(payload.as_ref().clone()),
                (PathSeg::Payload, Ty::Result(..)) => Some(Ty::error()),
                // Enum payload: dst type comes from val_types, trust typechecker
                (PathSeg::Payload, Ty::Enum { .. }) => Some(Ty::error()),
                (_, Ty::Error(_)) => Some(Ty::error()),
                _ => None,
            };
            let Some(next) = next else {
                errors.push(ValidationError {
                    scope: self.scope_name.clone(),
                    inst_index: pc,
                    span,
                    kind: ValidationErrorKind::InvalidConstructor {
                        inst_name: inst_name.to_string(),
                        expected_constructor: match seg {
                            PathSeg::Field(_) => "an object carrying that field".to_string(),
                            PathSeg::Index(i) => {
                                format!("an array or tuple carrying element {i}")
                            }
                            PathSeg::Payload => {
                                "an Option, a Result or an enum carrying a payload".to_string()
                            }
                        },
                        actual: at.clone(),
                    },
                });
                return None;
            };
            at = next;
        }
        Some((*m, at))
    }

    /// A type whose head is not the one the instruction is built on.
    fn invalid(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        expected_constructor: &str,
        actual: &Ty,
        errors: &mut Vec<ValidationError>,
    ) {
        errors.push(ValidationError {
            scope: self.scope_name.clone(),
            inst_index: pc,
            span,
            kind: ValidationErrorKind::InvalidConstructor {
                inst_name: inst_name.to_string(),
                expected_constructor: expected_constructor.to_string(),
                actual: actual.clone(),
            },
        });
    }

    fn assert_match(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        desc: &str,
        expected: &Ty,
        actual: &Ty,
        errors: &mut Vec<ValidationError>,
    ) -> bool {
        if !types_match(expected, actual) {
            errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: pc,
                span,
                kind: ValidationErrorKind::TypeMismatch {
                    inst_name: inst_name.to_string(),
                    desc: desc.to_string(),
                    expected: expected.clone(),
                    actual: actual.clone(),
                },
            });
            false
        } else {
            true
        }
    }

    /// Get block params for a label.
    /// One edge's arguments against the parameters they fill: as many, and
    /// each of the parameter's type.
    fn check_edge_args(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        params: &[ValueId],
        args: &[ValueId],
        vt: &FxHashMap<ValueId, Ty>,
        errors: &mut Vec<ValidationError>,
    ) {
        if args.len() != params.len() {
            errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: pc,
                span,
                kind: ValidationErrorKind::ArityMismatch {
                    inst_name: inst_name.to_string(),
                    expected: params.len(),
                    got: args.len(),
                },
            });
            return;
        }
        for (i, (arg, param)) in args.iter().zip(params).enumerate() {
            let (Some(param_ty), Some(arg_ty)) = (
                self.ty_of(*param, vt, span, pc, errors),
                self.ty_of(*arg, vt, span, pc, errors),
            ) else {
                continue;
            };
            self.assert_match(
                pc,
                span,
                inst_name,
                &format!("arg[{i}]"),
                param_ty,
                arg_ty,
                errors,
            );
        }
    }

    fn block_params(&self, label: &Label, insts: &[crate::ir::Inst]) -> Option<Vec<ValueId>> {
        let idx = self.label_map.get(label)?;
        match &insts[*idx].kind {
            InstKind::BlockLabel { params, .. } => Some(params.clone()),
            _ => None,
        }
    }

    // -- per-instruction check ------------------------------------------------

    fn check_inst(
        &self,
        pc: usize,
        span: Span,
        kind: &InstKind,
        vt: &FxHashMap<ValueId, Ty>,
        insts: &[crate::ir::Inst],
        errors: &mut Vec<ValidationError>,
    ) {
        // Macro to get ty_of with early-return-on-missing using fallback
        macro_rules! ty {
            ($id:expr) => {
                match self.ty_of($id, vt, span, pc, errors) {
                    Some(t) => t,
                    None => return,
                }
            };
        }

        match kind {
            // === Skip ===
            InstKind::Poison { .. }
            | InstKind::Undef { .. }
            | InstKind::Nop
            | InstKind::Diverge
            | InstKind::BlockLabel { .. } => {}

            InstKind::Drop { src } => {
                // Just validate src exists and has a type.
                let _ = self.ty_of(*src, vt, span, pc, errors);
            }

            // === Slices (RFC-0047) ===
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                instance: ExternInstance { .. },
            } => {
                let container_ty = ty!(*container);
                let Ty::Ref(held, _) = container_ty else {
                    self.invalid(pc, span, "AsSlice", "Ref", container_ty, errors);
                    return;
                };
                if *mutability == Mutability::Mut && *held != Mutability::Mut {
                    self.invalid(pc, span, "AsSlice", "Ref(Mut)", container_ty, errors);
                }
                let dst_ty = ty!(*dst);
                if run_of(dst_ty) != Some(*mutability) && !dst_ty.is_error() {
                    self.invalid(
                        pc,
                        span,
                        "AsSlice",
                        &format!("{}[_] or &str", mutability.prefix()),
                        dst_ty,
                        errors,
                    );
                }
            }
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            } => {
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "Index", "index", &Ty::U64, index_ty, errors);
                let slice_ty = ty!(*slice);
                let Some((mutability, element)) = slice_of(slice_ty) else {
                    self.invalid(pc, span, "Index", "Ref(_, Slice)", slice_ty, errors);
                    return;
                };
                let dst_ty = ty!(*dst);
                match mode {
                    IndexMode::Copy => {
                        self.assert_match(pc, span, "Index", "dst", element, dst_ty, errors);
                        if is_move_only(element) == Some(true) {
                            self.invalid(
                                pc,
                                span,
                                "Index",
                                "a word element, which Copy mode reads",
                                element,
                                errors,
                            );
                        }
                    }
                    IndexMode::Ref => {
                        // The element is reached through the slice, so it
                        // is reached at the slice's own mutability.
                        let expected =
                            Ty::Ref(mutability, Box::new(TypeArg::uniform(element.clone())));
                        self.assert_match(pc, span, "Index", "dst", &expected, dst_ty, errors);
                    }
                }
            }
            InstKind::IndexSet {
                slice,
                index,
                value,
            } => {
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "IndexSet", "index", &Ty::U64, index_ty, errors);
                let slice_ty = ty!(*slice);
                let Some((Mutability::Mut, element)) = slice_of(slice_ty) else {
                    self.invalid(pc, span, "IndexSet", "Ref(Mut, Slice)", slice_ty, errors);
                    return;
                };
                let value_ty = ty!(*value);
                self.assert_match(pc, span, "IndexSet", "value", element, value_ty, errors);
            }

            // === Const ===
            InstKind::Const { dst, value } => {
                let dst_ty = ty!(*dst);
                if let Literal::Int(_) = value {
                    if !matches!(dst_ty, Ty::Int(_)) {
                        self.assert_match(pc, span, "Const", "dst", &Ty::I64, dst_ty, errors);
                    }
                } else {
                    let lit_ty = literal_ty(value);
                    if !lit_ty.is_error() {
                        self.assert_match(pc, span, "Const", "dst", &lit_ty, dst_ty, errors);
                    }
                }
            }

            // === ConstStr ===
            InstKind::ConstStr { dst, .. } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "ConstStr", "dst", &str_ty(), dst_ty, errors);
            }

            // === Constructors ===
            InstKind::StringClone { dst, src } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringClone", "dst", &Ty::String, dst_ty, errors);
                let src_ty = ty!(*src);
                let is_string = match src_ty {
                    Ty::String | Ty::Error(_) => true,
                    Ty::Ref(_, inner) => matches!(inner.ty, Ty::String),
                    _ => false,
                };
                if !is_string {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "StringClone".to_string(),
                            expected_constructor: "String or &String".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
            }
            InstKind::StringEq { dst, a, b } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringEq", "dst", &Ty::Bool, dst_ty, errors);
                for operand in [a, b] {
                    let operand_ty = ty!(*operand);
                    let is_lent_text = matches!(operand_ty, Ty::Ref(_, inner) if matches!(inner.ty, Ty::String | Ty::Str))
                        || operand_ty.is_error();
                    if !is_lent_text {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "StringEq".to_string(),
                                expected_constructor: "&String or &str".to_string(),
                                actual: operand_ty.clone(),
                            },
                        });
                    }
                }
            }
            InstKind::StringConcat { dst, parts } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringConcat", "dst", &Ty::String, dst_ty, errors);
                for part in parts {
                    let part_ty = ty!(*part);
                    let is_text = match part_ty {
                        Ty::String => true,
                        Ty::Ref(_, inner) => matches!(inner.ty, Ty::String | Ty::Str),
                        Ty::Error(_) => true,
                        _ => false,
                    };
                    if !is_text {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "StringConcat".to_string(),
                                expected_constructor: "String, &String or &str".to_string(),
                                actual: part_ty.clone(),
                            },
                        });
                    }
                }
            }
            InstKind::StringAppend { target, part } => {
                let target_ty = ty!(*target);
                let expected = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::String)));
                self.assert_match(
                    pc,
                    span,
                    "StringAppend",
                    "target",
                    &expected,
                    target_ty,
                    errors,
                );
                let part_ty = ty!(*part);
                let is_text = match part_ty {
                    Ty::String => true,
                    Ty::Ref(_, inner) => matches!(inner.ty, Ty::String | Ty::Str),
                    Ty::Error(_) => true,
                    _ => false,
                };
                if !is_text {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "StringAppend".to_string(),
                            expected_constructor: "String, &String or &str".to_string(),
                            actual: part_ty.clone(),
                        },
                    });
                }
            }
            InstKind::MakeArray { dst, elements } => {
                let dst_ty = ty!(*dst);
                if let Ty::Array(inner, len) = dst_ty {
                    if len.get() != elements.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "MakeArray".to_string(),
                                expected_constructor: format!("Array of length {}", elements.len()),
                                actual: dst_ty.clone(),
                            },
                        });
                    }
                    for (i, elem) in elements.iter().enumerate() {
                        let elem_ty = ty!(*elem);
                        self.assert_match(
                            pc,
                            span,
                            "MakeArray",
                            &format!("element[{i}]"),
                            inner,
                            elem_ty,
                            errors,
                        );
                    }
                } else if !dst_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "MakeArray".to_string(),
                            expected_constructor: "Array".to_string(),
                            actual: dst_ty.clone(),
                        },
                    });
                }
            }

            InstKind::MakeObject { dst, fields } => {
                let dst_ty = ty!(*dst);
                if let Ty::Object(field_tys) = dst_ty {
                    for (key, val) in fields {
                        if let Some(expected_field_ty) = field_tys.get(key) {
                            let val_ty = ty!(*val);
                            self.assert_match(
                                pc,
                                span,
                                "MakeObject",
                                "field",
                                expected_field_ty,
                                val_ty,
                                errors,
                            );
                        }
                    }
                } else if !dst_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "MakeObject".to_string(),
                            expected_constructor: "Object".to_string(),
                            actual: dst_ty.clone(),
                        },
                    });
                }
            }

            InstKind::MakeTuple { dst, elements } => {
                let dst_ty = ty!(*dst);
                if let Ty::Tuple(elem_tys) = dst_ty {
                    if elem_tys.len() != elements.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "MakeTuple".to_string(),
                                expected: elem_tys.len(),
                                got: elements.len(),
                            },
                        });
                    } else {
                        for (i, (elem, expected)) in elements.iter().zip(elem_tys).enumerate() {
                            let elem_ty = ty!(*elem);
                            self.assert_match(
                                pc,
                                span,
                                "MakeTuple",
                                &format!("element[{i}]"),
                                expected,
                                elem_ty,
                                errors,
                            );
                        }
                    }
                } else if !dst_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "MakeTuple".to_string(),
                            expected_constructor: "Tuple".to_string(),
                            actual: dst_ty.clone(),
                        },
                    });
                }
            }

            InstKind::MakeClosure { dst, captures, .. } => {
                let dst_ty = ty!(*dst);
                if let Ty::Fn {
                    captures: cap_tys, ..
                } = dst_ty
                {
                    if cap_tys.len() != captures.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "MakeClosure".to_string(),
                                expected: cap_tys.len(),
                                got: captures.len(),
                            },
                        });
                    } else {
                        for (i, (cap, expected)) in captures.iter().zip(cap_tys).enumerate() {
                            let cap_ty = ty!(*cap);
                            self.assert_match(
                                pc,
                                span,
                                "MakeClosure",
                                &format!("capture[{i}]"),
                                expected,
                                cap_ty,
                                errors,
                            );
                        }
                    }
                }
                // If dst is not Fn (e.g. Error), skip
            }

            InstKind::MakeVariant { dst, tag, payload } => {
                let dst_ty = ty!(*dst);
                if let Ty::Enum { variants, .. } = dst_ty {
                    if let Some(variant_payload_ty) = variants.get(tag) {
                        match (variant_payload_ty, payload) {
                            (Some(expected), Some(val)) => {
                                let val_ty = ty!(*val);
                                self.assert_match(
                                    pc,
                                    span,
                                    "MakeVariant",
                                    "payload",
                                    expected,
                                    val_ty,
                                    errors,
                                );
                            }
                            (None, None) => {}
                            (Some(_), None) => {
                                errors.push(ValidationError {
                                    scope: self.scope_name.clone(),
                                    inst_index: pc,
                                    span,
                                    kind: ValidationErrorKind::ArityMismatch {
                                        inst_name: "MakeVariant".to_string(),
                                        expected: 1,
                                        got: 0,
                                    },
                                });
                            }
                            (None, Some(_)) => {
                                errors.push(ValidationError {
                                    scope: self.scope_name.clone(),
                                    inst_index: pc,
                                    span,
                                    kind: ValidationErrorKind::ArityMismatch {
                                        inst_name: "MakeVariant".to_string(),
                                        expected: 0,
                                        got: 1,
                                    },
                                });
                            }
                        }
                    }
                    // Tag not found in type - open enum, skip
                } else if let Ty::Option(inner) = dst_ty {
                    // Option is represented as enum with Some/None tags
                    if let Some(val) = payload {
                        let val_ty = ty!(*val);
                        self.assert_match(
                            pc,
                            span,
                            "MakeVariant",
                            "Option payload",
                            inner,
                            val_ty,
                            errors,
                        );
                    }
                }
            }

            // === BinOp ===
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let left_ty = ty!(*left);
                let right_ty = ty!(*right);
                let dst_ty = ty!(*dst);

                if binop_is_logical(*op) {
                    self.assert_match(
                        pc,
                        span,
                        "BinOp(logical)",
                        "left",
                        &Ty::Bool,
                        left_ty,
                        errors,
                    );
                    self.assert_match(
                        pc,
                        span,
                        "BinOp(logical)",
                        "right",
                        &Ty::Bool,
                        right_ty,
                        errors,
                    );
                    self.assert_match(pc, span, "BinOp(logical)", "dst", &Ty::Bool, dst_ty, errors);
                } else if binop_returns_bool(*op) {
                    self.assert_match(
                        pc,
                        span,
                        "BinOp(cmp)",
                        "left == right",
                        left_ty,
                        right_ty,
                        errors,
                    );
                    self.assert_match(pc, span, "BinOp(cmp)", "dst", &Ty::Bool, dst_ty, errors);
                } else {
                    self.assert_match(
                        pc,
                        span,
                        "BinOp",
                        "left == right",
                        left_ty,
                        right_ty,
                        errors,
                    );
                    self.assert_match(pc, span, "BinOp", "left == dst", left_ty, dst_ty, errors);
                }
            }

            // === UnaryOp ===
            InstKind::UnaryOp { dst, op, operand } => {
                let operand_ty = ty!(*operand);
                let dst_ty = ty!(*dst);
                match op {
                    UnaryOp::Deref => errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "UnaryOp(Deref)".to_string(),
                            expected_constructor: "Load".to_string(),
                            actual: operand_ty.clone(),
                        },
                    }),
                    UnaryOp::Not => {
                        self.assert_match(
                            pc,
                            span,
                            "UnaryOp(Not)",
                            "operand",
                            &Ty::Bool,
                            operand_ty,
                            errors,
                        );
                        self.assert_match(
                            pc,
                            span,
                            "UnaryOp(Not)",
                            "dst",
                            &Ty::Bool,
                            dst_ty,
                            errors,
                        );
                    }
                    UnaryOp::Neg => {
                        self.assert_match(
                            pc,
                            span,
                            "UnaryOp(Neg)",
                            "operand == dst",
                            operand_ty,
                            dst_ty,
                            errors,
                        );
                    }
                }
            }

            // === Cast (RFC-0049) ===
            InstKind::Cast { dst, src, to } => {
                let src_ty = ty!(*src);
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "Cast", "dst", &Ty::from(*to), dst_ty, errors);
                if CastTy::of_ty(src_ty).is_none() && !src_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "Cast".to_string(),
                            expected_constructor: "an integer width or f64".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
            }

            // === Projection ===
            InstKind::Take { dst, target, path } => {
                let dst_ty = ty!(*dst);
                if let RefTarget::Through(r) = target {
                    let Some((_, at)) = self.through(pc, span, "Take", *r, path, vt, errors) else {
                        return;
                    };
                    self.assert_match(pc, span, "Take", "dst", &at, dst_ty, errors);
                    if !at.is_primitive()
                        && !matches!(at, Ty::String | Ty::Ref(..))
                        && !at.is_error()
                    {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "Take".to_string(),
                                expected_constructor: "a primitive through a reference".to_string(),
                                actual: at.clone(),
                            },
                        });
                    }
                }
            }
            InstKind::Assign {
                value,
                target,
                path,
            } => {
                let val_ty = ty!(*value);
                if let RefTarget::Through(r) = target {
                    let Some((m, at)) = self.through(pc, span, "Assign", *r, path, vt, errors)
                    else {
                        return;
                    };
                    if m != Mutability::Mut {
                        let r_ty = ty!(*r).clone();
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "Assign".to_string(),
                                expected_constructor: "Ref(Mut)".to_string(),
                                actual: r_ty,
                            },
                        });
                    }
                    self.assert_match(pc, span, "Assign", "value", &at, val_ty, errors);
                }
            }
            InstKind::Fetch { dst, .. } => {
                let _ = self.ty_of(*dst, vt, span, pc, errors);
            }
            InstKind::Commit { value, .. } => {
                let _ = self.ty_of(*value, vt, span, pc, errors);
            }
            InstKind::Ref {
                dst,
                mutability,
                target,
                path,
            } => {
                let dst_ty = ty!(*dst);
                if let RefTarget::Through(r) = target {
                    let Some((m, at)) = self.through(pc, span, "Ref", *r, path, vt, errors) else {
                        return;
                    };
                    if *mutability == Mutability::Mut && m != Mutability::Mut {
                        let r_ty = ty!(*r).clone();
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "Ref".to_string(),
                                expected_constructor: "Ref(Mut) to reborrow mutably".to_string(),
                                actual: r_ty,
                            },
                        });
                    }
                    let expected = Ty::Ref(*mutability, Box::new(TypeArg::uniform(at)));
                    self.assert_match(pc, span, "Ref", "dst", &expected, dst_ty, errors);
                    return;
                }
                match dst_ty {
                    Ty::Ref(m, _) if m == mutability => {}
                    Ty::Error(_) => {}
                    other => errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "Ref".to_string(),
                            expected_constructor: format!("{}_", mutability.prefix()),
                            actual: other.clone(),
                        },
                    }),
                }
            }
            // === Scalar field access ===
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let obj_ty = ty!(*object);
                // Try direct type first, then unwrap one container level.
                let obj_ty = if matches!(obj_ty, Ty::Object(_) | Ty::Error(_)) {
                    obj_ty
                } else {
                    unwrap_element_ty(obj_ty)
                };
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(field) {
                        // Walk through rest fields to get the final type.
                        let mut resolved = field_ty.clone();
                        for r in rest {
                            if let Ty::Object(inner) = &resolved {
                                if let Some(next) = inner.get(r) {
                                    resolved = next.clone();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "FieldGet", "dst", &resolved, dst_ty, errors);
                    }
                } else if !obj_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "FieldGet".to_string(),
                            expected_constructor: "Object".to_string(),
                            actual: obj_ty.clone(),
                        },
                    });
                }
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                // object must be Object, value must match the leaf field type, dst must be same Object type.
                let obj_ty = ty!(*object);
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(field) {
                        // Walk through rest fields to get the leaf type.
                        let mut resolved = field_ty.clone();
                        for r in rest {
                            if let Ty::Object(inner) = &resolved {
                                if let Some(next) = inner.get(r) {
                                    resolved = next.clone();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        let val_ty = ty!(*value);
                        self.assert_match(pc, span, "FieldSet", "value", &resolved, val_ty, errors);
                    }
                    let dst_ty = ty!(*dst);
                    self.assert_match(
                        pc,
                        span,
                        "FieldSet",
                        "dst == object",
                        obj_ty,
                        dst_ty,
                        errors,
                    );
                } else if !obj_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "FieldSet".to_string(),
                            expected_constructor: "Object".to_string(),
                            actual: obj_ty.clone(),
                        },
                    });
                }
            }

            InstKind::ObjectGet { dst, object, key } => {
                let obj_ty = ty!(*object);
                let obj_ty = if matches!(obj_ty, Ty::Object(_) | Ty::Error(_)) {
                    obj_ty
                } else {
                    unwrap_element_ty(obj_ty)
                };
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(key) {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "ObjectGet", "dst", field_ty, dst_ty, errors);
                    }
                } else if !obj_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ObjectGet".to_string(),
                            expected_constructor: "Object".to_string(),
                            actual: obj_ty.clone(),
                        },
                    });
                }
            }

            InstKind::TupleIndex { dst, tuple, index } => {
                let tup_ty = ty!(*tuple);
                if let Ty::Tuple(elems) = tup_ty {
                    if let Some(elem_ty) = elems.get(*index) {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "TupleIndex", "dst", elem_ty, dst_ty, errors);
                    } else {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "TupleIndex".to_string(),
                                expected: elems.len(),
                                got: *index + 1,
                            },
                        });
                    }
                } else if !tup_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "TupleIndex".to_string(),
                            expected_constructor: "Tuple".to_string(),
                            actual: tup_ty.clone(),
                        },
                    });
                }
            }

            InstKind::ArrayIndex {
                dst, array: list, ..
            } => {
                let list_ty = ty!(*list);
                if let Some(inner) = as_array_inner(list_ty) {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ArrayIndex", "dst", inner, dst_ty, errors);
                } else if !list_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ArrayIndex".to_string(),
                            expected_constructor: "Array".to_string(),
                            actual: list_ty.clone(),
                        },
                    });
                }
            }

            // === Pattern tests (all produce Bool) ===
            InstKind::TestLiteral { dst, .. } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestLiteral", "dst", &Ty::Bool, dst_ty, errors);
            }

            InstKind::TestObjectKey { dst, src, .. } => {
                let src_ty = ty!(*src);
                let src_ty = match src_ty {
                    Ty::Ref(_, inner) => &inner.ty,
                    other => other,
                };
                let src_ty = if matches!(src_ty, Ty::Object(_) | Ty::Error(_)) {
                    src_ty
                } else {
                    unwrap_element_ty(src_ty)
                };
                if !matches!(src_ty, Ty::Object(_) | Ty::Error(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "TestObjectKey".to_string(),
                            expected_constructor: "Object".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestObjectKey", "dst", &Ty::Bool, dst_ty, errors);
            }

            InstKind::TestVariant { dst, src, .. } => {
                let src_ty = ty!(*src);
                let src_ty = match src_ty {
                    Ty::Ref(_, inner) => &inner.ty,
                    other => other,
                };
                if !matches!(
                    src_ty,
                    Ty::Enum { .. } | Ty::Option(_) | Ty::Result(..) | Ty::Error(_)
                ) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "TestVariant".to_string(),
                            expected_constructor: "Enum/Option".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestVariant", "dst", &Ty::Bool, dst_ty, errors);
            }

            InstKind::UnwrapVariant { dst, src } => {
                let src_ty = ty!(*src);
                match src_ty {
                    Ty::Option(inner) => {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "UnwrapVariant", "dst", inner, dst_ty, errors);
                    }
                    Ty::Enum { .. } | Ty::Result(..) => {
                        // Enum unwrap: dst type comes from val_types, trust typechecker
                    }
                    Ty::Error(_) => {}
                    _ => {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "UnwrapVariant".to_string(),
                                expected_constructor: "Enum/Option".to_string(),
                                actual: src_ty.clone(),
                            },
                        });
                    }
                }
            }

            // === Calls ===
            InstKind::LoadFunction { dst, .. } => {
                let _ = self.ty_of(*dst, vt, span, pc, errors);
            }

            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                order,
            } => {
                self.expect_order_edge(
                    "FunctionCall",
                    callee_ty,
                    order.map(|edge| edge.before),
                    vt,
                    span,
                    pc,
                    errors,
                );
                if let Some(edge) = order {
                    self.expect_order(edge.after, vt, span, pc, errors);
                }
                match callee {
                    Callee::Direct(_) | Callee::Extern { .. } => {
                        let _ = self.ty_of(*dst, vt, span, pc, errors);
                    }
                    Callee::Indirect(closure) => {
                        let closure_ty = ty!(*closure);
                        if let Ty::Fn { params, ret, .. } = closure_ty {
                            if args.len() != params.len() {
                                errors.push(ValidationError {
                                    scope: self.scope_name.clone(),
                                    inst_index: pc,
                                    span,
                                    kind: ValidationErrorKind::ArityMismatch {
                                        inst_name: "FunctionCall(Indirect)".to_string(),
                                        expected: params.len(),
                                        got: args.len(),
                                    },
                                });
                            } else {
                                for (i, (arg, param)) in args.iter().zip(params).enumerate() {
                                    let arg_ty = ty!(*arg);
                                    self.assert_match(
                                        pc,
                                        span,
                                        "FunctionCall(Indirect)",
                                        &format!("arg[{i}]"),
                                        &param.ty,
                                        arg_ty,
                                        errors,
                                    );
                                }
                            }
                            let dst_ty = ty!(*dst);
                            self.assert_match(
                                pc,
                                span,
                                "FunctionCall(Indirect)",
                                "return",
                                ret,
                                dst_ty,
                                errors,
                            );
                        }
                        // Fn type might be Error/Var - skip
                    }
                }
            }

            // === Spawn / Eval ===
            InstKind::Spawn {
                dst,
                callee,
                callee_ty,
                args,
                order,
            } => {
                self.expect_order_edge("Spawn", callee_ty, *order, vt, span, pc, errors);
                match callee {
                    Callee::Direct(_) | Callee::Extern { .. } => {
                        let dst_ty = ty!(*dst);
                        if !matches!(dst_ty, Ty::Handle(..) | Ty::Error(_)) {
                            errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::InvalidConstructor {
                                    inst_name: "Spawn".to_string(),
                                    expected_constructor: "Handle".to_string(),
                                    actual: dst_ty.clone(),
                                },
                            });
                        }
                    }
                    Callee::Indirect(closure) => {
                        let closure_ty = ty!(*closure);
                        if let Ty::Fn { params, ret, .. } = closure_ty {
                            if args.len() != params.len() {
                                errors.push(ValidationError {
                                    scope: self.scope_name.clone(),
                                    inst_index: pc,
                                    span,
                                    kind: ValidationErrorKind::ArityMismatch {
                                        inst_name: "Spawn(Indirect)".to_string(),
                                        expected: params.len(),
                                        got: args.len(),
                                    },
                                });
                            } else {
                                for (i, (arg, param)) in args.iter().zip(params).enumerate() {
                                    let arg_ty = ty!(*arg);
                                    self.assert_match(
                                        pc,
                                        span,
                                        "Spawn(Indirect)",
                                        &format!("arg[{i}]"),
                                        &param.ty,
                                        arg_ty,
                                        errors,
                                    );
                                }
                            }
                            let expected_dst = Ty::Handle(Box::new(ret.as_ref().clone()));
                            let dst_ty = ty!(*dst);
                            self.assert_match(
                                pc,
                                span,
                                "Spawn(Indirect)",
                                "dst",
                                &expected_dst,
                                dst_ty,
                                errors,
                            );
                        }
                        // Fn type might be Error/Var - skip
                    }
                }
            }

            InstKind::Merge { dst, orders } => {
                self.expect_order(*dst, vt, span, pc, errors);
                for o in orders {
                    self.expect_order(*o, vt, span, pc, errors);
                }
            }

            InstKind::Eval { dst, src, order } => {
                if let Some(o) = order {
                    self.expect_order(*o, vt, span, pc, errors);
                }
                let src_ty = ty!(*src);
                if let Ty::Handle(inner) = src_ty {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "Eval", "dst", inner, dst_ty, errors);
                } else if !src_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "Eval".to_string(),
                            expected_constructor: "Handle".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
            }

            // === Control flow ===
            // A `For` is the loop's condition (RFC-0057): the source decides
            // the element and the counter it hands the body, and the edges'
            // remaining arguments are checked the way a `Jump`'s are.
            InstKind::For {
                source,
                body,
                body_args,
                exit,
                exit_args,
            } => {
                let element = match source {
                    ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                        let slice_ty = ty!(*slice);
                        let wanted = match source {
                            ForSource::SliceMut(_) => Mutability::Mut,
                            _ => Mutability::Shared,
                        };
                        let Some((held, element)) = slice_of(slice_ty) else {
                            self.invalid(pc, span, "For", "Ref(_, Slice)", slice_ty, errors);
                            return;
                        };
                        if held != wanted {
                            self.invalid(
                                pc,
                                span,
                                "For",
                                &format!("{}[_]", wanted.prefix()),
                                slice_ty,
                                errors,
                            );
                        }
                        Ty::Ref(
                            wanted,
                            Box::new(crate::ty::TypeArg::uniform(element.clone())),
                        )
                    }
                    ForSource::Array(array) => {
                        let array_ty = ty!(*array);
                        let Ty::Array(element, _) = array_ty else {
                            self.invalid(pc, span, "For", "Array", array_ty, errors);
                            return;
                        };
                        (**element).clone()
                    }
                    ForSource::Range { at, hi } => {
                        let at_ty = ty!(*at).clone();
                        let hi_ty = ty!(*hi);
                        if !matches!(at_ty, Ty::Int(_)) {
                            self.invalid(pc, span, "For", "Int", &at_ty, errors);
                            return;
                        }
                        if !types_match(&at_ty, hi_ty) {
                            errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::ForRangeWidths {
                                    at: at_ty.clone(),
                                    hi: hi_ty.clone(),
                                },
                            });
                        }
                        at_ty
                    }
                };
                if let Some(params) = self.block_params(body, insts) {
                    let supplied = source.supplied_params();
                    if params.len() < supplied {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "For(body)".to_string(),
                                expected: supplied,
                                got: params.len(),
                            },
                        });
                        return;
                    }
                    let elem_ty = ty!(params[0]);
                    self.assert_match(pc, span, "For(body)", "elem", &element, elem_ty, errors);
                    if supplied == 2 {
                        let index_ty = ty!(params[1]);
                        self.assert_match(
                            pc,
                            span,
                            "For(body)",
                            "index",
                            &Ty::U64,
                            index_ty,
                            errors,
                        );
                    }
                    self.check_edge_args(
                        pc,
                        span,
                        "For(body)",
                        &params[supplied..],
                        body_args,
                        vt,
                        errors,
                    );
                }
                if let Some(params) = self.block_params(exit, insts) {
                    self.check_edge_args(pc, span, "For(exit)", &params, exit_args, vt, errors);
                }
            }

            // A `Switch` reads the tag of a variant and hands each edge the
            // block arguments its target takes (RFC-0051). The edge arities
            // are checked the way a `Jump`'s are.
            InstKind::Switch { tag, arms, default } => {
                ty!(*tag);
                let edges = arms
                    .iter()
                    .map(|(_, label, args)| (label, args))
                    .chain(default.iter().map(|(label, args)| (label, args)));
                for (label, args) in edges {
                    let Some(params) = self.block_params(label, insts) else {
                        continue;
                    };
                    if args.len() != params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "Switch(arm)".to_string(),
                                expected: params.len(),
                                got: args.len(),
                            },
                        });
                        continue;
                    }
                    for (i, (arg, param)) in args.iter().zip(&params).enumerate() {
                        let param_ty = ty!(*param);
                        let arg_ty = ty!(*arg);
                        self.assert_match(
                            pc,
                            span,
                            "Switch(arm)",
                            &format!("arg[{i}]"),
                            param_ty,
                            arg_ty,
                            errors,
                        );
                    }
                }
            }

            InstKind::Jump { label, args } => {
                if let Some(params) = self.block_params(label, insts) {
                    if args.len() != params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "Jump".to_string(),
                                expected: params.len(),
                                got: args.len(),
                            },
                        });
                    } else {
                        for (i, (arg, param)) in args.iter().zip(&params).enumerate() {
                            let param_ty = ty!(*param);
                            let arg_ty = ty!(*arg);
                            self.assert_match(
                                pc,
                                span,
                                "Jump",
                                &format!("arg[{i}]"),
                                param_ty,
                                arg_ty,
                                errors,
                            );
                        }
                    }
                }
            }

            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            }
            | InstKind::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                let name = match kind {
                    InstKind::Diamond { .. } => "Diamond",
                    _ => "JumpIf",
                };
                let cond_ty = ty!(*cond);
                self.assert_match(pc, span, name, "cond", &Ty::Bool, cond_ty, errors);

                let edges = [
                    BranchEdge {
                        side: "then",
                        label: *then_label,
                        args: then_args,
                    },
                    BranchEdge {
                        side: "else",
                        label: *else_label,
                        args: else_args,
                    },
                ];
                for BranchEdge { side, label, args } in edges {
                    if let InstKind::Diamond { join, .. } = kind
                        && !crate::ir::reaches(insts, label, *join)
                    {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::DiamondArmMissesJoin { side, join: *join },
                        });
                    }

                    let Some(params) = self.block_params(&label, insts) else {
                        continue;
                    };
                    let edge = format!("{name}({side})");
                    if args.len() != params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: edge,
                                expected: params.len(),
                                got: args.len(),
                            },
                        });
                        continue;
                    }
                    for (i, (arg, param)) in args.iter().zip(&params).enumerate() {
                        let param_ty = ty!(*param);
                        let arg_ty = ty!(*arg);
                        self.assert_match(
                            pc,
                            span,
                            &edge,
                            &format!("arg[{i}]"),
                            param_ty,
                            arg_ty,
                            errors,
                        );
                    }
                }
            }

            InstKind::Return { value, order } => {
                let value_ty = ty!(*value);
                // `!` has no value, so it satisfies any slot (RFC-0038).
                if let Some(declared) = &self.declared_ret
                    && !matches!(value_ty, Ty::Never)
                {
                    self.assert_match(pc, span, "Return", "value", declared, value_ty, errors);
                }
                if let Some(o) = order {
                    self.expect_order(*o, vt, span, pc, errors);
                }
            }
        }
    }
}

/// The mutability and element type of a `&[T]` or `&mut [T]`.
/// The mutability of the run an `AsSlice` hands back: a container's
/// elements (RFC-0047) or a `String`'s bytes (RFC-0062).
fn run_of(ty: &Ty) -> Option<Mutability> {
    let Ty::Ref(mutability, target) = ty else {
        return None;
    };
    matches!(target.ty, Ty::Slice(_) | Ty::Str).then_some(*mutability)
}

fn slice_of(ty: &Ty) -> Option<(Mutability, &Ty)> {
    let Ty::Ref(mutability, target) = ty else {
        return None;
    };
    let Ty::Slice(element) = &target.ty else {
        return None;
    };
    Some((*mutability, element))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, MirBody, MirModule};
    use acvus_ast::Literal;
    use acvus_utils::LocalFactory;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst { span: span(), kind }
    }

    fn make_module(insts: Vec<Inst>, val_types: FxHashMap<ValueId, Ty>) -> MirModule {
        MirModule {
            main: MirBody {
                demoted_diamonds: Default::default(),
                insts,
                val_types,
                params: Vec::new(),
                captures: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 10,
                order_param: None,
                task: crate::ty::Task::Sync,
            },
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
        }
    }

    /// A module whose `main` builds one closure of type `|| -> declared` and
    /// whose closure body returns a value of type `returned`.
    fn make_closure_module(declared: Ty, returned: Ty) -> MirModule {
        let label = Label(0);

        let mut main_types = FxHashMap::default();
        let mut main_factory = LocalFactory::<ValueId>::new();
        let closure = main_factory.next();
        main_types.insert(
            closure,
            Ty::Fn {
                params: Vec::new(),
                ret: Box::new(declared),
                captures: Vec::new(),
                effect: crate::ty::Effect::PURE.into(),
            },
        );
        let mut module = make_module(
            vec![inst(InstKind::MakeClosure {
                dst: closure,
                body: label,
                captures: Vec::new(),
            })],
            main_types,
        );
        module.main.val_factory = main_factory;

        let mut body_types = FxHashMap::default();
        let mut body_factory = LocalFactory::<ValueId>::new();
        let result = body_factory.next();
        body_types.insert(result, returned);
        let body = MirBody {
            insts: vec![inst(InstKind::Return {
                value: result,
                order: None,
            })],
            val_types: body_types,
            val_factory: body_factory,
            ..MirBody::new()
        };
        module.closures.insert(label, body);
        module
    }

    fn array_of_i64() -> Ty {
        Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3))
    }

    #[test]
    fn a_body_returning_what_it_declares_is_accepted() {
        let module = make_closure_module(array_of_i64(), array_of_i64());
        assert!(check_types(&module).is_empty());
    }

    #[test]
    fn a_body_declaring_a_value_returning_a_reference_is_refused() {
        let module = make_closure_module(
            array_of_i64(),
            Ty::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(array_of_i64())),
            ),
        );
        let errors = check_types(&module);
        let [
            ValidationError {
                kind:
                    ValidationErrorKind::TypeMismatch {
                        inst_name,
                        expected,
                        actual,
                        ..
                    },
                ..
            },
        ] = errors.as_slice()
        else {
            panic!("expected one type mismatch, got {errors:?}");
        };
        assert_eq!(inst_name, "Return");
        assert_eq!(*expected, array_of_i64());
        assert!(matches!(actual, Ty::Ref(..)), "{actual:?}");
    }

    #[test]
    fn a_body_returning_a_diverging_value_is_accepted() {
        let module = make_closure_module(array_of_i64(), Ty::Never);
        assert!(check_types(&module).is_empty());
    }

    #[test]
    fn a_body_declaring_never_accepts_the_value_it_returns() {
        let module = make_closure_module(Ty::Never, Ty::I64);
        assert!(check_types(&module).is_empty());
    }

    #[test]
    fn const_type_matches() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::I64);
        let module = make_module(
            vec![inst(InstKind::Const {
                dst: v0,
                value: Literal::Int(42),
            })],
            vt,
        );
        let errors = check_types(&module);
        assert!(errors.is_empty());
    }

    /// `types_match` enumerates the primitives by hand, so a type that is
    /// not in that list does not match itself. `&str` is the newest one and
    /// this is what holds its arm there.
    #[test]
    fn a_str_constant_matches_its_own_type() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, str_ty());
        let module = make_module(
            vec![inst(InstKind::ConstStr {
                dst: v0,
                text: "abc".to_string(),
            })],
            vt,
        );
        let errors = check_types(&module);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn a_str_constant_at_a_string_destination_is_caught() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::String);
        let module = make_module(
            vec![inst(InstKind::ConstStr {
                dst: v0,
                text: "abc".to_string(),
            })],
            vt,
        );
        let errors = check_types(&module);
        assert!(!errors.is_empty(), "a `&str` constant is not a `String`");
    }

    #[test]
    fn const_type_mismatch() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::String); // wrong: Literal::Int should be Int
        let module = make_module(
            vec![inst(InstKind::Const {
                dst: v0,
                value: Literal::Int(42),
            })],
            vt,
        );
        let errors = check_types(&module);
        assert!(!errors.is_empty(), "type mismatch should be caught");
    }

    #[test]
    fn binop_type_mismatch() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::I64);
        vt.insert(v1, Ty::String); // mismatch
        vt.insert(v2, Ty::I64);
        let module = make_module(
            vec![inst(InstKind::BinOp {
                dst: v2,
                op: acvus_ast::BinOp::Add,
                left: v0,
                right: v1,
            })],
            vt,
        );
        let errors = check_types(&module);
        assert!(!errors.is_empty(), "BinOp type mismatch should be caught");
    }

    #[test]
    fn make_tuple_arity_mismatch() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::I64);
        vt.insert(v1, Ty::Tuple(vec![Ty::I64, Ty::String])); // expects 2 elements
        let module = make_module(
            vec![inst(InstKind::MakeTuple {
                dst: v1,
                elements: vec![v0],
            })], // only 1
            vt,
        );
        let errors = check_types(&module);
        assert!(!errors.is_empty(), "tuple arity mismatch should be caught");
    }

    #[test]
    fn jump_args_type_match() {
        // BlockLabel with param, Jump with matching arg type
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::I64);
        vt.insert(v1, Ty::I64);
        let module = make_module(
            vec![
                inst(InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v1],
                }),
                inst(InstKind::Jump {
                    label: Label(0),
                    args: vec![v0],
                }),
            ],
            vt,
        );
        let errors = check_types(&module);
        assert!(
            errors.is_empty(),
            "matching jump arg types should pass: {errors:?}"
        );
    }

    fn diamond_module(else_rejoins: bool) -> MirModule {
        let mut vf = LocalFactory::<ValueId>::new();
        let cond = vf.next();
        let joined = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(cond, Ty::Bool);
        vt.insert(joined, Ty::Unit);
        let leave_else = match else_rejoins {
            true => inst(InstKind::Jump {
                label: Label(2),
                args: vec![],
            }),
            false => inst(InstKind::Return {
                value: joined,
                order: None,
            }),
        };
        make_module(
            vec![
                inst(InstKind::Diamond {
                    cond,
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                    join: Label(2),
                }),
                inst(InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                }),
                inst(InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                }),
                leave_else,
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                }),
                inst(InstKind::Return {
                    value: joined,
                    order: None,
                }),
            ],
            vt,
        )
    }

    fn demoted_module(marked: bool) -> MirModule {
        let mut vf = LocalFactory::<ValueId>::new();
        let cond = vf.next();
        let joined = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(cond, Ty::Bool);
        vt.insert(joined, Ty::Unit);
        let mut module = make_module(
            vec![
                inst(InstKind::JumpIf {
                    cond,
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                }),
                inst(InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                }),
                inst(InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                }),
                inst(InstKind::Return {
                    value: joined,
                    order: None,
                }),
            ],
            vt,
        );
        if marked {
            module.main.demoted_diamonds.insert(crate::cfg::ENTRY_LABEL);
        }
        module
    }

    fn meets_again_refusals(module: &MirModule) -> usize {
        check_types(module)
            .iter()
            .filter(|error| {
                matches!(
                    error.kind,
                    ValidationErrorKind::DemotedDiamondMeetsAgain { join: Label(2) }
                )
            })
            .count()
    }

    #[test]
    fn a_demoted_branch_whose_arms_meet_again_is_refused() {
        assert_eq!(meets_again_refusals(&demoted_module(true)), 1);
    }

    #[test]
    fn the_same_branch_with_no_demotion_behind_it_is_accepted() {
        let module = demoted_module(false);
        assert_eq!(meets_again_refusals(&module), 0);
        assert!(
            check_types(&module).is_empty(),
            "{:?}",
            check_types(&module)
        );
    }

    #[test]
    fn a_diamond_whose_arms_both_reach_the_join_is_accepted() {
        let errors = check_types(&diamond_module(true));
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn a_diamond_whose_arm_does_not_reach_the_join_is_refused() {
        let errors = check_types(&diamond_module(false));
        let refusals: Vec<&ValidationErrorKind> = errors
            .iter()
            .map(|error| &error.kind)
            .filter(|kind| {
                matches!(
                    kind,
                    ValidationErrorKind::DiamondArmMissesJoin { side: "else", .. }
                )
            })
            .collect();
        assert_eq!(refusals.len(), 1, "{errors:?}");
    }
}
