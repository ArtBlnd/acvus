//! MIR type-verification pass (error-collecting variant).
//!
//! Walks every instruction in a [`MirModule`] and checks that the types
//! recorded in `val_types` are exactly consistent with what each instruction
//! expects.  Any mismatch is collected as a [`ValidationError`] instead of
//! panicking.
//!
//! Design:
//! - Every value has a recorded type in `val_types`; a value with none is a
//!   refusal of its own (`MissingType`), never a match.
//! - A cast is a pure `FunctionCall` of the cast ExternFn; no instruction
//!   changes a value's type in place.
//! - Generic variance is invariant: inner types must match recursively.

use crate::ir::{
    Callee, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValOrigin, ValueId,
};
use crate::ir::{ExternImpl, ForSource, IndexMode};
use crate::ir::{known_effect, known_len};
use crate::ty::{CastTy, Concrete, IntTy, Mutability, Ty, TyKind, TypeArg, Types, copies};
use crate::validate::move_check::is_move_only;
use acvus_ast::{Literal, Span};

use crate::ir::{BinOp, UnaryOp};
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
            | ValidationErrorKind::LentToCall { labels, .. }
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
    /// A `for` that defines the trip count on its exit edge fills the exit
    /// block's first parameter, and another edge into that block would
    /// define it too (RFC-0057 rule 9). `entries` counts every edge into
    /// the block, the exit edge among them.
    TripBesideExitEdge {
        exit: Label,
        entries: usize,
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
    /// The stage chain of the `For` ending `header` is not RFC-0089
    /// rule 1's.
    StageShape {
        header: Label,
        fault: crate::analysis::loop_deps::ShapeFault,
    },
    /// A dependence cycle through `tokens` lies in several stages of the
    /// `For` ending `header` (RFC-0089 rule 3).
    CycleCrossesBoundary {
        header: Label,
        tokens: Vec<crate::analysis::loop_deps::Token>,
        stages: Vec<usize>,
    },
    /// Stage `stage` reads `token`'s state, and that token's cycle lies in
    /// the later stage `cycle_stage` (RFC-0089 rules 3 and 6).
    StateReadBeforeItsCycle {
        header: Label,
        token: crate::analysis::loop_deps::Token,
        stage: usize,
        cycle_stage: usize,
    },
    /// RFC-0089 rule 5.
    WorkAheadOfExit {
        header: Label,
        stage: usize,
        exit_stage: usize,
        held_back: crate::analysis::loop_deps::HeldBack,
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
    /// A place taken out for a call touched before the call's restore puts
    /// it back (RFC-0041). `labels` names the argument it was lent to the
    /// call as.
    LentToCall {
        storage: Option<ValOrigin>,
        touch: ConflictTouch,
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
    /// The body's result, or what it wrote into storage outside it, holds a
    /// loan on a local, whose storage the run is about to leave (RFC-0079
    /// rule 9).
    ReferenceToLocalLeavesBody {
        storage: Option<ValOrigin>,
    },
    /// An output of the body holds a loan on an input its function type's
    /// flows do not name (RFC-0079 rule 5): the checker's inference of the
    /// flows missed what the body does.
    FlowNotStated {
        to: crate::ty::FlowEnd,
        from: crate::ty::FlowEnd,
    },
    /// A parameter's register is storage the call fills before the body's
    /// first instruction. Any other register is a definition, and
    /// `inst_info::uses` does not count a storage target as its read, so
    /// `dce` sweeps the instruction that made it.
    NotAParameter {
        value_id: u32,
    },
    /// A `MakeClosure` names a body its module does not hold in
    /// `MirModule::closures`, the only place `prepare` looks it up.
    UnheldClosure {
        body: Label,
    },
    /// An `Index` or `IndexSet` marked `Proven` whose bound the interval
    /// domain does not derive (RFC-0047 rule 7).
    UnprovenBound,
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
/// The types an instruction is checked against are interned into the
/// module's own table.
pub fn check_types(module: &mut MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let MirModule {
        main,
        closures,
        ret,
        types,
        ..
    } = module;
    let declared = declared_closure_returns(types, main, closures);

    let mut ctx = CheckCtx::new("main".to_string(), Some(*ret), types);
    ctx.check_body(main, closures, &mut errors);

    for (label, closure) in closures.iter() {
        let name = format!("closure({:?})", label);
        let mut ctx = CheckCtx::new(name, declared.get(label).copied(), types);
        ctx.check_body(closure, closures, &mut errors);
    }

    errors
}

/// What each closure body declares it returns: the `ret` of the function
/// type that the `MakeClosure` building it gives its value. A closure whose
/// `MakeClosure` is gone - inlined away, or dead - declares nothing.
fn declared_closure_returns(
    types: &Types<Concrete>,
    main: &MirBody,
    closures: &FxHashMap<Label, MirBody>,
) -> FxHashMap<Label, Ty> {
    std::iter::once(main)
        .chain(closures.values())
        .flat_map(|maker| {
            maker.insts.iter().filter_map(|inst| {
                let InstKind::MakeClosure {
                    dst, body: made, ..
                } = &inst.kind
                else {
                    return None;
                };
                let TyKind::Fn { ret, .. } = types.kind(maker.ty_of(*dst)) else {
                    return None;
                };
                Some((*made, *ret))
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
// Fitting a slot
// ---------------------------------------------------------------------------

/// Whether a value of type `actual` fits a slot of type `expected`. Types
/// are hash-consed, so two are the same exactly when their ids are equal; a
/// value of the bottom type fits any slot (types §4).
fn types_match(types: &Types<Concrete>, expected: Ty, actual: Ty) -> bool {
    expected == actual || matches!(types.kind(actual), TyKind::Never)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The type a literal constant has, where the literal alone decides it. A
/// list literal's type is the one recorded for the constant's value.
fn literal_ty(types: &mut Types<Concrete>, lit: &Literal) -> Option<Ty> {
    let kind = match lit {
        Literal::String(_) => TyKind::String,
        Literal::Int(_) => TyKind::Int(IntTy::I64),
        Literal::IntOf(n) => TyKind::Int(n.width.into()),
        Literal::Float(_) => TyKind::Float,
        Literal::Char(_) => TyKind::Char,
        Literal::Bytes(bytes) => {
            let byte = types.intern(TyKind::Int(IntTy::U8));
            TyKind::Array(byte, crate::ty::LenTerm::Known(bytes.len()))
        }
        Literal::Bool(_) => TyKind::Bool,
        Literal::List(_) => return None,
        Literal::Unit => TyKind::Unit,
    };
    Some(types.intern(kind))
}

/// When the lowerer reuses a ValueId for both a collection and its iterated
/// element (e.g. pattern-match iteration over a List), the val_types map may
/// record the collection type rather than the element type.  This helper
/// unwraps one level of Array so pattern-match checks don't produce
/// false positives.
fn unwrap_element_ty(types: &Types<Concrete>, ty: Ty) -> Ty {
    match types.kind(ty) {
        TyKind::Array(inner, _) => *inner,
        _ => ty,
    }
}

fn as_array_inner(types: &Types<Concrete>, ty: Ty) -> Option<Ty> {
    match types.kind(ty) {
        TyKind::Array(inner, _) => Some(*inner),
        _ => None,
    }
}

/// The types every body is checked against, interned once.
#[derive(Clone, Copy)]
struct Common {
    bool: Ty,
    u64: Ty,
    i64: Ty,
    string: Ty,
    order: Ty,
    /// `&str`, a string constant's type.
    str_ref: Ty,
    /// `&mut String`, what a `StringAppend` appends into.
    string_mut: Ty,
}

impl Common {
    fn of(types: &mut Types<Concrete>) -> Self {
        let str_ty = types.intern(TyKind::Str);
        let string = types.intern(TyKind::String);
        Common {
            bool: types.intern(TyKind::Bool),
            u64: types.intern(TyKind::Int(IntTy::U64)),
            i64: types.intern(TyKind::Int(IntTy::I64)),
            string,
            order: types.intern(TyKind::Order),
            str_ref: types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(str_ty))),
            string_mut: types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(string))),
        }
    }
}

/// How a reference of `mutability` is written before its target.
fn ref_prefix(mutability: Mutability) -> &'static str {
    match mutability {
        Mutability::Shared => "&",
        Mutability::Mut => "&mut ",
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

struct CheckCtx<'t> {
    scope_name: String,
    /// label -> index in `insts` (for Jump target block param lookup)
    label_map: FxHashMap<Label, usize>,
    declared_ret: Option<Ty>,
    types: &'t mut Types<Concrete>,
    common: Common,
}

impl<'t> CheckCtx<'t> {
    fn new(scope_name: String, declared_ret: Option<Ty>, types: &'t mut Types<Concrete>) -> Self {
        let common = Common::of(types);
        Self {
            scope_name,
            label_map: FxHashMap::default(),
            declared_ret,
            types,
            common,
        }
    }

    fn kind(&self, ty: Ty) -> &TyKind<Concrete> {
        self.types.kind(ty)
    }

    fn lent(&mut self, mutability: Mutability, ty: Ty) -> Ty {
        self.types.intern(TyKind::Ref(mutability, TypeArg::Uniform(ty)))
    }

    fn check_body(
        &mut self,
        body: &MirBody,
        closures: &FxHashMap<Label, MirBody>,
        errors: &mut Vec<ValidationError>,
    ) {
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
            if let InstKind::Ref { target, .. }
            | InstKind::Take { target, .. }
            | InstKind::Assign { target, .. } = &inst.kind
                && let RefTarget::Param(slot) = target
                && !body.params.iter().any(|(_, param)| param == slot)
            {
                errors.push(ValidationError {
                    scope: self.scope_name.clone(),
                    inst_index: pc,
                    span: inst.span,
                    kind: ValidationErrorKind::NotAParameter {
                        value_id: slot.to_raw() as u32,
                    },
                });
            }
            if let InstKind::MakeClosure {
                body: made,
                captures,
                ..
            } = &inst.kind
            {
                match closures.get(made) {
                    None => errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span: inst.span,
                        kind: ValidationErrorKind::UnheldClosure { body: *made },
                    }),
                    Some(closure) if closure.captures.len() != captures.len() => errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span: inst.span,
                        kind: ValidationErrorKind::ArityMismatch {
                            inst_name: "MakeClosure".to_string(),
                            expected: closure.captures.len(),
                            got: captures.len(),
                        },
                    }),
                    // Each captured value against the register of the body
                    // it is bound to.
                    Some(closure) => {
                        for (i, (captured, (_, register))) in captures.iter().zip(&closure.captures).enumerate() {
                            let (Some(captured_ty), Some(register_ty)) = (
                                self.ty_of(*captured, &body.val_types, inst.span, pc, errors),
                                self.ty_of(*register, &closure.val_types, inst.span, pc, errors),
                            ) else {
                                continue;
                            };
                            self.assert_match(
                                pc,
                                inst.span,
                                "MakeClosure",
                                &format!("capture[{i}]"),
                                register_ty,
                                captured_ty,
                                errors,
                            );
                        }
                    }
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

    /// Get the type of a ValueId.  Pushes `MissingType` if absent.
    fn ty_of(
        &self,
        id: ValueId,
        val_types: &FxHashMap<ValueId, Ty>,
        span: Span,
        pc: usize,
        errors: &mut Vec<ValidationError>,
    ) -> Option<Ty> {
        match val_types.get(&id) {
            Some(ty) => Some(*ty),
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
            self.assert_match(pc, span, "Order", "order", self.common.order, ty, errors);
        }
    }

    /// A call takes an `Order` exactly when its callee's effect is not Pure.
    /// A callee reached through a reference is called as the function it
    /// points at.
    fn expect_order_edge(
        &self,
        inst_name: &str,
        callee_ty: Ty,
        before: Option<ValueId>,
        val_types: &FxHashMap<ValueId, Ty>,
        span: Span,
        pc: usize,
        errors: &mut Vec<ValidationError>,
    ) {
        let Some((_, ret, effect)) = self.fn_parts(callee_ty) else {
            self.invalid(pc, span, inst_name, "a function type", callee_ty, errors);
            return;
        };
        let _ = ret;
        let pure = known_effect(effect).reissue() == crate::ty::Reissue::Pure;
        match (pure, before) {
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

    /// The type at `path` below what the reference `r` names, with the
    /// reference's mutability. A payload of a `Result` or an enum is not
    /// known from the path, and its type is `recorded`, the one recorded for
    /// the instruction's value.
    #[allow(clippy::too_many_arguments)]
    fn through(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        r: ValueId,
        path: &[PathSeg],
        recorded: Ty,
        vt: &FxHashMap<ValueId, Ty>,
        errors: &mut Vec<ValidationError>,
    ) -> Option<(Mutability, Ty)> {
        let r_ty = self.ty_of(r, vt, span, pc, errors)?;
        let TyKind::Ref(m, inner) = self.kind(r_ty) else {
            self.invalid(pc, span, inst_name, "Ref", r_ty, errors);
            return None;
        };
        let (m, inner) = (*m, inner.ty());
        let at = self.walk(pc, span, inst_name, inner, path, recorded, errors)?;
        Some((m, at))
    }

    /// The type at `path` below the storage `slot` names directly.
    #[allow(clippy::too_many_arguments)]
    fn in_storage(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        slot: ValueId,
        path: &[PathSeg],
        recorded: Ty,
        vt: &FxHashMap<ValueId, Ty>,
        errors: &mut Vec<ValidationError>,
    ) -> Option<Ty> {
        let root = self.ty_of(slot, vt, span, pc, errors)?;
        self.walk(pc, span, inst_name, root, path, recorded, errors)
    }

    /// The type `path` reaches below `root`. A payload of a `Result` or an
    /// enum names no one variant, so the part it reaches is the one recorded
    /// for the instruction's value, `recorded`.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        root: Ty,
        path: &[PathSeg],
        recorded: Ty,
        errors: &mut Vec<ValidationError>,
    ) -> Option<Ty> {
        let mut at = root;
        for seg in path {
            let next = match (seg, self.kind(at)) {
                (PathSeg::Field(field), TyKind::Object(object)) => object
                    .fields()
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, ty)| *ty),
                (PathSeg::Index(_), TyKind::Array(elem, _)) => Some(*elem),
                (PathSeg::Index(i), TyKind::Tuple(elems)) => elems.get(*i).copied(),
                (PathSeg::Payload, TyKind::Option(payload)) => Some(*payload),
                (PathSeg::Payload, TyKind::Result(..) | TyKind::Enum { .. }) => return Some(recorded),
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
                        actual: at,
                    },
                });
                return None;
            };
            at = next;
        }
        Some(at)
    }

    /// A type whose head is not the one the instruction is built on.
    fn invalid(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        expected_constructor: &str,
        actual: Ty,
        errors: &mut Vec<ValidationError>,
    ) {
        errors.push(ValidationError {
            scope: self.scope_name.clone(),
            inst_index: pc,
            span,
            kind: ValidationErrorKind::InvalidConstructor {
                inst_name: inst_name.to_string(),
                expected_constructor: expected_constructor.to_string(),
                actual,
            },
        });
    }

    /// The parameters, result and effect of the function type `ty`, or of
    /// the one it references.
    fn fn_parts(&self, ty: Ty) -> Option<(Vec<Ty>, Ty, &crate::ty::EffectTerm<Concrete>)> {
        let fn_ty = match self.kind(ty) {
            TyKind::Ref(_, target) => target.ty(),
            _ => ty,
        };
        match self.kind(fn_ty) {
            TyKind::Fn {
                params, ret, effect, ..
            } => Some((params.to_vec(), *ret, effect)),
            _ => None,
        }
    }

    fn assert_match(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        desc: &str,
        expected: Ty,
        actual: Ty,
        errors: &mut Vec<ValidationError>,
    ) -> bool {
        if !types_match(self.types, expected, actual) {
            errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: pc,
                span,
                kind: ValidationErrorKind::TypeMismatch {
                    inst_name: inst_name.to_string(),
                    desc: desc.to_string(),
                    expected,
                    actual,
                },
            });
            false
        } else {
            true
        }
    }

    fn assert_lends_structural(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        operand_ty: Ty,
        errors: &mut Vec<ValidationError>,
    ) {
        let lends_structural = matches!(
            self.kind(operand_ty),
            TyKind::Ref(_, lent) if crate::structural::components(self.types, lent.ty()).is_some()
        );
        if !lends_structural {
            errors.push(ValidationError {
                scope: self.scope_name.clone(),
                inst_index: pc,
                span,
                kind: ValidationErrorKind::InvalidConstructor {
                    inst_name: inst_name.to_string(),
                    expected_constructor: "a reference to a structural type".to_string(),
                    actual: operand_ty,
                },
            });
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

    /// Text a string instruction reads: a `String`, or a reference to a
    /// `String` or a `str`.
    fn is_text(&self, ty: Ty) -> bool {
        match self.kind(ty) {
            TyKind::String => true,
            TyKind::Ref(_, inner) => matches!(self.kind(inner.ty()), TyKind::String | TyKind::Str),
            _ => false,
        }
    }

    /// The type of the field `field` of the object `object`, then of each of
    /// `rest` below it; `None` where a field is not there, which the
    /// instruction's own recorded type answers for.
    fn field_path(&self, object: Ty, field: acvus_utils::Astr, rest: &[acvus_utils::Astr]) -> Option<Ty> {
        let mut at = object;
        for name in std::iter::once(&field).chain(rest) {
            let TyKind::Object(fields) = self.kind(at) else {
                return None;
            };
            at = fields.fields().iter().find(|(held, _)| held == name)?.1;
        }
        Some(at)
    }

    /// A call's arguments against its callee's parameters.
    #[allow(clippy::too_many_arguments)]
    fn check_call_args(
        &self,
        pc: usize,
        span: Span,
        inst_name: &str,
        params: &[Ty],
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
            let Some(arg_ty) = self.ty_of(*arg, vt, span, pc, errors) else {
                continue;
            };
            self.assert_match(pc, span, inst_name, &format!("arg[{i}]"), *param, arg_ty, errors);
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
        &mut self,
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
                r#impl: ExternImpl { .. },
            } => {
                let container_ty = ty!(*container);
                let TyKind::Ref(held, _) = *self.kind(container_ty) else {
                    self.invalid(pc, span, "AsSlice", "Ref", container_ty, errors);
                    return;
                };
                if *mutability == Mutability::Mut && held != Mutability::Mut {
                    self.invalid(pc, span, "AsSlice", "Ref(Mut)", container_ty, errors);
                }
                let dst_ty = ty!(*dst);
                if run_of(self.types, dst_ty) != Some(*mutability) {
                    self.invalid(
                        pc,
                        span,
                        "AsSlice",
                        &format!("{}[_] or &str", ref_prefix(*mutability)),
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
                bound: _,
            } => {
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "Index", "index", self.common.u64, index_ty, errors);
                let slice_ty = ty!(*slice);
                let Some((mutability, element)) = slice_of(self.types, slice_ty) else {
                    self.invalid(pc, span, "Index", "Ref(_, Slice)", slice_ty, errors);
                    return;
                };
                let dst_ty = ty!(*dst);
                match mode {
                    IndexMode::Copy => {
                        self.assert_match(pc, span, "Index", "dst", element, dst_ty, errors);
                        if is_move_only(self.types, element) {
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
                        let expected = self.lent(mutability, element);
                        self.assert_match(pc, span, "Index", "dst", expected, dst_ty, errors);
                    }
                }
            }
            InstKind::IndexSet {
                slice,
                index,
                value,
                bound: _,
            } => {
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "IndexSet", "index", self.common.u64, index_ty, errors);
                let slice_ty = ty!(*slice);
                let Some((Mutability::Mut, element)) = slice_of(self.types, slice_ty) else {
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
                    if !matches!(self.kind(dst_ty), TyKind::Int(_)) {
                        self.assert_match(pc, span, "Const", "dst", self.common.i64, dst_ty, errors);
                    }
                } else if let Some(lit_ty) = literal_ty(self.types, value) {
                    self.assert_match(pc, span, "Const", "dst", lit_ty, dst_ty, errors);
                }
            }

            // === ConstStr ===
            InstKind::ConstStr { dst, .. } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "ConstStr", "dst", self.common.str_ref, dst_ty, errors);
            }

            // === Constructors ===
            InstKind::StringClone { dst, src } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringClone", "dst", self.common.string, dst_ty, errors);
                let src_ty = ty!(*src);
                let is_string = match self.kind(src_ty) {
                    TyKind::String => true,
                    TyKind::Ref(_, inner) => matches!(self.kind(inner.ty()), TyKind::String),
                    _ => false,
                };
                if !is_string {
                    self.invalid(pc, span, "StringClone", "String or &String", src_ty, errors);
                }
            }
            InstKind::StringEq { dst, a, b } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringEq", "dst", self.common.bool, dst_ty, errors);
                for operand in [a, b] {
                    let operand_ty = ty!(*operand);
                    let is_lent_text = matches!(
                        self.kind(operand_ty),
                        TyKind::Ref(_, inner) if matches!(self.kind(inner.ty()), TyKind::String | TyKind::Str)
                    );
                    if !is_lent_text {
                        self.invalid(pc, span, "StringEq", "&String or &str", operand_ty, errors);
                    }
                }
            }
            InstKind::StructuralEq { dst, a, b, .. } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StructuralEq", "dst", self.common.bool, dst_ty, errors);
                let a_ty = ty!(*a);
                let b_ty = ty!(*b);
                for operand_ty in [a_ty, b_ty] {
                    self.assert_lends_structural(pc, span, "StructuralEq", operand_ty, errors);
                }
                if let (TyKind::Ref(_, a_lent), TyKind::Ref(_, b_lent)) = (self.kind(a_ty), self.kind(b_ty)) {
                    let (a_lent, b_lent) = (a_lent.ty(), b_lent.ty());
                    self.assert_match(pc, span, "StructuralEq", "b", a_lent, b_lent, errors);
                }
            }
            InstKind::StructuralClone { dst, src, .. } => {
                let src_ty = ty!(*src);
                self.assert_lends_structural(pc, span, "StructuralClone", src_ty, errors);
                if let TyKind::Ref(_, lent) = self.kind(src_ty) {
                    let lent = lent.ty();
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "StructuralClone", "dst", lent, dst_ty, errors);
                }
            }
            InstKind::StringConcat { dst, parts } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringConcat", "dst", self.common.string, dst_ty, errors);
                for part in parts {
                    let part_ty = ty!(*part);
                    if !self.is_text(part_ty) {
                        self.invalid(pc, span, "StringConcat", "String, &String or &str", part_ty, errors);
                    }
                }
            }
            InstKind::StringAppend { target, part } => {
                let target_ty = ty!(*target);
                self.assert_match(
                    pc,
                    span,
                    "StringAppend",
                    "target",
                    self.common.string_mut,
                    target_ty,
                    errors,
                );
                let part_ty = ty!(*part);
                if !self.is_text(part_ty) {
                    self.invalid(pc, span, "StringAppend", "String, &String or &str", part_ty, errors);
                }
            }
            InstKind::ArrayBegin { dst, .. } => {
                let dst_ty = ty!(*dst);
                match self.kind(dst_ty) {
                    TyKind::Array(_, len) if known_len(len) == 0 => {}
                    _ => self.invalid(pc, span, "ArrayBegin", "Array of length 0", dst_ty, errors),
                }
            }
            InstKind::ArrayPush { dst, array, value } => {
                let dst_ty = ty!(*dst);
                let array_ty = ty!(*array);
                let value_ty = ty!(*value);
                match (self.kind(dst_ty), self.kind(array_ty)) {
                    (TyKind::Array(inner, len), TyKind::Array(held, held_len))
                        if known_len(len) == known_len(held_len) + 1 =>
                    {
                        let (inner, held) = (*inner, *held);
                        self.assert_match(pc, span, "ArrayPush", "array", inner, held, errors);
                        self.assert_match(pc, span, "ArrayPush", "value", inner, value_ty, errors);
                    }
                    _ => {
                        let expected = format!("Array one longer than {array_ty:?}");
                        self.invalid(pc, span, "ArrayPush", &expected, dst_ty, errors)
                    }
                }
            }

            InstKind::MakeObject { dst, fields } => {
                let dst_ty = ty!(*dst);
                let TyKind::Object(object) = self.kind(dst_ty) else {
                    self.invalid(pc, span, "MakeObject", "Object", dst_ty, errors);
                    return;
                };
                let declared: Vec<(acvus_utils::Astr, Ty)> = object.fields().to_vec();
                for (key, val) in fields {
                    if let Some((_, expected_field_ty)) = declared.iter().find(|(name, _)| name == key) {
                        let val_ty = ty!(*val);
                        self.assert_match(pc, span, "MakeObject", "field", *expected_field_ty, val_ty, errors);
                    }
                }
            }

            InstKind::MakeTuple { dst, elements } => {
                let dst_ty = ty!(*dst);
                let TyKind::Tuple(elem_tys) = self.kind(dst_ty) else {
                    self.invalid(pc, span, "MakeTuple", "Tuple", dst_ty, errors);
                    return;
                };
                let elem_tys: Vec<Ty> = elem_tys.to_vec();
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
                        self.assert_match(pc, span, "MakeTuple", &format!("element[{i}]"), expected, elem_ty, errors);
                    }
                }
            }

            // A function type states no capture list (D-c): the captures are
            // checked against the body it makes, in `check_body`.
            InstKind::MakeClosure { dst, .. } => {
                let dst_ty = ty!(*dst);
                if !matches!(self.kind(dst_ty), TyKind::Fn { .. }) {
                    self.invalid(pc, span, "MakeClosure", "Fn", dst_ty, errors);
                }
            }

            InstKind::MakeVariant { dst, tag, payload } => {
                let dst_ty = ty!(*dst);
                match self.kind(dst_ty) {
                    TyKind::Enum { variants, .. } => {
                        // Tag not found in type - open enum, skip
                        let Some((_, variant_payload_ty)) = variants.iter().find(|(name, _)| name == tag) else {
                            return;
                        };
                        match (*variant_payload_ty, payload) {
                            (Some(expected), Some(val)) => {
                                let val_ty = ty!(*val);
                                self.assert_match(pc, span, "MakeVariant", "payload", expected, val_ty, errors);
                            }
                            (None, None) => {}
                            (Some(_), None) => errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::ArityMismatch {
                                    inst_name: "MakeVariant".to_string(),
                                    expected: 1,
                                    got: 0,
                                },
                            }),
                            (None, Some(_)) => errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::ArityMismatch {
                                    inst_name: "MakeVariant".to_string(),
                                    expected: 0,
                                    got: 1,
                                },
                            }),
                        }
                    }
                    TyKind::Option(inner) => {
                        let inner = *inner;
                        if let Some(val) = payload {
                            let val_ty = ty!(*val);
                            self.assert_match(pc, span, "MakeVariant", "Option payload", inner, val_ty, errors);
                        }
                    }
                    _ => {}
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
                let bool_ty = self.common.bool;

                if matches!(op, BinOp::Min | BinOp::Max) {
                    if !matches!(self.kind(left_ty), TyKind::Int(_)) {
                        self.invalid(pc, span, "BinOp(min/max)", "Int", left_ty, errors);
                    }
                    self.assert_match(pc, span, "BinOp(min/max)", "left == right", left_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp(min/max)", "left == dst", left_ty, dst_ty, errors);
                } else if binop_is_logical(*op) {
                    self.assert_match(pc, span, "BinOp(logical)", "left", bool_ty, left_ty, errors);
                    self.assert_match(pc, span, "BinOp(logical)", "right", bool_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp(logical)", "dst", bool_ty, dst_ty, errors);
                } else if binop_returns_bool(*op) {
                    self.assert_match(pc, span, "BinOp(cmp)", "left == right", left_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp(cmp)", "dst", bool_ty, dst_ty, errors);
                } else {
                    self.assert_match(pc, span, "BinOp", "left == right", left_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp", "left == dst", left_ty, dst_ty, errors);
                }
            }

            InstKind::Check { left, right, .. } => {
                let left_ty = ty!(*left);
                let right_ty = ty!(*right);
                if !matches!(self.kind(left_ty), TyKind::Int(_)) {
                    self.invalid(pc, span, "Check", "Int", left_ty, errors);
                }
                self.assert_match(pc, span, "Check", "left == right", left_ty, right_ty, errors);
            }

            InstKind::CheckSteps { from, step, count } => {
                let from_ty = ty!(*from);
                let step_ty = ty!(*step);
                let count_ty = ty!(*count);
                if !matches!(self.kind(from_ty), TyKind::Int(_)) {
                    self.invalid(pc, span, "CheckSteps", "Int", from_ty, errors);
                }
                self.assert_match(pc, span, "CheckSteps", "from == step", from_ty, step_ty, errors);
                if count_ty != self.common.u64 {
                    self.invalid(pc, span, "CheckSteps", "U64", count_ty, errors);
                }
            }

            // === UnaryOp ===
            InstKind::UnaryOp { dst, op, operand } => {
                let operand_ty = ty!(*operand);
                let dst_ty = ty!(*dst);
                match op {
                    UnaryOp::Not => {
                        let bool_ty = self.common.bool;
                        self.assert_match(pc, span, "UnaryOp(Not)", "operand", bool_ty, operand_ty, errors);
                        self.assert_match(pc, span, "UnaryOp(Not)", "dst", bool_ty, dst_ty, errors);
                    }
                    UnaryOp::Neg(_) => {
                        self.assert_match(pc, span, "UnaryOp(Neg)", "operand == dst", operand_ty, dst_ty, errors);
                    }
                }
            }

            // === Cast (RFC-0049) ===
            InstKind::Cast { dst, src, to } => {
                let src_ty = ty!(*src);
                let dst_ty = ty!(*dst);
                let to_ty = self.types.intern(TyKind::from(*to));
                self.assert_match(pc, span, "Cast", "dst", to_ty, dst_ty, errors);
                if CastTy::of_kind(self.kind(src_ty)).is_none() {
                    self.invalid(pc, span, "Cast", "an integer width or f64", src_ty, errors);
                }
            }

            // === Projection ===
            InstKind::Take {
                dst, target, path, ..
            } => {
                let dst_ty = ty!(*dst);
                if let RefTarget::Through(r) = target {
                    let Some((_, at)) = self.through(pc, span, "Take", *r, path, dst_ty, vt, errors) else {
                        return;
                    };
                    self.assert_match(pc, span, "Take", "dst", at, dst_ty, errors);
                    if !copies(self.types, at) {
                        self.invalid(pc, span, "Take", "a word or a String through a reference", at, errors);
                    }
                } else if let RefTarget::Var(slot) | RefTarget::Param(slot) = target
                    && let Some(at) = self.in_storage(pc, span, "Take", *slot, path, dst_ty, vt, errors)
                {
                    self.assert_match(pc, span, "Take", "dst", at, dst_ty, errors);
                }
            }
            InstKind::Assign {
                value,
                target,
                path,
                ..
            } => {
                let val_ty = ty!(*value);
                if let RefTarget::Through(r) = target {
                    let Some((m, at)) = self.through(pc, span, "Assign", *r, path, val_ty, vt, errors) else {
                        return;
                    };
                    if m != Mutability::Mut {
                        let r_ty = ty!(*r);
                        self.invalid(pc, span, "Assign", "Ref(Mut)", r_ty, errors);
                    }
                    self.assert_match(pc, span, "Assign", "value", at, val_ty, errors);
                } else if let RefTarget::Var(slot) | RefTarget::Param(slot) = target
                    && let Some(at) = self.in_storage(pc, span, "Assign", *slot, path, val_ty, vt, errors)
                {
                    self.assert_match(pc, span, "Assign", "value", at, val_ty, errors);
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
                // What a reference names is the target of its own type.
                let recorded = match self.kind(dst_ty) {
                    TyKind::Ref(_, lent) => lent.ty(),
                    _ => {
                        self.invalid(pc, span, "Ref", &format!("{}_", ref_prefix(*mutability)), dst_ty, errors);
                        return;
                    }
                };
                if let RefTarget::Through(r) = target {
                    let Some((m, at)) = self.through(pc, span, "Ref", *r, path, recorded, vt, errors) else {
                        return;
                    };
                    if *mutability == Mutability::Mut && m != Mutability::Mut {
                        let r_ty = ty!(*r);
                        self.invalid(pc, span, "Ref", "Ref(Mut) to reborrow mutably", r_ty, errors);
                    }
                    let expected = self.lent(*mutability, at);
                    self.assert_match(pc, span, "Ref", "dst", expected, dst_ty, errors);
                    return;
                }
                if let RefTarget::Var(slot) | RefTarget::Param(slot) = target
                    && let Some(at) = self.in_storage(pc, span, "Ref", *slot, path, recorded, vt, errors)
                {
                    let expected = self.lent(*mutability, at);
                    self.assert_match(pc, span, "Ref", "dst", expected, dst_ty, errors);
                    return;
                }
                if !matches!(self.kind(dst_ty), TyKind::Ref(m, _) if m == mutability) {
                    self.invalid(pc, span, "Ref", &format!("{}_", ref_prefix(*mutability)), dst_ty, errors);
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
                let obj_ty = match self.kind(obj_ty) {
                    TyKind::Object(_) => obj_ty,
                    _ => unwrap_element_ty(self.types, obj_ty),
                };
                if !matches!(self.kind(obj_ty), TyKind::Object(_)) {
                    self.invalid(pc, span, "FieldGet", "Object", obj_ty, errors);
                    return;
                }
                if let Some(resolved) = self.field_path(obj_ty, *field, rest) {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "FieldGet", "dst", resolved, dst_ty, errors);
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
                if !matches!(self.kind(obj_ty), TyKind::Object(_)) {
                    self.invalid(pc, span, "FieldSet", "Object", obj_ty, errors);
                    return;
                }
                if let Some(resolved) = self.field_path(obj_ty, *field, rest) {
                    let val_ty = ty!(*value);
                    self.assert_match(pc, span, "FieldSet", "value", resolved, val_ty, errors);
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "FieldSet", "dst == object", obj_ty, dst_ty, errors);
            }

            InstKind::ObjectGet { dst, object, key } => {
                let obj_ty = ty!(*object);
                let obj_ty = match self.kind(obj_ty) {
                    TyKind::Object(_) => obj_ty,
                    _ => unwrap_element_ty(self.types, obj_ty),
                };
                let TyKind::Object(object) = self.kind(obj_ty) else {
                    self.invalid(pc, span, "ObjectGet", "Object", obj_ty, errors);
                    return;
                };
                if let Some((_, field_ty)) = object.fields().iter().find(|(name, _)| name == key) {
                    let field_ty = *field_ty;
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ObjectGet", "dst", field_ty, dst_ty, errors);
                }
            }

            InstKind::TupleIndex { dst, tuple, index } => {
                let tup_ty = ty!(*tuple);
                let TyKind::Tuple(elems) = self.kind(tup_ty) else {
                    self.invalid(pc, span, "TupleIndex", "Tuple", tup_ty, errors);
                    return;
                };
                match elems.get(*index).copied() {
                    Some(elem_ty) => {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "TupleIndex", "dst", elem_ty, dst_ty, errors);
                    }
                    None => {
                        let expected = elems.len();
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "TupleIndex".to_string(),
                                expected,
                                got: *index + 1,
                            },
                        });
                    }
                }
            }

            InstKind::ArrayIndex {
                dst, array: list, ..
            } => {
                let list_ty = ty!(*list);
                let Some(inner) = as_array_inner(self.types, list_ty) else {
                    self.invalid(pc, span, "ArrayIndex", "Array", list_ty, errors);
                    return;
                };
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "ArrayIndex", "dst", inner, dst_ty, errors);
            }

            // === Pattern tests (all produce Bool) ===
            InstKind::TestLiteral { dst, .. } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestLiteral", "dst", self.common.bool, dst_ty, errors);
            }

            InstKind::TestObjectKey { dst, src, .. } => {
                let src_ty = ty!(*src);
                let src_ty = match self.kind(src_ty) {
                    TyKind::Ref(_, inner) => inner.ty(),
                    _ => src_ty,
                };
                let src_ty = match self.kind(src_ty) {
                    TyKind::Object(_) => src_ty,
                    _ => unwrap_element_ty(self.types, src_ty),
                };
                if !matches!(self.kind(src_ty), TyKind::Object(_)) {
                    self.invalid(pc, span, "TestObjectKey", "Object", src_ty, errors);
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestObjectKey", "dst", self.common.bool, dst_ty, errors);
            }

            InstKind::TestVariant { dst, src, .. } => {
                let src_ty = ty!(*src);
                let src_ty = match self.kind(src_ty) {
                    TyKind::Ref(_, inner) => inner.ty(),
                    _ => src_ty,
                };
                if !matches!(
                    self.kind(src_ty),
                    TyKind::Enum { .. } | TyKind::Option(_) | TyKind::Result(..)
                ) {
                    self.invalid(pc, span, "TestVariant", "Enum/Option", src_ty, errors);
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestVariant", "dst", self.common.bool, dst_ty, errors);
            }

            InstKind::UnwrapVariant { dst, src } => {
                let src_ty = ty!(*src);
                match self.kind(src_ty) {
                    TyKind::Option(inner) => {
                        let inner = *inner;
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "UnwrapVariant", "dst", inner, dst_ty, errors);
                    }
                    // A payload of a `Result` or an enum names no one
                    // variant: its type is the one recorded for `dst`.
                    TyKind::Enum { .. } | TyKind::Result(..) => {
                        let _ = self.ty_of(*dst, vt, span, pc, errors);
                    }
                    _ => self.invalid(pc, span, "UnwrapVariant", "Enum/Option", src_ty, errors),
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
                    *callee_ty,
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
                        let Some((params, ret, _)) = self.fn_parts(closure_ty) else {
                            self.invalid(pc, span, "FunctionCall(Indirect)", "Fn", closure_ty, errors);
                            return;
                        };
                        self.check_call_args(pc, span, "FunctionCall(Indirect)", &params, args, vt, errors);
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "FunctionCall(Indirect)", "return", ret, dst_ty, errors);
                    }
                }
            }

            // === Spawn / TakeEffect ===
            InstKind::Spawn {
                dst,
                callee,
                callee_ty,
                args,
                order,
            } => {
                self.expect_order_edge("Spawn", *callee_ty, *order, vt, span, pc, errors);
                match callee {
                    Callee::Direct(_) | Callee::Extern { .. } => {
                        let dst_ty = ty!(*dst);
                        if !matches!(self.kind(dst_ty), TyKind::Handle(..)) {
                            self.invalid(pc, span, "Spawn", "Handle", dst_ty, errors);
                        }
                    }
                    Callee::Indirect(closure) => {
                        let closure_ty = ty!(*closure);
                        let Some((params, ret, _)) = self.fn_parts(closure_ty) else {
                            self.invalid(pc, span, "Spawn(Indirect)", "Fn", closure_ty, errors);
                            return;
                        };
                        self.check_call_args(pc, span, "Spawn(Indirect)", &params, args, vt, errors);
                        let expected_dst = self.types.intern(TyKind::Handle(ret));
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "Spawn(Indirect)", "dst", expected_dst, dst_ty, errors);
                    }
                }
            }

            InstKind::Merge { dst, orders } => {
                self.expect_order(*dst, vt, span, pc, errors);
                for o in orders {
                    self.expect_order(*o, vt, span, pc, errors);
                }
            }

            InstKind::TakeEffect { dst, src, order } => {
                if let Some(o) = order {
                    self.expect_order(*o, vt, span, pc, errors);
                }
                let src_ty = ty!(*src);
                let TyKind::Handle(inner) = *self.kind(src_ty) else {
                    self.invalid(pc, span, "TakeEffect", "Handle", src_ty, errors);
                    return;
                };
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TakeEffect", "dst", inner, dst_ty, errors);
            }

            // === Control flow ===
            // A `For` is the loop's condition (RFC-0057): the source decides
            // the element and the counter it hands the body, and the edges'
            // remaining arguments are checked the way a `Jump`'s are.
            InstKind::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                let body = &stages.body();
                let element = match source {
                    ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                        let slice_ty = ty!(*slice);
                        let wanted = match source {
                            ForSource::SliceMut(_) => Mutability::Mut,
                            _ => Mutability::Shared,
                        };
                        let Some((held, element)) = slice_of(self.types, slice_ty) else {
                            self.invalid(pc, span, "For", "Ref(_, Slice)", slice_ty, errors);
                            return;
                        };
                        if held != wanted {
                            self.invalid(pc, span, "For", &format!("{}[_]", ref_prefix(wanted)), slice_ty, errors);
                        }
                        self.lent(wanted, element)
                    }
                    ForSource::Array(array) => {
                        let array_ty = ty!(*array);
                        let TyKind::Array(element, _) = *self.kind(array_ty) else {
                            self.invalid(pc, span, "For", "Array", array_ty, errors);
                            return;
                        };
                        element
                    }
                    ForSource::Range { at, hi } => {
                        let at_ty = ty!(*at);
                        let hi_ty = ty!(*hi);
                        if !matches!(self.kind(at_ty), TyKind::Int(_)) {
                            self.invalid(pc, span, "For", "Int", at_ty, errors);
                            return;
                        }
                        if !types_match(self.types, at_ty, hi_ty) {
                            errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::ForRangeWidths { at: at_ty, hi: hi_ty },
                            });
                        }
                        at_ty
                    }
                };
                if let Some(params) = self.block_params(body, insts) {
                    let supplied = source.supplied_params();
                    if params.len() != supplied {
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
                    self.assert_match(pc, span, "For(body)", "elem", element, elem_ty, errors);
                    if supplied == 2 {
                        let index_ty = ty!(params[1]);
                        self.assert_match(pc, span, "For(body)", "index", self.common.u64, index_ty, errors);
                    }
                }
                if let Some(params) = self.block_params(exit, insts) {
                    let supplied = exit_trip.supplied_params();
                    if params.len() < supplied {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "For(exit)".to_string(),
                                expected: supplied,
                                got: params.len(),
                            },
                        });
                        return;
                    }
                    if let Some(trip) = exit_trip.trip_param(&params) {
                        let trip_ty = ty!(trip);
                        self.assert_match(pc, span, "For(exit)", "trip", self.common.u64, trip_ty, errors);
                        let entries = entries_into(*exit, insts);
                        if entries != 1 {
                            errors.push(ValidationError {
                                scope: self.scope_name.clone(),
                                inst_index: pc,
                                span,
                                kind: ValidationErrorKind::TripBesideExitEdge {
                                    exit: *exit,
                                    entries,
                                },
                            });
                        }
                    }
                    self.check_edge_args(
                        pc,
                        span,
                        "For(exit)",
                        exit_trip.carried_params(&params),
                        exit_args,
                        vt,
                        errors,
                    );
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
                self.assert_match(pc, span, name, "cond", self.common.bool, cond_ty, errors);

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

            InstKind::While {
                cond,
                stages,
                exit,
                exit_args,
            } => {
                let cond_ty = ty!(*cond);
                self.assert_match(pc, span, "While", "cond", self.common.bool, cond_ty, errors);
                let edges = [
                    BranchEdge {
                        side: "body",
                        label: stages.body(),
                        args: &[],
                    },
                    BranchEdge {
                        side: "exit",
                        label: *exit,
                        args: exit_args,
                    },
                ];
                for BranchEdge { side, label, args } in edges {
                    let Some(params) = self.block_params(&label, insts) else {
                        continue;
                    };
                    let edge = format!("While({side})");
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
                if let Some(declared) = self.declared_ret {
                    self.assert_match(pc, span, "Return", "value", declared, value_ty, errors);
                }
                if let Some(o) = order {
                    self.expect_order(*o, vt, span, pc, errors);
                }
            }
        }
    }
}

/// The mutability of the run an `AsSlice` hands back: a container's
/// elements (RFC-0047) or a `String`'s bytes (RFC-0062).
fn run_of(types: &Types<Concrete>, ty: Ty) -> Option<Mutability> {
    let TyKind::Ref(mutability, target) = types.kind(ty) else {
        return None;
    };
    matches!(types.kind(target.ty()), TyKind::Slice(_) | TyKind::Str).then_some(*mutability)
}

/// The mutability and element type of a `&[T]` or `&mut [T]`.
fn slice_of(types: &Types<Concrete>, ty: Ty) -> Option<(Mutability, Ty)> {
    let TyKind::Ref(mutability, target) = types.kind(ty) else {
        return None;
    };
    let TyKind::Slice(element) = types.kind(target.ty()) else {
        return None;
    };
    Some((*mutability, *element))
}

/// How many edges enter the block `label` heads: every terminator edge that
/// names it, and the fall from the block above where that block ends in no
/// terminator. A `Diamond`'s `join` names where its arms meet and is not an
/// edge of its own.
fn entries_into(label: Label, insts: &[crate::ir::Inst]) -> usize {
    let named: usize = insts
        .iter()
        .map(|inst| match &inst.kind {
            InstKind::Jump { label: to, .. } => usize::from(*to == label),
            InstKind::JumpIf {
                then_label,
                else_label,
                ..
            }
            | InstKind::Diamond {
                then_label,
                else_label,
                ..
            } => usize::from(*then_label == label) + usize::from(*else_label == label),
            InstKind::Switch { arms, default, .. } => arms
                .iter()
                .map(|(_, to, _)| to)
                .chain(default.iter().map(|(to, _)| to))
                .filter(|to| **to == label)
                .count(),
            InstKind::For { stages, exit, .. } | InstKind::While { stages, exit, .. } => {
                usize::from(stages.body() == label) + usize::from(*exit == label)
            }
            _ => 0,
        })
        .sum();
    let heads_at = insts.iter().position(
        |inst| matches!(&inst.kind, InstKind::BlockLabel { label: at, .. } if *at == label),
    );
    let falls_in = match heads_at {
        Some(at) if at > 0 => !matches!(
            insts[at - 1].kind,
            InstKind::Jump { .. }
                | InstKind::JumpIf { .. }
                | InstKind::Diamond { .. }
                | InstKind::Switch { .. }
                | InstKind::For { .. }
                | InstKind::While { .. }
                | InstKind::Return { .. }
                | InstKind::Diverge
        ),
        _ => false,
    };
    named + usize::from(falls_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, MirBody, MirModule};
    use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Holds, LenTerm};
    use acvus_ast::Literal;
    use acvus_utils::LocalFactory;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst { span: span(), kind }
    }

    fn make_module(insts: Vec<Inst>, val_types: FxHashMap<ValueId, Ty>, mut types: Types<Concrete>) -> MirModule {
        let unit_ty = types.intern(TyKind::Unit);
        MirModule {
            declared_params: 0,
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
            },
            closures: FxHashMap::default(),
            ret: unit_ty,
            flows: crate::ty::Flows::Every,
            types,
        }
    }

    /// A module whose `main` builds one closure of type `|| -> declared` and
    /// whose closure body returns a value of type `returned`; `declared` and
    /// `returned` are already interned in `types`, which the module then
    /// owns, so a caller compares against `module.types` afterward.
    fn make_closure_module(mut types: Types<Concrete>, declared: Ty, returned: Ty) -> MirModule {
        let label = Label(0);

        let mut main_types = FxHashMap::default();
        let mut main_factory = LocalFactory::<ValueId>::new();
        let closure = main_factory.next();
        let closure_ty = types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: declared,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(crate::ty::Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        main_types.insert(closure, closure_ty);
        let mut module = make_module(
            vec![inst(InstKind::MakeClosure {
                dst: closure,
                body: label,
                captures: Vec::new(),
            })],
            main_types,
            types,
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

    fn array_of_i64(types: &mut Types<Concrete>) -> Ty {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        types.intern(TyKind::Array(i64_ty, LenTerm::Known(3)))
    }

    /// `&str`: a shared reference to the run of bytes a `String` lends.
    fn str_ty(types: &mut Types<Concrete>) -> Ty {
        let str_ty = types.intern(TyKind::Str);
        types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(str_ty)))
    }

    #[test]
    fn a_body_returning_what_it_declares_is_accepted() {
        let mut types = Types::<Concrete>::new();
        let arr = array_of_i64(&mut types);
        let mut module = make_closure_module(types, arr, arr);
        assert!(check_types(&mut module).is_empty());
    }

    #[test]
    fn a_body_declaring_a_value_returning_a_reference_is_refused() {
        let mut types = Types::<Concrete>::new();
        let arr = array_of_i64(&mut types);
        let arr_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(arr)));
        let mut module = make_closure_module(types, arr, arr_ref);
        let errors = check_types(&mut module);
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
        assert_eq!(*expected, array_of_i64(&mut module.types));
        assert!(matches!(module.types.kind(*actual), TyKind::Ref(..)), "{actual:?}");
    }

    #[test]
    fn a_closure_made_from_a_body_the_module_does_not_hold_is_refused() {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut module = make_closure_module(types, i64_ty, i64_ty);
        let body = module
            .closures
            .remove(&Label(0))
            .expect("the module holds the closure it makes");
        module.closures.insert(Label(1), body);
        let errors = check_types(&mut module);
        let [
            ValidationError {
                kind: ValidationErrorKind::UnheldClosure { body: Label(0) },
                inst_index: 0,
                ..
            },
        ] = errors.as_slice()
        else {
            panic!("expected one UnheldClosure at the MakeClosure, got {errors:?}");
        };
    }

    #[test]
    fn a_body_returning_a_diverging_value_is_accepted() {
        let mut types = Types::<Concrete>::new();
        let arr = array_of_i64(&mut types);
        let never_ty = types.intern(TyKind::Never);
        let mut module = make_closure_module(types, arr, never_ty);
        assert!(check_types(&mut module).is_empty());
    }

    /// `!` is the bottom type (types §4): a value of it fits any slot, but a
    /// slot declared `!` accepts no value of another type.
    #[test]
    fn a_body_declaring_never_refuses_the_value_it_returns() {
        let mut types = Types::<Concrete>::new();
        let never_ty = types.intern(TyKind::Never);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut module = make_closure_module(types, never_ty, i64_ty);
        let errors = check_types(&mut module);
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
        assert!(matches!(module.types.kind(*expected), TyKind::Never), "{expected:?}");
        assert!(matches!(module.types.kind(*actual), TyKind::Int(IntTy::I64)), "{actual:?}");
    }

    enum LentSlot {
        Parameter,
        DefinedByConst,
    }

    fn lends_through_param(slot_is: LentSlot) -> MirModule {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let shared_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(i64_ty)));
        let mut vf = LocalFactory::<ValueId>::new();
        let slot = vf.next();
        let lent = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(slot, i64_ty);
        vt.insert(lent, shared_ref);
        let loan = inst(InstKind::Ref {
            dst: lent,
            target: RefTarget::Param(slot),
            path: Vec::new(),
            mutability: Mutability::Shared,
        });
        let interner = acvus_utils::Interner::new();
        let (insts, params) = match slot_is {
            LentSlot::Parameter => (vec![loan], vec![(interner.intern("p"), slot)]),
            LentSlot::DefinedByConst => {
                let defined = inst(InstKind::Const {
                    dst: slot,
                    value: Literal::Int(1),
                });
                (vec![defined, loan], Vec::new())
            }
        };
        let mut module = make_module(insts, vt, types);
        module.main.params = params;
        module.main.val_factory = vf;
        module
    }

    #[test]
    fn a_parameter_lent_as_a_parameter_is_accepted() {
        assert!(check_types(&mut lends_through_param(LentSlot::Parameter)).is_empty());
    }

    #[test]
    fn a_register_an_instruction_defines_is_not_a_parameter() {
        let errors = check_types(&mut lends_through_param(LentSlot::DefinedByConst));
        let [
            ValidationError {
                kind: ValidationErrorKind::NotAParameter { value_id: 0 },
                inst_index: 1,
                ..
            },
        ] = errors.as_slice()
        else {
            panic!("expected one NotAParameter at the loan, got {errors:?}");
        };
    }

    #[test]
    fn const_type_matches() {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, i64_ty);
        let mut module = make_module(
            vec![inst(InstKind::Const {
                dst: v0,
                value: Literal::Int(42),
            })],
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(errors.is_empty());
    }

    /// `types_match` enumerates the primitives by hand, so a type that is
    /// not in that list does not match itself. `&str` is the newest one and
    /// this is what holds its arm there.
    #[test]
    fn a_str_constant_matches_its_own_type() {
        let mut types = Types::<Concrete>::new();
        let str_ty = str_ty(&mut types);
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, str_ty);
        let mut module = make_module(
            vec![inst(InstKind::ConstStr {
                dst: v0,
                text: "abc".to_string(),
            })],
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn a_str_constant_at_a_string_destination_is_caught() {
        let mut types = Types::<Concrete>::new();
        let string_ty = types.intern(TyKind::String);
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, string_ty);
        let mut module = make_module(
            vec![inst(InstKind::ConstStr {
                dst: v0,
                text: "abc".to_string(),
            })],
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(!errors.is_empty(), "a `&str` constant is not a `String`");
    }

    #[test]
    fn const_type_mismatch() {
        let mut types = Types::<Concrete>::new();
        let string_ty = types.intern(TyKind::String);
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, string_ty); // wrong: Literal::Int should be Int
        let mut module = make_module(
            vec![inst(InstKind::Const {
                dst: v0,
                value: Literal::Int(42),
            })],
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(!errors.is_empty(), "type mismatch should be caught");
    }

    #[test]
    fn binop_type_mismatch() {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let string_ty = types.intern(TyKind::String);
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, i64_ty);
        vt.insert(v1, string_ty); // mismatch
        vt.insert(v2, i64_ty);
        let mut module = make_module(
            vec![inst(InstKind::BinOp {
                dst: v2,
                op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap),
                left: v0,
                right: v1,
            })],
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(!errors.is_empty(), "BinOp type mismatch should be caught");
    }

    #[test]
    fn make_tuple_arity_mismatch() {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let string_ty = types.intern(TyKind::String);
        let tuple_ty = types.intern(TyKind::Tuple(Box::new([i64_ty, string_ty])));
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, i64_ty);
        vt.insert(v1, tuple_ty); // expects 2 elements
        let mut module = make_module(
            vec![inst(InstKind::MakeTuple {
                dst: v1,
                elements: vec![v0],
            })], // only 1
            vt,
            types,
        );
        let errors = check_types(&mut module);
        assert!(!errors.is_empty(), "tuple arity mismatch should be caught");
    }

    #[test]
    fn jump_args_type_match() {
        // BlockLabel with param, Jump with matching arg type
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, i64_ty);
        vt.insert(v1, i64_ty);
        let mut module = make_module(
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
            types,
        );
        let errors = check_types(&mut module);
        assert!(
            errors.is_empty(),
            "matching jump arg types should pass: {errors:?}"
        );
    }

    fn diamond_module(else_rejoins: bool) -> MirModule {
        let mut types = Types::<Concrete>::new();
        let bool_ty = types.intern(TyKind::Bool);
        let unit_ty = types.intern(TyKind::Unit);
        let mut vf = LocalFactory::<ValueId>::new();
        let cond = vf.next();
        let joined = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(cond, bool_ty);
        vt.insert(joined, unit_ty);
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
            types,
        )
    }

    fn demoted_module(marked: bool) -> MirModule {
        let mut types = Types::<Concrete>::new();
        let bool_ty = types.intern(TyKind::Bool);
        let unit_ty = types.intern(TyKind::Unit);
        let mut vf = LocalFactory::<ValueId>::new();
        let cond = vf.next();
        let joined = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(cond, bool_ty);
        vt.insert(joined, unit_ty);
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
            types,
        );
        if marked {
            module.main.demoted_diamonds.insert(crate::cfg::ENTRY_LABEL);
        }
        module
    }

    fn meets_again_refusals(module: &mut MirModule) -> usize {
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
        assert_eq!(meets_again_refusals(&mut demoted_module(true)), 1);
    }

    #[test]
    fn the_same_branch_with_no_demotion_behind_it_is_accepted() {
        let mut module = demoted_module(false);
        assert_eq!(meets_again_refusals(&mut module), 0);
        assert!(
            check_types(&mut module).is_empty(),
            "{:?}",
            check_types(&mut module)
        );
    }

    #[test]
    fn a_diamond_whose_arms_both_reach_the_join_is_accepted() {
        let errors = check_types(&mut diamond_module(true));
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn a_diamond_whose_arm_does_not_reach_the_join_is_refused() {
        let errors = check_types(&mut diamond_module(false));
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
