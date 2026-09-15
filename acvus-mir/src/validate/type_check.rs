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

use crate::ir::{Callee, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValueId};
use crate::ty::{Mutability, Ty};
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
    InvalidConstructor {
        inst_name: String,
        expected_constructor: String,
        actual: Ty,
    },
    /// Use of a move-only value after it has been consumed.
    UseAfterMove {
        value_id: u32,
        moved_at: usize,
        ty: Ty,
    },
    /// A storage touched while a reference to it excludes that (RFC-0018).
    BorrowConflict {
        storage: String,
        reference: u32,
    },
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Check type consistency of the entire module.  Returns all errors found.
pub fn check_types(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let mut ctx = CheckCtx::new("main".to_string());
    ctx.check_body(&module.main, &mut errors);

    for (label, closure) in &module.closures {
        let name = format!("closure({:?})", label);
        let mut ctx = CheckCtx::new(name);
        ctx.check_body(closure, &mut errors);
    }

    errors
}

// ---------------------------------------------------------------------------
// Structural type equality (invariant, with Error escape)
// ---------------------------------------------------------------------------

/// Returns `true` if two identity slots match.  `Ty::Error` matches anything.
/// `Ty::Var(Infallible)` is uninhabitable for concrete types.
fn identities_match(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Error(_), _) | (_, Ty::Error(_)) => true,
        (Ty::Var(v), _) | (_, Ty::Var(v)) => match *v {},
        _ => a == b,
    }
}

/// Returns `true` if `a` and `b` are structurally equal under invariant
/// variance.  `Ty::Error` matches anything (poison).
/// `Ty::Var(Infallible)` is uninhabitable for concrete types.
fn types_match(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        // Poison - accept anything.
        (Ty::Error(_), _) | (_, Ty::Error(_)) => true,
        // Uninhabitable - concrete types never have Var.
        (Ty::Var(v), _) | (_, Ty::Var(v)) => match *v {},

        // Primitives
        (Ty::Int, Ty::Int) => true,
        (Ty::Float, Ty::Float) => true,
        (Ty::String, Ty::String) => true,
        (Ty::Bool, Ty::Bool) => true,
        (Ty::Unit, Ty::Unit) => true,
        (Ty::Byte, Ty::Byte) => true,
        (Ty::Order, Ty::Order) => true,

        // Containers (invariant inner)
        (Ty::Array(a, la), Ty::Array(b, lb)) => la == lb && types_match(a, b),
        (Ty::Option(a), Ty::Option(b)) => types_match(a, b),
        (Ty::Ref(ma, a), Ty::Ref(mb, b)) => ma == mb && types_match(a, b),
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

fn literal_ty(lit: &Literal) -> Ty {
    match lit {
        Literal::String(_) => Ty::String,
        Literal::Int(_) => Ty::Int,
        Literal::Float(_) => Ty::Float,
        Literal::Bool(_) => Ty::Bool,
        Literal::Byte(_) => Ty::Byte,
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
}

impl CheckCtx {
    fn new(scope_name: String) -> Self {
        Self {
            scope_name,
            label_map: FxHashMap::default(),
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
        let mut at: Ty = inner.as_ref().clone();
        for seg in path {
            let next = match (seg, &at) {
                (PathSeg::Field(field), Ty::Object(fields)) => fields.get(field).cloned(),
                (PathSeg::Index(i), Ty::Array(elem, _)) => Some(elem.as_ref().clone()).filter(|_| *i < usize::MAX),
                (PathSeg::Index(i), Ty::Tuple(elems)) => elems.get(*i).cloned(),
                (PathSeg::Payload, Ty::Option(payload)) => Some(payload.as_ref().clone()),
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
                        expected_constructor: format!("a type with {seg:?}"),
                        actual: at.clone(),
                    },
                });
                return None;
            };
            at = next;
        }
        Some((*m, at))
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
            | InstKind::BlockLabel { .. } => {}

            InstKind::Drop { src } => {
                // Just validate src exists and has a type.
                let _ = self.ty_of(*src, vt, span, pc, errors);
            }

            // === Const ===
            InstKind::Const { dst, value } => {
                let lit_ty = literal_ty(value);
                if !lit_ty.is_error() {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "Const", "dst", &lit_ty, dst_ty, errors);
                }
            }

            // === Constructors ===
            InstKind::StringClone { dst, src } => {
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "StringClone", "dst", &Ty::String, dst_ty, errors);
                let src_ty = ty!(*src);
                let is_string = match src_ty {
                    Ty::String | Ty::Error(_) => true,
                    Ty::Ref(_, inner) => matches!(inner.as_ref(), Ty::String),
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
                    let is_string_ref = matches!(operand_ty, Ty::Ref(_, inner) if matches!(inner.as_ref(), Ty::String))
                        || operand_ty.is_error();
                    if !is_string_ref {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "StringEq".to_string(),
                                expected_constructor: "&String".to_string(),
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
                    let is_string = match part_ty {
                        Ty::String => true,
                        Ty::Ref(_, inner) => matches!(inner.as_ref(), Ty::String),
                        Ty::Error(_) => true,
                        _ => false,
                    };
                    if !is_string {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::InvalidConstructor {
                                inst_name: "StringConcat".to_string(),
                                expected_constructor: "String or &String".to_string(),
                                actual: part_ty.clone(),
                            },
                        });
                    }
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

            // === Projection ===
            InstKind::Take { dst, target, path } => {
                let dst_ty = ty!(*dst);
                if let RefTarget::Through(r) = target {
                    let Some((_, at)) = self.through(pc, span, "Take", *r, path, vt, errors) else {
                        return;
                    };
                    self.assert_match(pc, span, "Take", "dst", &at, dst_ty, errors);
                    if !at.is_primitive() && !matches!(at, Ty::String | Ty::Ref(..)) && !at.is_error() {
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
                    let Some((m, at)) = self.through(pc, span, "Assign", *r, path, vt, errors) else {
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
                    let expected = Ty::Ref(*mutability, Box::new(at));
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
                            expected_constructor: format!("Ref({mutability:?})"),
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

            InstKind::ArrayGet {
                dst,
                array: list,
                index,
            } => {
                let list_ty = ty!(*list);
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "ArrayGet", "index", &Ty::Int, index_ty, errors);
                if let Some(inner) = as_array_inner(list_ty) {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ArrayGet", "dst", inner, dst_ty, errors);
                } else if !list_ty.is_error() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ArrayGet".to_string(),
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
                    Ty::Ref(_, inner) => inner.as_ref(),
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
                    Ty::Ref(_, inner) => inner.as_ref(),
                    other => other,
                };
                if !matches!(src_ty, Ty::Enum { .. } | Ty::Option(_) | Ty::Error(_)) {
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
                    Ty::Enum { .. } => {
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
                    Callee::Direct(_) => {
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
                    Callee::Direct(_) => {
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
            } => {
                let cond_ty = ty!(*cond);
                self.assert_match(pc, span, "JumpIf", "cond", &Ty::Bool, cond_ty, errors);

                if let Some(then_params) = self.block_params(then_label, insts) {
                    if then_args.len() != then_params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "JumpIf(then)".to_string(),
                                expected: then_params.len(),
                                got: then_args.len(),
                            },
                        });
                    } else {
                        for (i, (arg, param)) in then_args.iter().zip(&then_params).enumerate() {
                            let param_ty = ty!(*param);
                            let arg_ty = ty!(*arg);
                            self.assert_match(
                                pc,
                                span,
                                "JumpIf(then)",
                                &format!("arg[{i}]"),
                                param_ty,
                                arg_ty,
                                errors,
                            );
                        }
                    }
                }

                if let Some(else_params) = self.block_params(else_label, insts) {
                    if else_args.len() != else_params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "JumpIf(else)".to_string(),
                                expected: else_params.len(),
                                got: else_args.len(),
                            },
                        });
                    } else {
                        for (i, (arg, param)) in else_args.iter().zip(&else_params).enumerate() {
                            let param_ty = ty!(*param);
                            let arg_ty = ty!(*arg);
                            self.assert_match(
                                pc,
                                span,
                                "JumpIf(else)",
                                &format!("arg[{i}]"),
                                param_ty,
                                arg_ty,
                                errors,
                            );
                        }
                    }
                }
            }

            InstKind::Return { value, order } => {
                let _ = self.ty_of(*value, vt, span, pc, errors);
                if let Some(o) = order {
                    self.expect_order(*o, vt, span, pc, errors);
                }
            }
        }
    }
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
        }
    }

    #[test]
    fn const_type_matches() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let mut vt = FxHashMap::default();
        vt.insert(v0, Ty::Int);
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
        vt.insert(v0, Ty::Int);
        vt.insert(v1, Ty::String); // mismatch
        vt.insert(v2, Ty::Int);
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
        vt.insert(v0, Ty::Int);
        vt.insert(v1, Ty::Tuple(vec![Ty::Int, Ty::String])); // expects 2 elements
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
        vt.insert(v0, Ty::Int);
        vt.insert(v1, Ty::Int);
        let module = make_module(
            vec![
                inst(InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v1],
                    merge_of: None,
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
}
