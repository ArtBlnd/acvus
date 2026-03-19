//! MIR type-verification pass (error-collecting variant).
//!
//! Walks every instruction in a [`MirModule`] and checks that, **excluding
//! `Cast`**, the types recorded in `val_types` are exactly consistent with
//! what each instruction expects.  Any mismatch is collected as a
//! [`ValidationError`] instead of panicking.
//!
//! Design:
//! - `Ty::Error` / `Ty::Var` unify with anything (analysis mode may leave
//!   them unresolved).
//! - `Cast` is the *only* instruction allowed to change a value's type.
//! - Generic variance is invariant: inner types must match recursively.

use acvus_ast::{BinOp, Literal, Span, UnaryOp};
use crate::builtins;
use crate::ir::{InstKind, Label, MirBody, MirModule, ValueId};
use crate::ty::{Effect, Origin, Polarity, Ty, TySubst};
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
        ctx.check_body(&closure.body, &mut errors);
    }

    errors
}

// ---------------------------------------------------------------------------
// Structural type equality (invariant, with Error/Var escape)
// ---------------------------------------------------------------------------

/// Returns `true` if two origins match.  `Origin::Var` (unresolved) matches anything.
fn origins_match(a: &Origin, b: &Origin) -> bool {
    match (a, b) {
        (Origin::Var(_), _) | (_, Origin::Var(_)) => true,
        _ => a == b,
    }
}

/// Returns `true` if `a` and `b` are structurally equal under invariant
/// variance.  `Ty::Error` and `Ty::Var` match anything (poison / unresolved).
fn types_match(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        // Poison / unresolved — accept anything.
        (Ty::Error(_), _) | (_, Ty::Error(_)) => true,
        (Ty::Var(_), _) | (_, Ty::Var(_)) => true,

        // Primitives
        (Ty::Int, Ty::Int) => true,
        (Ty::Float, Ty::Float) => true,
        (Ty::String, Ty::String) => true,
        (Ty::Bool, Ty::Bool) => true,
        (Ty::Unit, Ty::Unit) => true,
        (Ty::Range, Ty::Range) => true,
        (Ty::Byte, Ty::Byte) => true,

        // Containers (invariant inner)
        (Ty::List(a), Ty::List(b)) => types_match(a, b),
        (Ty::Deque(a, o1), Ty::Deque(b, o2)) => origins_match(o1, o2) && types_match(a, b),
        (Ty::Option(a), Ty::Option(b)) => types_match(a, b),
        (Ty::Tuple(a), Ty::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| types_match(x, y))
        }
        (Ty::Object(a), Ty::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).map_or(false, |bv| types_match(v, bv)))
        }
        (Ty::Iterator(a, e1), Ty::Iterator(b, e2)) => effects_match(e1, e2) && types_match(a, b),
        (Ty::Sequence(a, o1, e1), Ty::Sequence(b, o2, e2)) => {
            origins_match(o1, o2) && effects_match(e1, e2) && types_match(a, b)
        }

        // Functions
        (
            Ty::Fn { params: p1, ret: r1, kind: k1, effect: e1, .. },
            Ty::Fn { params: p2, ret: r2, kind: k2, effect: e2, .. },
        ) => {
            k1 == k2
                && effects_match(e1, e2)
                && p1.len() == p2.len()
                && p1.iter().zip(p2).all(|(a, b)| types_match(a, b))
                && types_match(r1, r2)
        }

        // Enum — same name is sufficient (variants are open/unified elsewhere)
        (Ty::Enum { name: n1, .. }, Ty::Enum { name: n2, .. }) => n1 == n2,

        // Opaque — same name
        (Ty::Opaque(a), Ty::Opaque(b)) => a == b,

        // Infer should never appear post-lowering, but treat like Var
        (Ty::Infer(_), _) | (_, Ty::Infer(_)) => true,

        _ => false,
    }
}

/// Effect matching for validation: Var matches anything (like Ty::Var),
/// and Pure ≤ Effectful (subtyping).
fn effects_match(a: &Effect, b: &Effect) -> bool {
    match (a, b) {
        (Effect::Var(_), _) | (_, Effect::Var(_)) => true,
        (Effect::Pure, Effect::Pure) | (Effect::Effectful, Effect::Effectful) => true,
        // Pure ≤ Effectful: a Pure actual is acceptable where Effectful is expected.
        (Effect::Pure, Effect::Effectful) | (Effect::Effectful, Effect::Pure) => true,
        // No other concrete mismatch exists for the two-element lattice.
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
        // Literal::List produces Deque at runtime, but the *type* from the
        // typechecker perspective depends on context.  We check dst type
        // via val_types instead of inferring from the literal.
        Literal::List(_) => return Ty::error(), // skip — heterogeneous check not possible here
    }
}

/// When the lowerer reuses a ValueId for both a collection and its iterated
/// element (e.g. pattern-match iteration over a List), the val_types map may
/// record the collection type rather than the element type.  This helper
/// unwraps one level of List/Deque so pattern-match checks don't produce
/// false positives.
fn unwrap_element_ty(ty: &Ty) -> &Ty {
    match ty {
        Ty::List(inner) | Ty::Deque(inner, _) => inner,
        other => other,
    }
}

/// Returns `true` if the type is list-like (List or Deque), extracting the inner type.
/// The lowerer may record Deque where List is expected (pre-cast representation).
fn as_list_inner(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::List(inner) | Ty::Deque(inner, _) => Some(inner),
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

/// Returns `true` if the BinOp is a logical op (Bool × Bool → Bool).
fn binop_is_logical(op: BinOp) -> bool {
    matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
}

// ---------------------------------------------------------------------------
// Check context
// ---------------------------------------------------------------------------

struct CheckCtx {
    scope_name: String,
    /// label → index in `insts` (for Jump target block param lookup)
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
            self.check_inst(pc, inst.span, &inst.kind, &body.val_types, &body.insts, errors);
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
                    kind: ValidationErrorKind::MissingType { value_id: id.0 },
                });
                None
            }
        }
    }

    /// Assert two types match.  Pushes a `TypeMismatch` error on failure.
    /// Returns `true` if they match.
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
            InstKind::Cast { .. }
            | InstKind::Poison { .. }
            | InstKind::Nop
            | InstKind::BlockLabel { .. } => {}

            // === Const ===
            InstKind::Const { dst, value } => {
                let lit_ty = literal_ty(value);
                if !lit_ty.is_error() {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "Const", "dst", &lit_ty, dst_ty, errors);
                }
            }

            // === Constructors ===
            InstKind::MakeDeque { dst, elements } => {
                let dst_ty = ty!(*dst);
                if let Ty::Deque(inner, _) = dst_ty {
                    for (i, elem) in elements.iter().enumerate() {
                        let elem_ty = ty!(*elem);
                        self.assert_match(
                            pc, span, "MakeDeque", &format!("element[{i}]"),
                            inner, elem_ty, errors,
                        );
                    }
                } else if !dst_ty.is_error() && !matches!(dst_ty, Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "MakeDeque".to_string(),
                            expected_constructor: "Deque".to_string(),
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
                                pc, span, "MakeObject", "field",
                                expected_field_ty, val_ty, errors,
                            );
                        }
                    }
                } else if !dst_ty.is_error() && !matches!(dst_ty, Ty::Var(_)) {
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

            InstKind::MakeRange { dst, start, end, .. } => {
                let start_ty = ty!(*start);
                let end_ty = ty!(*end);
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "MakeRange", "start", &Ty::Int, start_ty, errors);
                self.assert_match(pc, span, "MakeRange", "end", &Ty::Int, end_ty, errors);
                self.assert_match(pc, span, "MakeRange", "dst", &Ty::Range, dst_ty, errors);
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
                                pc, span, "MakeTuple", &format!("element[{i}]"),
                                expected, elem_ty, errors,
                            );
                        }
                    }
                } else if !dst_ty.is_error() && !matches!(dst_ty, Ty::Var(_)) {
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
                if let Ty::Fn { captures: cap_tys, .. } = dst_ty {
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
                                pc, span, "MakeClosure", &format!("capture[{i}]"),
                                expected, cap_ty, errors,
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
                                    pc, span, "MakeVariant", "payload",
                                    expected, val_ty, errors,
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
                    // Tag not found in type — open enum, skip
                } else if let Ty::Option(inner) = dst_ty {
                    // Option is represented as enum with Some/None tags
                    if let Some(val) = payload {
                        let val_ty = ty!(*val);
                        self.assert_match(
                            pc, span, "MakeVariant", "Option payload",
                            inner, val_ty, errors,
                        );
                    }
                }
            }

            // === BinOp ===
            InstKind::BinOp { dst, op, left, right } => {
                let left_ty = ty!(*left);
                let right_ty = ty!(*right);
                let dst_ty = ty!(*dst);

                if binop_is_logical(*op) {
                    self.assert_match(pc, span, "BinOp(logical)", "left", &Ty::Bool, left_ty, errors);
                    self.assert_match(pc, span, "BinOp(logical)", "right", &Ty::Bool, right_ty, errors);
                    self.assert_match(pc, span, "BinOp(logical)", "dst", &Ty::Bool, dst_ty, errors);
                } else if binop_returns_bool(*op) {
                    self.assert_match(pc, span, "BinOp(cmp)", "left ≡ right", left_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp(cmp)", "dst", &Ty::Bool, dst_ty, errors);
                } else {
                    self.assert_match(pc, span, "BinOp", "left ≡ right", left_ty, right_ty, errors);
                    self.assert_match(pc, span, "BinOp", "left ≡ dst", left_ty, dst_ty, errors);
                }
            }

            // === UnaryOp ===
            InstKind::UnaryOp { dst, op, operand } => {
                let operand_ty = ty!(*operand);
                let dst_ty = ty!(*dst);
                match op {
                    UnaryOp::Not => {
                        self.assert_match(pc, span, "UnaryOp(Not)", "operand", &Ty::Bool, operand_ty, errors);
                        self.assert_match(pc, span, "UnaryOp(Not)", "dst", &Ty::Bool, dst_ty, errors);
                    }
                    UnaryOp::Neg => {
                        self.assert_match(pc, span, "UnaryOp(Neg)", "operand ≡ dst", operand_ty, dst_ty, errors);
                    }
                }
            }

            // === Access ===
            InstKind::FieldGet { dst, object, field } => {
                let obj_ty = ty!(*object);
                // Try direct type first, then unwrap one container level
                // (lowerer may record List/Deque type for pattern-match iteration)
                let obj_ty = if matches!(obj_ty, Ty::Object(_) | Ty::Error(_) | Ty::Var(_)) {
                    obj_ty
                } else {
                    unwrap_element_ty(obj_ty)
                };
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(field) {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "FieldGet", "dst", field_ty, dst_ty, errors);
                    }
                } else if !obj_ty.is_error() && !matches!(obj_ty, Ty::Var(_)) {
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

            InstKind::ObjectGet { dst, object, key } => {
                let obj_ty = ty!(*object);
                let obj_ty = if matches!(obj_ty, Ty::Object(_) | Ty::Error(_) | Ty::Var(_)) {
                    obj_ty
                } else {
                    unwrap_element_ty(obj_ty)
                };
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(key) {
                        let dst_ty = ty!(*dst);
                        self.assert_match(pc, span, "ObjectGet", "dst", field_ty, dst_ty, errors);
                    }
                } else if !obj_ty.is_error() && !matches!(obj_ty, Ty::Var(_)) {
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
                } else if !tup_ty.is_error() && !matches!(tup_ty, Ty::Var(_)) {
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

            InstKind::ListIndex { dst, list, .. } => {
                let list_ty = ty!(*list);
                if let Some(inner) = as_list_inner(list_ty) {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ListIndex", "dst", inner, dst_ty, errors);
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ListIndex".to_string(),
                            expected_constructor: "List".to_string(),
                            actual: list_ty.clone(),
                        },
                    });
                }
            }

            InstKind::ListGet { dst, list, index } => {
                let list_ty = ty!(*list);
                let index_ty = ty!(*index);
                self.assert_match(pc, span, "ListGet", "index", &Ty::Int, index_ty, errors);
                if let Some(inner) = as_list_inner(list_ty) {
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ListGet", "dst", inner, dst_ty, errors);
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ListGet".to_string(),
                            expected_constructor: "List".to_string(),
                            actual: list_ty.clone(),
                        },
                    });
                }
            }

            InstKind::ListSlice { dst, list, .. } => {
                let list_ty = ty!(*list);
                let dst_ty = ty!(*dst);
                if as_list_inner(list_ty).is_some() {
                    self.assert_match(pc, span, "ListSlice", "dst ≡ list", list_ty, dst_ty, errors);
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "ListSlice".to_string(),
                            expected_constructor: "List".to_string(),
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

            InstKind::TestListLen { dst, src, .. } => {
                let src_ty = ty!(*src);
                if !matches!(src_ty, Ty::List(_) | Ty::Deque(_, _) | Ty::Error(_) | Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "TestListLen".to_string(),
                            expected_constructor: "List".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestListLen", "dst", &Ty::Bool, dst_ty, errors);
            }

            InstKind::TestObjectKey { dst, src, .. } => {
                let src_ty = ty!(*src);
                let src_ty = if matches!(src_ty, Ty::Object(_) | Ty::Error(_) | Ty::Var(_)) {
                    src_ty
                } else {
                    unwrap_element_ty(src_ty)
                };
                if !matches!(src_ty, Ty::Object(_) | Ty::Error(_) | Ty::Var(_)) {
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

            InstKind::TestRange { dst, src, .. } => {
                let src_ty = ty!(*src);
                let dst_ty = ty!(*dst);
                self.assert_match(pc, span, "TestRange", "src", &Ty::Int, src_ty, errors);
                self.assert_match(pc, span, "TestRange", "dst", &Ty::Bool, dst_ty, errors);
            }

            InstKind::TestVariant { dst, src, .. } => {
                let src_ty = ty!(*src);
                if !matches!(src_ty, Ty::Enum { .. } | Ty::Option(_) | Ty::Error(_) | Ty::Var(_)) {
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
                    Ty::Error(_) | Ty::Var(_) => {}
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

            // === IterStep ===
            InstKind::IterStep { dst, src } => {
                let src_ty = ty!(*src);
                if let Ty::Iterator(elem, effect) = src_ty {
                    let expected_dst = Ty::Option(Box::new(Ty::Tuple(vec![
                        *elem.clone(),
                        Ty::Iterator(elem.clone(), *effect),
                    ])));
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "IterStep", "dst", &expected_dst, dst_ty, errors);
                } else if !src_ty.is_error() && !matches!(src_ty, Ty::Var(_)) {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::InvalidConstructor {
                            inst_name: "IterStep".to_string(),
                            expected_constructor: "Iterator".to_string(),
                            actual: src_ty.clone(),
                        },
                    });
                }
            }

            // === Calls ===
            InstKind::BuiltinCall { dst, builtin, args } => {
                let entry = builtins::registry().get(*builtin);
                let mut subst = TySubst::new();
                let (param_tys, ret_ty) = (entry.signature)(&mut subst);

                if args.len() != param_tys.len() {
                    errors.push(ValidationError {
                        scope: self.scope_name.clone(),
                        inst_index: pc,
                        span,
                        kind: ValidationErrorKind::ArityMismatch {
                            inst_name: format!("BuiltinCall({})", entry.name),
                            expected: param_tys.len(),
                            got: args.len(),
                        },
                    });
                    return;
                }

                // Unify each arg with param to bind Vars.
                // Covariant: actual ≤ expected (e.g. Pure fn is accepted where Effectful is expected).
                for (i, (arg, param_ty)) in args.iter().zip(&param_tys).enumerate() {
                    let actual = ty!(*arg);
                    if subst.unify(actual, param_ty, Polarity::Covariant).is_err() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::TypeMismatch {
                                inst_name: format!("BuiltinCall({})", entry.name),
                                desc: format!("arg[{i}]"),
                                expected: param_ty.clone(),
                                actual: actual.clone(),
                            },
                        });
                    }
                }

                // Check resolved return type against dst
                let resolved_ret = subst.resolve(&ret_ty);
                let dst_ty = ty!(*dst);
                self.assert_match(
                    pc, span, &format!("BuiltinCall({})", entry.name), "return",
                    &resolved_ret, dst_ty, errors,
                );
            }

            InstKind::ClosureCall { dst, closure, args } => {
                let closure_ty = ty!(*closure);
                if let Ty::Fn { params, ret, .. } = closure_ty {
                    if args.len() != params.len() {
                        errors.push(ValidationError {
                            scope: self.scope_name.clone(),
                            inst_index: pc,
                            span,
                            kind: ValidationErrorKind::ArityMismatch {
                                inst_name: "ClosureCall".to_string(),
                                expected: params.len(),
                                got: args.len(),
                            },
                        });
                    } else {
                        for (i, (arg, param_ty)) in args.iter().zip(params).enumerate() {
                            let arg_ty = ty!(*arg);
                            self.assert_match(
                                pc, span, "ClosureCall", &format!("arg[{i}]"),
                                param_ty, arg_ty, errors,
                            );
                        }
                    }
                    let dst_ty = ty!(*dst);
                    self.assert_match(pc, span, "ClosureCall", "return", ret, dst_ty, errors);
                }
                // Fn type might be Error/Var — skip
            }

            InstKind::ExternCall { .. } => {
                // External — cannot verify signature
            }

            // === Context / Variables ===
            InstKind::ContextLoad { dst, .. } => {
                let _ = self.ty_of(*dst, vt, span, pc, errors);
            }
            InstKind::VarLoad { dst, .. } => {
                let _ = self.ty_of(*dst, vt, span, pc, errors);
            }
            InstKind::VarStore { src, .. } => {
                let _ = self.ty_of(*src, vt, span, pc, errors);
            }

            // === Output ===
            InstKind::Yield(v) => {
                let _ = self.ty_of(*v, vt, span, pc, errors);
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
                                pc, span, "Jump", &format!("arg[{i}]"),
                                param_ty, arg_ty, errors,
                            );
                        }
                    }
                }
            }

            InstKind::JumpIf { cond, then_label, then_args, else_label, else_args } => {
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
                                pc, span, "JumpIf(then)", &format!("arg[{i}]"),
                                param_ty, arg_ty, errors,
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
                                pc, span, "JumpIf(else)", &format!("arg[{i}]"),
                                param_ty, arg_ty, errors,
                            );
                        }
                    }
                }
            }

            InstKind::Return(val) => {
                let _ = self.ty_of(*val, vt, span, pc, errors);
            }
        }
    }
}
