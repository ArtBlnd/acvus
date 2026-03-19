//! Backend type-verification pass.
//!
//! Walks every instruction in a [`MirModule`] and asserts that, **excluding
//! `Cast`**, the types recorded in `val_types` are exactly consistent with
//! what each instruction expects.  Any mismatch is a backend invariant
//! violation and causes a panic — the IR must not be executed.
//!
//! Design:
//! - `Ty::Error` / `Ty::Var` unify with anything (analysis mode may leave
//!   them unresolved).
//! - `Cast` is the *only* instruction allowed to change a value's type.
//! - Generic variance is invariant: inner types must match recursively.

use acvus_ast::{BinOp, Literal, UnaryOp};
use acvus_mir::builtins;
use acvus_mir::ir::{InstKind, Label, MirBody, MirModule, ValueId};
use acvus_mir::ty::{Polarity, Ty, TySubst};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Verify type consistency of the entire module.  Panics on first mismatch.
pub fn verify_module(module: &MirModule) {
    let mut ctx = VerifyCtx::new("main");
    ctx.verify_body(&module.main);

    for (label, closure) in &module.closures {
        let name = format!("closure({:?})", label);
        let mut ctx = VerifyCtx::new(&name);
        ctx.verify_body(&closure.body);
    }
}

// ---------------------------------------------------------------------------
// Structural type equality (invariant, with Error/Var escape)
// ---------------------------------------------------------------------------

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
        (Ty::Deque(a, o1), Ty::Deque(b, o2)) => o1 == o2 && types_match(a, b),
        (Ty::Option(a), Ty::Option(b)) => types_match(a, b),
        (Ty::Tuple(a), Ty::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| types_match(x, y))
        }
        (Ty::Object(a), Ty::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).map_or(false, |bv| types_match(v, bv)))
        }
        (Ty::Iterator(a, e1), Ty::Iterator(b, e2)) => e1 == e2 && types_match(a, b),
        (Ty::Sequence(a, o1, e1), Ty::Sequence(b, o2, e2)) => {
            o1 == o2 && e1 == e2 && types_match(a, b)
        }

        // Functions
        (
            Ty::Fn { params: p1, ret: r1, kind: k1, effect: e1, .. },
            Ty::Fn { params: p2, ret: r2, kind: k2, effect: e2, .. },
        ) => {
            k1 == k2
                && e1 == e2
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
// Verification context
// ---------------------------------------------------------------------------

struct VerifyCtx<'a> {
    scope_name: &'a str,
    /// label → index in `insts` (for Jump target block param lookup)
    label_map: FxHashMap<Label, usize>,
}

impl<'a> VerifyCtx<'a> {
    fn new(scope_name: &'a str) -> Self {
        Self {
            scope_name,
            label_map: FxHashMap::default(),
        }
    }

    fn verify_body(&mut self, body: &MirBody) {
        // Build label map
        self.label_map.clear();
        for (i, inst) in body.insts.iter().enumerate() {
            if let InstKind::BlockLabel { label, .. } = &inst.kind {
                self.label_map.insert(*label, i);
            }
        }

        for (pc, inst) in body.insts.iter().enumerate() {
            self.verify_inst(pc, &inst.kind, &body.val_types, &body.insts);
        }
    }

    /// Get the type of a ValueId, panicking if absent.
    fn ty_of<'b>(&self, id: ValueId, val_types: &'b FxHashMap<ValueId, Ty>) -> &'b Ty {
        val_types
            .get(&id)
            .unwrap_or_else(|| panic!("[{}] Val({}) has no type entry", self.scope_name, id.0))
    }

    /// Assert two types match, panicking with context on failure.
    fn assert_match(
        &self,
        pc: usize,
        inst_name: &str,
        desc: &str,
        expected: &Ty,
        actual: &Ty,
    ) {
        if !types_match(expected, actual) {
            panic!(
                "[{}] type mismatch at inst #{pc} ({inst_name}), {desc}:\n  expected: {expected:?}\n  actual:   {actual:?}",
                self.scope_name,
            );
        }
    }

    /// Get block params for a label.
    fn block_params(&self, label: &Label, insts: &[acvus_mir::ir::Inst]) -> Vec<ValueId> {
        let idx = self.label_map.get(label).unwrap_or_else(|| {
            panic!("[{}] label {:?} not found in label_map", self.scope_name, label)
        });
        match &insts[*idx].kind {
            InstKind::BlockLabel { params, .. } => params.clone(),
            _ => panic!("[{}] label {:?} points to non-BlockLabel", self.scope_name, label),
        }
    }

    // -- per-instruction verification -----------------------------------------

    fn verify_inst(
        &self,
        pc: usize,
        kind: &InstKind,
        vt: &FxHashMap<ValueId, Ty>,
        insts: &[acvus_mir::ir::Inst],
    ) {
        match kind {
            // === Skip ===
            InstKind::Cast { .. }
            | InstKind::Poison { .. }
            | InstKind::Nop
            | InstKind::BlockLabel { .. } => {}

            // === Const ===
            InstKind::Const { dst, value } => {
                let lit_ty = literal_ty(value);
                // lit_ty is Ty::Error for List literals — skip that case
                if !lit_ty.is_error() {
                    self.assert_match(pc, "Const", "dst", &lit_ty, self.ty_of(*dst, vt));
                }
            }

            // === Constructors ===
            InstKind::MakeDeque { dst, elements } => {
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::Deque(inner, _) = dst_ty {
                    for (i, elem) in elements.iter().enumerate() {
                        self.assert_match(
                            pc, "MakeDeque", &format!("element[{i}]"),
                            inner, self.ty_of(*elem, vt),
                        );
                    }
                } else {
                    panic!(
                        "[{}] inst #{pc} MakeDeque: dst type is not Deque, got {dst_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::MakeObject { dst, fields } => {
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::Object(field_tys) = dst_ty {
                    for (key, val) in fields {
                        if let Some(expected_field_ty) = field_tys.get(key) {
                            self.assert_match(
                                pc, "MakeObject", &format!("field"),
                                expected_field_ty, self.ty_of(*val, vt),
                            );
                        }
                        // If key not in type, the typechecker may have a broader object — skip
                    }
                } else {
                    panic!(
                        "[{}] inst #{pc} MakeObject: dst type is not Object, got {dst_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::MakeRange { dst, start, end, .. } => {
                self.assert_match(pc, "MakeRange", "start", &Ty::Int, self.ty_of(*start, vt));
                self.assert_match(pc, "MakeRange", "end", &Ty::Int, self.ty_of(*end, vt));
                self.assert_match(pc, "MakeRange", "dst", &Ty::Range, self.ty_of(*dst, vt));
            }

            InstKind::MakeTuple { dst, elements } => {
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::Tuple(elem_tys) = dst_ty {
                    if elem_tys.len() != elements.len() {
                        panic!(
                            "[{}] inst #{pc} MakeTuple: arity mismatch, type has {} elements but {} provided",
                            self.scope_name, elem_tys.len(), elements.len(),
                        );
                    }
                    for (i, (elem, expected)) in elements.iter().zip(elem_tys).enumerate() {
                        self.assert_match(
                            pc, "MakeTuple", &format!("element[{i}]"),
                            expected, self.ty_of(*elem, vt),
                        );
                    }
                } else {
                    panic!(
                        "[{}] inst #{pc} MakeTuple: dst type is not Tuple, got {dst_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::MakeClosure { dst, captures, .. } => {
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::Fn { captures: cap_tys, .. } = dst_ty {
                    if cap_tys.len() != captures.len() {
                        panic!(
                            "[{}] inst #{pc} MakeClosure: capture count mismatch, type has {} but {} provided",
                            self.scope_name, cap_tys.len(), captures.len(),
                        );
                    }
                    for (i, (cap, expected)) in captures.iter().zip(cap_tys).enumerate() {
                        self.assert_match(
                            pc, "MakeClosure", &format!("capture[{i}]"),
                            expected, self.ty_of(*cap, vt),
                        );
                    }
                }
                // If dst is not Fn (e.g. Error), skip
            }

            InstKind::MakeVariant { dst, tag, payload } => {
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::Enum { variants, .. } = dst_ty {
                    if let Some(variant_payload_ty) = variants.get(tag) {
                        match (variant_payload_ty, payload) {
                            (Some(expected), Some(val)) => {
                                self.assert_match(
                                    pc, "MakeVariant", "payload",
                                    expected, self.ty_of(*val, vt),
                                );
                            }
                            (None, None) => {}
                            (Some(_), None) => {
                                panic!(
                                    "[{}] inst #{pc} MakeVariant: variant expects payload but none given",
                                    self.scope_name,
                                );
                            }
                            (None, Some(_)) => {
                                panic!(
                                    "[{}] inst #{pc} MakeVariant: variant has no payload but one given",
                                    self.scope_name,
                                );
                            }
                        }
                    }
                    // Tag not found in type — open enum, skip
                } else if let Ty::Option(inner) = dst_ty {
                    // Option is represented as enum with Some/None tags
                    if let Some(val) = payload {
                        self.assert_match(
                            pc, "MakeVariant", "Option payload",
                            inner, self.ty_of(*val, vt),
                        );
                    }
                }
            }

            // === BinOp ===
            InstKind::BinOp { dst, op, left, right } => {
                let left_ty = self.ty_of(*left, vt);
                let right_ty = self.ty_of(*right, vt);
                let dst_ty = self.ty_of(*dst, vt);

                if binop_is_logical(*op) {
                    // Bool × Bool → Bool
                    self.assert_match(pc, "BinOp(logical)", "left", &Ty::Bool, left_ty);
                    self.assert_match(pc, "BinOp(logical)", "right", &Ty::Bool, right_ty);
                    self.assert_match(pc, "BinOp(logical)", "dst", &Ty::Bool, dst_ty);
                } else if binop_returns_bool(*op) {
                    // T × T → Bool
                    self.assert_match(pc, "BinOp(cmp)", "left ≡ right", left_ty, right_ty);
                    self.assert_match(pc, "BinOp(cmp)", "dst", &Ty::Bool, dst_ty);
                } else {
                    // T × T → T
                    self.assert_match(pc, "BinOp", "left ≡ right", left_ty, right_ty);
                    self.assert_match(pc, "BinOp", "left ≡ dst", left_ty, dst_ty);
                }
            }

            // === UnaryOp ===
            InstKind::UnaryOp { dst, op, operand } => {
                let operand_ty = self.ty_of(*operand, vt);
                let dst_ty = self.ty_of(*dst, vt);
                match op {
                    UnaryOp::Not => {
                        self.assert_match(pc, "UnaryOp(Not)", "operand", &Ty::Bool, operand_ty);
                        self.assert_match(pc, "UnaryOp(Not)", "dst", &Ty::Bool, dst_ty);
                    }
                    UnaryOp::Neg => {
                        // T → T (Int or Float)
                        self.assert_match(pc, "UnaryOp(Neg)", "operand ≡ dst", operand_ty, dst_ty);
                    }
                }
            }

            // === Access ===
            InstKind::FieldGet { dst, object, field } => {
                let obj_ty = self.ty_of(*object, vt);
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(field) {
                        self.assert_match(pc, "FieldGet", "dst", field_ty, self.ty_of(*dst, vt));
                    }
                } else if !obj_ty.is_error() && !matches!(obj_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} FieldGet: object type is not Object, got {obj_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::ObjectGet { dst, object, key } => {
                let obj_ty = self.ty_of(*object, vt);
                if let Ty::Object(fields) = obj_ty {
                    if let Some(field_ty) = fields.get(key) {
                        self.assert_match(pc, "ObjectGet", "dst", field_ty, self.ty_of(*dst, vt));
                    }
                } else if !obj_ty.is_error() && !matches!(obj_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} ObjectGet: object type is not Object, got {obj_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::TupleIndex { dst, tuple, index } => {
                let tup_ty = self.ty_of(*tuple, vt);
                if let Ty::Tuple(elems) = tup_ty {
                    if let Some(elem_ty) = elems.get(*index) {
                        self.assert_match(pc, "TupleIndex", "dst", elem_ty, self.ty_of(*dst, vt));
                    } else {
                        panic!(
                            "[{}] inst #{pc} TupleIndex: index {index} out of bounds (tuple has {} elements)",
                            self.scope_name, elems.len(),
                        );
                    }
                } else if !tup_ty.is_error() && !matches!(tup_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} TupleIndex: not a Tuple, got {tup_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::ListIndex { dst, list, .. } => {
                let list_ty = self.ty_of(*list, vt);
                if let Ty::List(inner) = list_ty {
                    self.assert_match(pc, "ListIndex", "dst", inner, self.ty_of(*dst, vt));
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} ListIndex: not a List, got {list_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::ListGet { dst, list, index } => {
                let list_ty = self.ty_of(*list, vt);
                self.assert_match(pc, "ListGet", "index", &Ty::Int, self.ty_of(*index, vt));
                if let Ty::List(inner) = list_ty {
                    self.assert_match(pc, "ListGet", "dst", inner, self.ty_of(*dst, vt));
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} ListGet: not a List, got {list_ty:?}",
                        self.scope_name,
                    );
                }
            }

            InstKind::ListSlice { dst, list, .. } => {
                let list_ty = self.ty_of(*list, vt);
                let dst_ty = self.ty_of(*dst, vt);
                if let Ty::List(_) = list_ty {
                    self.assert_match(pc, "ListSlice", "dst ≡ list", list_ty, dst_ty);
                } else if !list_ty.is_error() && !matches!(list_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} ListSlice: not a List, got {list_ty:?}",
                        self.scope_name,
                    );
                }
            }

            // === Pattern tests (all produce Bool) ===
            InstKind::TestLiteral { dst, .. } => {
                self.assert_match(pc, "TestLiteral", "dst", &Ty::Bool, self.ty_of(*dst, vt));
            }

            InstKind::TestListLen { dst, src, .. } => {
                let src_ty = self.ty_of(*src, vt);
                if !matches!(src_ty, Ty::List(_) | Ty::Error(_) | Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} TestListLen: src is not List, got {src_ty:?}",
                        self.scope_name,
                    );
                }
                self.assert_match(pc, "TestListLen", "dst", &Ty::Bool, self.ty_of(*dst, vt));
            }

            InstKind::TestObjectKey { dst, src, .. } => {
                let src_ty = self.ty_of(*src, vt);
                if !matches!(src_ty, Ty::Object(_) | Ty::Error(_) | Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} TestObjectKey: src is not Object, got {src_ty:?}",
                        self.scope_name,
                    );
                }
                self.assert_match(pc, "TestObjectKey", "dst", &Ty::Bool, self.ty_of(*dst, vt));
            }

            InstKind::TestRange { dst, src, .. } => {
                self.assert_match(pc, "TestRange", "src", &Ty::Int, self.ty_of(*src, vt));
                self.assert_match(pc, "TestRange", "dst", &Ty::Bool, self.ty_of(*dst, vt));
            }

            InstKind::TestVariant { dst, src, .. } => {
                let src_ty = self.ty_of(*src, vt);
                if !matches!(src_ty, Ty::Enum { .. } | Ty::Option(_) | Ty::Error(_) | Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} TestVariant: src is not Enum/Option, got {src_ty:?}",
                        self.scope_name,
                    );
                }
                self.assert_match(pc, "TestVariant", "dst", &Ty::Bool, self.ty_of(*dst, vt));
            }

            InstKind::UnwrapVariant { dst, src } => {
                let src_ty = self.ty_of(*src, vt);
                match src_ty {
                    Ty::Option(inner) => {
                        self.assert_match(pc, "UnwrapVariant", "dst", inner, self.ty_of(*dst, vt));
                    }
                    Ty::Enum { .. } => {
                        // Enum unwrap: dst type comes from val_types, trust typechecker
                        // (exact variant payload depends on control-flow context)
                    }
                    Ty::Error(_) | Ty::Var(_) => {}
                    _ => {
                        panic!(
                            "[{}] inst #{pc} UnwrapVariant: src is not Enum/Option, got {src_ty:?}",
                            self.scope_name,
                        );
                    }
                }
            }

            // === IterStep ===
            InstKind::IterStep { dst, src } => {
                let src_ty = self.ty_of(*src, vt);
                if let Ty::Iterator(elem, effect) = src_ty {
                    // dst should be Option(Tuple([T, Iterator(T, E)]))
                    let expected_dst = Ty::Option(Box::new(Ty::Tuple(vec![
                        *elem.clone(),
                        Ty::Iterator(elem.clone(), *effect),
                    ])));
                    self.assert_match(pc, "IterStep", "dst", &expected_dst, self.ty_of(*dst, vt));
                } else if !src_ty.is_error() && !matches!(src_ty, Ty::Var(_)) {
                    panic!(
                        "[{}] inst #{pc} IterStep: src is not Iterator, got {src_ty:?}",
                        self.scope_name,
                    );
                }
            }

            // === Calls ===
            InstKind::BuiltinCall { dst, builtin, args } => {
                let entry = builtins::registry().get(*builtin);
                let mut subst = TySubst::new();
                let (param_tys, ret_ty) = (entry.signature)(&mut subst);

                if args.len() != param_tys.len() {
                    panic!(
                        "[{}] inst #{pc} BuiltinCall({}): arity mismatch, signature has {} params but {} args",
                        self.scope_name, entry.name, param_tys.len(), args.len(),
                    );
                }

                // Unify each arg with param to bind Vars
                for (i, (arg, param_ty)) in args.iter().zip(&param_tys).enumerate() {
                    let actual = self.ty_of(*arg, vt);
                    if subst.unify(actual, param_ty, Polarity::Invariant).is_err() {
                        panic!(
                            "[{}] inst #{pc} BuiltinCall({}): arg[{i}] type mismatch:\n  signature: {param_ty:?}\n  actual:    {actual:?}",
                            self.scope_name, entry.name,
                        );
                    }
                }

                // Check resolved return type against dst
                let resolved_ret = subst.resolve(&ret_ty);
                self.assert_match(
                    pc, &format!("BuiltinCall({})", entry.name), "return",
                    &resolved_ret, self.ty_of(*dst, vt),
                );
            }

            InstKind::ClosureCall { dst, closure, args } => {
                let closure_ty = self.ty_of(*closure, vt);
                if let Ty::Fn { params, ret, .. } = closure_ty {
                    if args.len() != params.len() {
                        panic!(
                            "[{}] inst #{pc} ClosureCall: arity mismatch, fn has {} params but {} args",
                            self.scope_name, params.len(), args.len(),
                        );
                    }
                    for (i, (arg, param_ty)) in args.iter().zip(params).enumerate() {
                        self.assert_match(
                            pc, "ClosureCall", &format!("arg[{i}]"),
                            param_ty, self.ty_of(*arg, vt),
                        );
                    }
                    self.assert_match(pc, "ClosureCall", "return", ret, self.ty_of(*dst, vt));
                }
                // Fn type might be Error/Var — skip
            }

            InstKind::ExternCall { .. } => {
                // External — cannot verify signature
            }

            // === Context / Variables ===
            InstKind::ContextLoad { dst, .. } => {
                // External input, just verify type entry exists
                let _ = self.ty_of(*dst, vt);
            }
            InstKind::VarLoad { dst, .. } => {
                let _ = self.ty_of(*dst, vt);
            }
            InstKind::VarStore { src, .. } => {
                let _ = self.ty_of(*src, vt);
            }

            // === Output ===
            InstKind::Yield(v) => {
                let _ = self.ty_of(*v, vt);
            }

            // === Control flow ===
            InstKind::Jump { label, args } => {
                let params = self.block_params(label, insts);
                if args.len() != params.len() {
                    panic!(
                        "[{}] inst #{pc} Jump: arity mismatch, block has {} params but {} args",
                        self.scope_name, params.len(), args.len(),
                    );
                }
                for (i, (arg, param)) in args.iter().zip(&params).enumerate() {
                    self.assert_match(
                        pc, "Jump", &format!("arg[{i}]"),
                        self.ty_of(*param, vt), self.ty_of(*arg, vt),
                    );
                }
            }

            InstKind::JumpIf { cond, then_label, then_args, else_label, else_args } => {
                self.assert_match(pc, "JumpIf", "cond", &Ty::Bool, self.ty_of(*cond, vt));

                let then_params = self.block_params(then_label, insts);
                if then_args.len() != then_params.len() {
                    panic!(
                        "[{}] inst #{pc} JumpIf(then): arity mismatch, {} params but {} args",
                        self.scope_name, then_params.len(), then_args.len(),
                    );
                }
                for (i, (arg, param)) in then_args.iter().zip(&then_params).enumerate() {
                    self.assert_match(
                        pc, "JumpIf(then)", &format!("arg[{i}]"),
                        self.ty_of(*param, vt), self.ty_of(*arg, vt),
                    );
                }

                let else_params = self.block_params(else_label, insts);
                if else_args.len() != else_params.len() {
                    panic!(
                        "[{}] inst #{pc} JumpIf(else): arity mismatch, {} params but {} args",
                        self.scope_name, else_params.len(), else_args.len(),
                    );
                }
                for (i, (arg, param)) in else_args.iter().zip(&else_params).enumerate() {
                    self.assert_match(
                        pc, "JumpIf(else)", &format!("arg[{i}]"),
                        self.ty_of(*param, vt), self.ty_of(*arg, vt),
                    );
                }
            }

            InstKind::Return(val) => {
                let _ = self.ty_of(*val, vt);
            }
        }
    }
}
