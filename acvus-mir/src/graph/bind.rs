//! A bound `$` is a constant (RFC-0071 rule 5), and it holds any value a
//! literal writes (RFC-0087).
//!
//! Obligation across passes: `type_bound` is the one typing of a bound
//! value. `Bindings::bind` admits a value only when it types there, the
//! checker types it there again in every body that reads it, and the
//! constant written here is built at the type that body closed it to. The
//! parameter this drops is one `graph::optimize` will no longer report as
//! an input the host must supply.

use std::collections::BTreeMap;
use std::fmt;

use acvus_ast::{Expr, Literal, ObjectExprField, RefKind, Span, TupleElem, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::error::{MirError, MirErrorKind};
use crate::ir::{Inst, InstKind, MirBody, RefTarget, ValueId};
use crate::solver::Solver;
use crate::ty::{Home, InferTy, IntTy, LenTerm, Mutability, ObjectTy, Ty, TyTerm, TypeArg};

use super::types::{Bindings, QualifiedRef};

// -- The value ---------------------------------------------------------

/// A value a literal writes (RFC-0087 rule 1). An object holds each field
/// once, by construction, and in one order.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundValue {
    /// An unsuffixed integer: `i64`, as its literal is where nothing
    /// demands another width.
    Int(i128),
    /// `10u8`: typed at the width it names (RFC-0058 rule 1).
    IntOf {
        value: i128,
        width: IntTy,
    },
    Float(f64),
    Bool(bool),
    Char(char),
    String(String),
    Bytes(Vec<u8>),
    Unit,
    Array(Vec<BoundValue>),
    Tuple(Vec<BoundValue>),
    Object(BTreeMap<Astr, BoundValue>),
    /// `A::B` or `A::B(payload)`: one variant of a structural enum.
    Variant {
        name: QualifiedRef,
        tag: Astr,
        payload: Option<Box<BoundValue>>,
    },
    Option(Option<Box<BoundValue>>),
    Result(Result<Box<BoundValue>, Box<BoundValue>>),
}

impl BoundValue {
    /// The value in acvus literal syntax, as `acvus run name=<literal>` reads
    /// it back.
    pub fn display<'a>(&'a self, interner: &'a Interner) -> BoundValueDisplay<'a> {
        BoundValueDisplay {
            value: self,
            interner,
        }
    }
}

pub struct BoundValueDisplay<'a> {
    value: &'a BoundValue,
    interner: &'a Interner,
}

impl fmt::Display for BoundValueDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interner = self.interner;
        let shown = |value: &'_ BoundValue| -> String { value.display(interner).to_string() };
        match self.value {
            BoundValue::Int(n) => write!(f, "{n}"),
            BoundValue::IntOf { value, width } => write!(f, "{value}{}", width.name()),
            BoundValue::Float(x) => write!(f, "{x:?}"),
            BoundValue::Bool(b) => write!(f, "{b}"),
            BoundValue::Char(c) => write!(f, "{c:?}"),
            BoundValue::String(text) => write!(f, "{text:?}"),
            BoundValue::Bytes(bytes) => {
                write!(f, "b\"")?;
                for byte in bytes {
                    match byte {
                        b'"' => write!(f, "\\\"")?,
                        b'\\' => write!(f, "\\\\")?,
                        0x20..=0x7e => write!(f, "{}", char::from(*byte))?,
                        other => write!(f, "\\x{other:02x}")?,
                    }
                }
                write!(f, "\"")
            }
            BoundValue::Unit => write!(f, "()"),
            BoundValue::Array(elements) => {
                write!(f, "[")?;
                for (at, element) in elements.iter().enumerate() {
                    if at > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", shown(element))?;
                }
                write!(f, "]")
            }
            BoundValue::Tuple(elements) => {
                write!(f, "(")?;
                for (at, element) in elements.iter().enumerate() {
                    if at > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", shown(element))?;
                }
                // `(x)` is a parenthesized `x`; one element needs the comma.
                if elements.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            BoundValue::Object(fields) => {
                write!(f, "{{")?;
                for (at, (key, field)) in fields.iter().enumerate() {
                    if at > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}: {},", interner.resolve(*key), shown(field))?;
                }
                write!(f, "}}")
            }
            BoundValue::Variant { name, tag, payload } => {
                if let Some(namespace) = name.namespace {
                    write!(f, "{}::", interner.resolve(namespace))?;
                }
                write!(
                    f,
                    "{}::{}",
                    interner.resolve(name.name),
                    interner.resolve(*tag)
                )?;
                match payload {
                    Some(payload) => write!(f, "({})", shown(payload)),
                    None => Ok(()),
                }
            }
            BoundValue::Option(Some(inner)) => write!(f, "Some({})", shown(inner)),
            BoundValue::Option(None) => write!(f, "None"),
            BoundValue::Result(Ok(inner)) => write!(f, "Ok({})", shown(inner)),
            BoundValue::Result(Err(inner)) => write!(f, "Err({})", shown(inner)),
        }
    }
}

// -- Reading ------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotABoundValue {
    NotALiteral { form: Span },
    RepeatedField { key: Astr, second: Span },
}

impl BoundValue {
    /// The value a parsed literal expression writes (RFC-0087 rule 1), for
    /// any host that reads a binding in the script's own syntax
    /// (RFC-0031 rule 4). A `-` before a float literal is its sign, and a
    /// qualified call of one argument is a variant with a payload, as the
    /// checker reads one.
    pub fn from_expr(interner: &Interner, expr: &Expr) -> Result<BoundValue, NotABoundValue> {
        let each = |elements: &[Expr]| -> Result<Vec<BoundValue>, NotABoundValue> {
            elements
                .iter()
                .map(|element| BoundValue::from_expr(interner, element))
                .collect()
        };
        match expr {
            Expr::Literal { value, .. } => Ok(BoundValue::of_literal(value)),
            Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand,
                span,
                ..
            } => match operand.as_ref() {
                Expr::Literal {
                    value: Literal::Float(x),
                    ..
                } => Ok(BoundValue::Float(-x)),
                _ => Err(NotABoundValue::NotALiteral { form: *span }),
            },
            Expr::Paren { inner, .. } => BoundValue::from_expr(interner, inner),
            Expr::List {
                head,
                rest: None,
                tail,
                ..
            } if tail.is_empty() => Ok(BoundValue::Array(each(head)?)),
            Expr::Tuple { elements, .. } => {
                let mut held = Vec::with_capacity(elements.len());
                for element in elements {
                    let TupleElem::Expr(element) = element else {
                        return Err(NotABoundValue::NotALiteral { form: expr.span() });
                    };
                    held.push(BoundValue::from_expr(interner, element)?);
                }
                Ok(BoundValue::Tuple(held))
            }
            Expr::Object { fields, .. } => {
                let mut held = BTreeMap::new();
                for ObjectExprField {
                    key, value, span, ..
                } in fields
                {
                    let value = BoundValue::from_expr(interner, value)?;
                    if held.insert(*key, value).is_some() {
                        return Err(NotABoundValue::RepeatedField {
                            key: *key,
                            second: *span,
                        });
                    }
                }
                Ok(BoundValue::Object(held))
            }
            Expr::Variant {
                enum_name,
                tag,
                payload,
                span,
                ..
            } => {
                let payload = match payload {
                    Some(payload) => Some(Box::new(BoundValue::from_expr(interner, payload)?)),
                    None => None,
                };
                match builtin_variant(interner, *enum_name, *tag, payload) {
                    BuiltinVariant::Written(value) => Ok(value),
                    BuiltinVariant::PayloadMismatch => {
                        Err(NotABoundValue::NotALiteral { form: *span })
                    }
                    BuiltinVariant::Structural(payload) => {
                        let Some(enum_name) = enum_name else {
                            return Err(NotABoundValue::NotALiteral { form: *span });
                        };
                        Ok(BoundValue::Variant {
                            name: QualifiedRef::root(*enum_name),
                            tag: *tag,
                            payload,
                        })
                    }
                }
            }
            Expr::FuncCall {
                func, args, span, ..
            } => match (func.as_ref(), args.as_slice()) {
                (
                    Expr::Ident {
                        name:
                            QualifiedRef {
                                namespace: Some(enum_name),
                                name: tag,
                            },
                        ref_kind: RefKind::Value,
                        ..
                    },
                    [payload],
                ) => Ok(BoundValue::Variant {
                    name: QualifiedRef::root(*enum_name),
                    tag: *tag,
                    payload: Some(Box::new(BoundValue::from_expr(interner, payload)?)),
                }),
                _ => Err(NotABoundValue::NotALiteral { form: *span }),
            },
            other => Err(NotABoundValue::NotALiteral { form: other.span() }),
        }
    }

    fn of_literal(literal: &Literal) -> BoundValue {
        match literal {
            Literal::String(text) => BoundValue::String(text.clone()),
            Literal::Int(n) => BoundValue::Int(*n),
            Literal::IntOf(suffixed) => BoundValue::IntOf {
                value: suffixed.value,
                width: IntTy::from(suffixed.width),
            },
            Literal::Float(x) => BoundValue::Float(*x),
            Literal::Char(c) => BoundValue::Char(*c),
            Literal::Bytes(bytes) => BoundValue::Bytes(bytes.clone()),
            Literal::Bool(b) => BoundValue::Bool(*b),
            Literal::List(elements) => {
                BoundValue::Array(elements.iter().map(BoundValue::of_literal).collect())
            }
            Literal::Unit => BoundValue::Unit,
        }
    }
}

enum BuiltinVariant {
    Written(BoundValue),
    PayloadMismatch,
    Structural(Option<Box<BoundValue>>),
}

/// The checker's `resolve_builtin_variant`: a builtin tag with no enum
/// name, or with the enum it belongs to, is the builtin.
fn builtin_variant(
    interner: &Interner,
    enum_name: Option<Astr>,
    tag: Astr,
    payload: Option<Box<BoundValue>>,
) -> BuiltinVariant {
    let owner = match interner.resolve(tag) {
        "Some" | "None" => "Option",
        "Ok" | "Err" => "Result",
        _ => return BuiltinVariant::Structural(payload),
    };
    if let Some(named) = enum_name
        && interner.resolve(named) != owner
    {
        return BuiltinVariant::Structural(payload);
    }
    match (interner.resolve(tag), payload) {
        ("Some", Some(inner)) => BuiltinVariant::Written(BoundValue::Option(Some(inner))),
        ("None", None) => BuiltinVariant::Written(BoundValue::Option(None)),
        ("Ok", Some(inner)) => BuiltinVariant::Written(BoundValue::Result(Ok(inner))),
        ("Err", Some(inner)) => BuiltinVariant::Written(BoundValue::Result(Err(inner))),
        _ => BuiltinVariant::PayloadMismatch,
    }
}

// -- Typing -------------------------------------------------------------

/// A value that has no type, refused where it is bound (RFC-0087 rule 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingRefused {
    /// Array elements that have no one type, the variants of one enum
    /// among them included.
    ElementsDisagree,
    /// An object wider than an object type admits.
    ObjectTooWide { fields: usize },
    /// A suffixed integer that does not fit its width.
    IntOutOfRange { value: i128, width: IntTy },
}

impl fmt::Display for BindingRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindingRefused::ElementsDisagree => {
                write!(f, "the elements of an array in the value have no one type")
            }
            BindingRefused::ObjectTooWide { fields } => write!(
                f,
                "an object in the value has {fields} fields, and an object type has at most {}",
                ObjectTy::<crate::ty::Infer>::MAX_FIELDS
            ),
            BindingRefused::IntOutOfRange { value, width } => {
                write!(f, "{value} does not fit in {}", width.name())
            }
        }
    }
}

impl std::error::Error for BindingRefused {}

/// A bound value's type, with the variant each structural enum position
/// of it holds still apart from that position (RFC-0087 rule 2).
pub(crate) struct Typed {
    pub ty: InferTy,
    pub deferred: Vec<DeferredJoin>,
}

/// A structural enum position of a bound value: `var` is the position,
/// which the body's uses type, and `variant` the enum the value's variant
/// alone makes. The join is applied once the uses were checked: an arm
/// naming a variant a known enum lacks is refused (RFC-0051 rule 2), so the
/// value's own enum would refuse the arms the binding decides against.
pub(crate) struct DeferredJoin {
    pub var: InferTy,
    pub variant: InferTy,
}

/// Where a value stands: text is `&str` as the input itself and `String`
/// inside a value, since data holds no reference (RFC-0087 rule 2).
#[derive(Clone, Copy)]
enum Standing {
    Input,
    Data,
}

/// The one typing of a bound value, as the checker types the literal
/// expression that writes it (RFC-0087 rule 2). The deferred joins are
/// returned, not applied.
pub(crate) fn type_bound(
    solver: &mut Solver<'_>,
    value: &BoundValue,
) -> Result<Typed, BindingRefused> {
    let mut deferred = Vec::new();
    let ty = type_at(solver, value, Standing::Input, &mut deferred)?;
    Ok(Typed { ty, deferred })
}

fn type_at(
    solver: &mut Solver<'_>,
    value: &BoundValue,
    standing: Standing,
    deferred: &mut Vec<DeferredJoin>,
) -> Result<InferTy, BindingRefused> {
    let ty = match value {
        BoundValue::Int(_) => TyTerm::I64,
        BoundValue::IntOf { value, width } => {
            if !width.holds(*value) {
                return Err(BindingRefused::IntOutOfRange {
                    value: *value,
                    width: *width,
                });
            }
            TyTerm::Int(*width)
        }
        BoundValue::Float(_) => TyTerm::Float,
        BoundValue::Bool(_) => TyTerm::Bool,
        BoundValue::Char(_) => TyTerm::Char,
        BoundValue::Unit => TyTerm::Unit,
        BoundValue::String(_) => match standing {
            Standing::Input => {
                TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(TyTerm::Str)))
            }
            Standing::Data => TyTerm::String,
        },
        BoundValue::Bytes(bytes) => {
            TyTerm::Array(Box::new(TyTerm::U8), LenTerm::Known(bytes.len()))
        }
        BoundValue::Array(elements) => {
            let element = solver.fresh_ty_var();
            for held in elements {
                let ty = type_at(solver, held, Standing::Data, deferred)?;
                solver
                    .unify(&ty, &element)
                    .map_err(|_| BindingRefused::ElementsDisagree)?;
            }
            TyTerm::Array(Box::new(element), LenTerm::Known(elements.len()))
        }
        BoundValue::Tuple(elements) => TyTerm::Tuple(
            elements
                .iter()
                .map(|held| type_at(solver, held, Standing::Data, deferred))
                .collect::<Result<_, _>>()?,
        ),
        BoundValue::Object(fields) => {
            let mut typed = FxHashMap::default();
            for (key, held) in fields {
                typed.insert(*key, type_at(solver, held, Standing::Data, deferred)?);
            }
            if let Some(fields) = ObjectTy::<crate::ty::Infer>::too_wide(&typed) {
                return Err(BindingRefused::ObjectTooWide { fields });
            }
            solver.construct(TyTerm::Object(ObjectTy::written(typed)))
        }
        BoundValue::Variant { name, tag, payload } => {
            let payload = match payload {
                Some(held) => Some(Box::new(type_at(solver, held, Standing::Data, deferred)?)),
                None => None,
            };
            let mut variants = FxHashMap::default();
            variants.insert(*tag, payload);
            let variant = solver.construct(TyTerm::Enum {
                name: *name,
                variants,
                home: Home::NONE,
            });
            let var = solver.fresh_ty_var();
            deferred.push(DeferredJoin {
                var: var.clone(),
                variant,
            });
            var
        }
        BoundValue::Option(held) => {
            let inner = match held {
                Some(held) => type_at(solver, held, Standing::Data, deferred)?,
                None => solver.fresh_ty_var(),
            };
            TyTerm::Option(Box::new(inner))
        }
        BoundValue::Result(held) => {
            let (ok, err) = match held {
                Ok(held) => (
                    type_at(solver, held, Standing::Data, deferred)?,
                    solver.fresh_ty_var(),
                ),
                Err(held) => (
                    solver.fresh_ty_var(),
                    type_at(solver, held, Standing::Data, deferred)?,
                ),
            };
            TyTerm::Result(Box::new(ok), Box::new(err))
        }
    };
    Ok(ty)
}

/// Whether a value types on its own, apart from any body: typed in a
/// scratch solver, with its deferred joins applied there (RFC-0087
/// rule 3). A join that fails there is two variants an array made one
/// position, which no body can mend.
pub(super) fn admit(value: &BoundValue) -> Result<(), BindingRefused> {
    let mut sources = crate::solver::Sources::new();
    let registry = crate::ty::TypeRegistry::new();
    let signatures = FxHashMap::default();
    let mut solver = Solver::new(&mut sources, &registry, &signatures);
    let typed = type_bound(&mut solver, value)?;
    for DeferredJoin { var, variant } in &typed.deferred {
        solver
            .unify(variant, var)
            .map_err(|_| BindingRefused::ElementsDisagree)?;
    }
    Ok(())
}

// -- The constant ---------------------------------------------------------

/// A parameter the declaration names is filled by its call, so a binding of
/// the same name leaves it standing, as the checker does (RFC-0054 rule 6).
pub fn substitute(
    interner: &Interner,
    body: &mut MirBody,
    declared_params: usize,
    bindings: &Bindings,
) -> Vec<MirError> {
    let mut errors = Vec::new();
    for (name, value) in bindings.iter() {
        let Some(at) = body.params[declared_params..]
            .iter()
            .position(|(held, _)| *held == name)
            .map(|at| declared_params + at)
        else {
            continue;
        };
        let (_, slot) = body.params[at];
        let ty = body
            .val_types
            .get(&slot)
            .cloned()
            .expect("lowering gives every parameter of a body its type");
        let mut insts = Vec::new();
        match constant(interner, body, &ty, value, &mut insts) {
            Ok(written) => {
                body.params.remove(at);
                read_as_local(body, slot);
                let assign = Inst {
                    span: Span::ZERO,
                    kind: InstKind::Assign {
                        target: RefTarget::Var(slot),
                        path: Vec::new(),
                        value: written,
                        restores: false,
                    },
                };
                body.insts
                    .splice(0..0, insts.into_iter().chain(std::iter::once(assign)));
            }
            Err(Mismatch) => errors.push(MirError {
                kind: MirErrorKind::BindingTypeMismatch {
                    name: interner.resolve(name).to_string(),
                    value: value.clone(),
                    ty,
                },
                span: Span::ZERO,
                labels: Vec::new(),
            }),
        }
    }
    errors
}

struct Mismatch;

/// The value's constructors at the type the body closed it to, as the
/// lowering writes the literal (RFC-0087 rule 5): a part is written before
/// what holds it, each at its own type. A value the type does not hold is
/// `Mismatch`: an object the body read a field of it lacks, or an integer
/// the width does not fit.
fn constant(
    interner: &Interner,
    body: &mut MirBody,
    ty: &Ty,
    value: &BoundValue,
    insts: &mut Vec<Inst>,
) -> Result<ValueId, Mismatch> {
    let emit = |insts: &mut Vec<Inst>, kind: InstKind| {
        insts.push(Inst {
            span: Span::ZERO,
            kind,
        })
    };
    let built = match (ty, value) {
        (Ty::Int(width), BoundValue::Int(n)) if width.holds(*n) => {
            ConstKind::Scalar(Literal::Int(*n))
        }
        (Ty::Int(width), BoundValue::IntOf { value, width: held })
            if width == held && width.holds(*value) =>
        {
            ConstKind::Scalar(Literal::Int(*value))
        }
        (Ty::Float, BoundValue::Float(x)) => ConstKind::Scalar(Literal::Float(*x)),
        (Ty::Bool, BoundValue::Bool(b)) => ConstKind::Scalar(Literal::Bool(*b)),
        (Ty::Char, BoundValue::Char(c)) => ConstKind::Scalar(Literal::Char(*c)),
        (Ty::Unit, BoundValue::Unit) => ConstKind::Scalar(Literal::Unit),
        (Ty::Str, BoundValue::String(text)) => ConstKind::Str(text.clone()),
        (Ty::Ref(_, inner), BoundValue::String(text))
            if matches!(*inner.ty(), Ty::Str | Ty::String) =>
        {
            ConstKind::Str(text.clone())
        }
        // `StringClone` reads a `String`, and the lowering reaches a
        // `String` from a literal only through a call; a one-part concat is
        // the instruction that owns the bytes a `&str` constant lends.
        (Ty::String, BoundValue::String(text)) => {
            let borrowed = body.val_factory.next();
            body.val_types.insert(
                borrowed,
                Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str))),
            );
            emit(
                insts,
                InstKind::ConstStr {
                    dst: borrowed,
                    text: text.clone(),
                },
            );
            ConstKind::Owned(borrowed)
        }
        // A byte string is a list of `u8`s, the lowering's one spelling of
        // it (RFC-0058 rule 5).
        (Ty::Array(element, LenTerm::Known(len)), BoundValue::Bytes(bytes))
            if **element == Ty::U8 && *len == bytes.len() =>
        {
            ConstKind::Scalar(Literal::List(
                bytes.iter().map(|b| Literal::Int(i128::from(*b))).collect(),
            ))
        }
        (Ty::Array(element, LenTerm::Known(len)), BoundValue::Array(held))
            if *len == held.len() =>
        {
            return bound_array(interner, body, element, held, insts);
        }
        (Ty::Tuple(types), BoundValue::Tuple(held)) if types.len() == held.len() => {
            let elements = types
                .iter()
                .zip(held)
                .map(|(ty, held)| constant(interner, body, ty, held, insts))
                .collect::<Result<_, _>>()?;
            ConstKind::Tuple(elements)
        }
        (Ty::Object(object), BoundValue::Object(held))
            if object.len() == held.len() && held.keys().all(|key| object.contains_key(key)) =>
        {
            let mut fields = Vec::with_capacity(held.len());
            for (key, held) in held {
                let at = object
                    .get(key)
                    .expect("the field sets were compared equal above");
                fields.push((*key, constant(interner, body, at, held, insts)?));
            }
            ConstKind::Object(fields)
        }
        (
            Ty::Enum { name, variants, .. },
            BoundValue::Variant {
                name: held_name,
                tag,
                payload,
            },
        ) if name == held_name => {
            let Some(payload_ty) = variants.get(tag) else {
                return Err(Mismatch);
            };
            let payload = match (payload_ty, payload) {
                (Some(ty), Some(held)) => Some(constant(interner, body, ty, held, insts)?),
                (None, None) => None,
                _ => return Err(Mismatch),
            };
            ConstKind::Variant { tag: *tag, payload }
        }
        (Ty::Option(inner), BoundValue::Option(held)) => match held {
            Some(held) => ConstKind::Variant {
                tag: interner.intern("Some"),
                payload: Some(constant(interner, body, inner, held, insts)?),
            },
            None => ConstKind::Variant {
                tag: interner.intern("None"),
                payload: None,
            },
        },
        (Ty::Result(ok, err), BoundValue::Result(held)) => match held {
            Ok(held) => ConstKind::Variant {
                tag: interner.intern("Ok"),
                payload: Some(constant(interner, body, ok, held, insts)?),
            },
            Err(held) => ConstKind::Variant {
                tag: interner.intern("Err"),
                payload: Some(constant(interner, body, err, held, insts)?),
            },
        },
        _ => return Err(Mismatch),
    };
    let dst = body.val_factory.next();
    body.val_types.insert(dst, ty.clone());
    let kind = match built {
        ConstKind::Scalar(value) => InstKind::Const { dst, value },
        ConstKind::Str(text) => InstKind::ConstStr { dst, text },
        ConstKind::Owned(part) => InstKind::StringConcat {
            dst,
            parts: vec![part],
        },
        ConstKind::Tuple(elements) => InstKind::MakeTuple { dst, elements },
        ConstKind::Object(fields) => InstKind::MakeObject { dst, fields },
        ConstKind::Variant { tag, payload } => InstKind::MakeVariant { dst, tag, payload },
    };
    emit(insts, kind);
    Ok(dst)
}

/// An array of the bound elements, each written and pushed before the next
/// is written, as the lowering builds an array literal.
fn bound_array(
    interner: &Interner,
    body: &mut MirBody,
    element: &Ty,
    held: &[BoundValue],
    insts: &mut Vec<Inst>,
) -> Result<ValueId, Mismatch> {
    let mut array = body.val_factory.next();
    body.val_types.insert(
        array,
        Ty::Array(Box::new(element.clone()), LenTerm::Known(0)),
    );
    insts.push(Inst {
        span: Span::ZERO,
        kind: InstKind::ArrayBegin {
            dst: array,
            capacity: held.len(),
        },
    });
    for (at, held) in held.iter().enumerate() {
        let value = constant(interner, body, element, held, insts)?;
        let dst = body.val_factory.next();
        body.val_types.insert(
            dst,
            Ty::Array(Box::new(element.clone()), LenTerm::Known(at + 1)),
        );
        insts.push(Inst {
            span: Span::ZERO,
            kind: InstKind::ArrayPush { dst, array, value },
        });
        array = dst;
    }
    Ok(array)
}

/// The instruction that writes one part, its own parts already written.
enum ConstKind {
    Scalar(Literal),
    Str(String),
    Owned(ValueId),
    Tuple(Vec<ValueId>),
    Object(Vec<(Astr, ValueId)>),
    Variant { tag: Astr, payload: Option<ValueId> },
}

fn read_as_local(body: &mut MirBody, slot: ValueId) {
    let targets = body
        .insts
        .iter_mut()
        .filter_map(|inst| match &mut inst.kind {
            InstKind::Ref { target, .. }
            | InstKind::Take { target, .. }
            | InstKind::Assign { target, .. } => Some(target),
            _ => None,
        });
    for target in targets {
        if let RefTarget::Param(held) = target
            && *held == slot
        {
            *target = RefTarget::Var(slot);
        }
    }
}
