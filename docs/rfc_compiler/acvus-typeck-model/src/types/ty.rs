//! Closed types, and the shape of a term that closed types and structures share.
//!
//! A type is a finite term: one constructor, its members and its parts, each
//! part again a type. Two types are equal when their constructors,
//! members and parts are. Each type has one representation, so `Eq` is
//! that equality.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::fmt;
use std::rc::Rc;

/// The names of `Option`'s and `Result`'s variants.
pub const SOME: &str = "Some";
pub const NONE: &str = "None";
pub const OK: &str = "Ok";
pub const ERR: &str = "Err";

/// The name of an extern-defined type, an enum, a struct's module, a field, a
/// variant, a lambda or an identity.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(Rc<str>);

impl Name {
    pub fn new(text: &str) -> Name {
        Name(Rc::from(text))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A reference's mutability, the weaker first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Mutability {
    Shared,
    Exclusive,
}

/// The lesser of two mutabilities, the language's own order `&mut ≤ &`
/// read on mutability, which rule 2 joins two references at, and a lend
/// through a reference and a binding reached through references lend at: `Shared` where either is `Shared`,
/// `Exclusive` where both are, and none where neither is `Shared` and one is
/// not known. It reads no declaration.
pub fn lesser(a: Option<Mutability>, b: Option<Mutability>) -> Option<Mutability> {
    match (a, b) {
        (Some(Mutability::Shared), _) | (_, Some(Mutability::Shared)) => Some(Mutability::Shared),
        (Some(Mutability::Exclusive), Some(Mutability::Exclusive)) => Some(Mutability::Exclusive),
        _ => None,
    }
}

/// The integer types of 8 to 64 bits.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntTy {
    pub fn name(self) -> &'static str {
        match self {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
        }
    }
}

/// The float types `f32` and `f64`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn name(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }
}

/// Rust's scalar types.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Scalar {
    Int(IntTy),
    Float(FloatTy),
    Bool,
    Char,
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Int(int) => f.write_str(int.name()),
            Scalar::Float(float) => f.write_str(float.name()),
            Scalar::Bool => f.write_str("bool"),
            Scalar::Char => f.write_str("char"),
        }
    }
}

/// A constructor, its members and its parts, each part a `P`: a closed type in
/// `Ty`, an equivalence class in a structure.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Term<P> {
    /// `&T` or `&mut T`, whose part is the target.
    Ref(Mutability, P),
    /// `(T₁, …, Tₙ)` with `n ≥ 1`; `()` is `Unit`.
    Tuple(Vec<P>),
    /// `[T; n]`.
    Array(u64, P),
    /// `[T]`.
    Slice(P),
    /// `Option⟨T⟩`, whose one payload `Some` and `None` share.
    Option(P),
    /// `Result⟨T, E⟩`.
    Result {
        ok: P,
        err: P,
    },
    /// `Fn(P̄ → R)`, the type a declaration writes.
    Function {
        params: Vec<P>,
        ret: P,
    },
    /// The type of a function value. Its parameters and result lie in its
    /// lambdas' binding groups, and Values names those lambdas.
    FunctionValue {
        arity: usize,
    },
    /// `{f̄: T̄}`, by field.
    Object(BTreeMap<Name, P>),
    /// `E{V̄(T̄)}`, by variant.
    Enum(Name, BTreeMap<Name, P>),
    /// `Error⟨E⟩`, the `Error` with no variant and one part, the value it
    /// wraps.
    ErrorOf(P),
    /// A module's struct, with one part for each field in declared order.
    Struct(Name, Vec<P>),
    /// `N⟨T̄, ι⟩`.
    Extern {
        name: Name,
        args: Vec<P>,
        identity: Option<P>,
    },
    Scalar(Scalar),
    String,
    Str,
    Unit,
    /// `!`.
    Never,
    /// The name of the storage an extern-defined type's values reach:
    /// a type with no parts, lying only at that type's identity.
    Identity(Name),
}

/// What remains of a term without its members and parts.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Constructor {
    Ref(Mutability),
    Tuple(usize),
    Array(u64),
    Slice,
    Option,
    Result,
    Function(usize),
    Object,
    Enum(Name),
    /// `Error⟨E⟩`'s, which is not that of an `Error` with variants.
    ErrorOf,
    Struct(Name),
    Extern(Name),
    Scalar(Scalar),
    String,
    Str,
    Unit,
    Never,
    Identity(Name),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Member {
    Field(Name),
    Variant(Name),
}

/// The name of a part. Parts of one name in two terms of one constructor
/// correspond.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PartKey {
    Target,
    Component(usize),
    Element,
    Payload(Name),
    Param(usize),
    Return,
    Field(Name),
    Arg(usize),
    Identity,
    /// `Error⟨E⟩`'s one part.
    Wrapped,
}

impl<P> Term<P> {
    pub fn constructor(&self) -> Constructor {
        match self {
            Term::Ref(mutability, _) => Constructor::Ref(*mutability),
            Term::Tuple(components) => Constructor::Tuple(components.len()),
            Term::Array(len, _) => Constructor::Array(*len),
            Term::Slice(_) => Constructor::Slice,
            Term::Option(_) => Constructor::Option,
            Term::Result { .. } => Constructor::Result,
            Term::Function { params, .. } => Constructor::Function(params.len()),
            Term::FunctionValue { arity } => Constructor::Function(*arity),
            Term::Object(_) => Constructor::Object,
            Term::Enum(name, _) => Constructor::Enum(name.clone()),
            Term::ErrorOf(_) => Constructor::ErrorOf,
            Term::Struct(module, _) => Constructor::Struct(module.clone()),
            Term::Extern { name, .. } => Constructor::Extern(name.clone()),
            Term::Scalar(scalar) => Constructor::Scalar(*scalar),
            Term::String => Constructor::String,
            Term::Str => Constructor::Str,
            Term::Unit => Constructor::Unit,
            Term::Never => Constructor::Never,
            Term::Identity(name) => Constructor::Identity(name.clone()),
        }
    }

    /// An object's fields, an enum's variants, `Option`'s `Some` and `None`,
    /// and `Result`'s `Ok` and `Err`.
    pub fn members(&self) -> BTreeSet<Member> {
        match self {
            Term::Option(_) => [SOME, NONE].map(|v| Member::Variant(Name::new(v))).into(),
            Term::Result { .. } => [OK, ERR].map(|v| Member::Variant(Name::new(v))).into(),
            Term::Object(fields) => fields.keys().cloned().map(Member::Field).collect(),
            Term::Enum(_, variants) => variants.keys().cloned().map(Member::Variant).collect(),
            _ => BTreeSet::new(),
        }
    }

    /// Each part with its name, in the order the term's notation writes them.
    pub fn parts(&self) -> Vec<(PartKey, &P)> {
        match self {
            Term::Ref(_, target) => vec![(PartKey::Target, target)],
            Term::Tuple(components) => components
                .iter()
                .enumerate()
                .map(|(i, component)| (PartKey::Component(i), component))
                .collect(),
            Term::Array(_, element) | Term::Slice(element) => vec![(PartKey::Element, element)],
            Term::Option(payload) => vec![(PartKey::Payload(Name::new(SOME)), payload)],
            Term::Result { ok, err } => vec![
                (PartKey::Payload(Name::new(OK)), ok),
                (PartKey::Payload(Name::new(ERR)), err),
            ],
            Term::Function { params, ret } => params
                .iter()
                .enumerate()
                .map(|(i, param)| (PartKey::Param(i), param))
                .chain([(PartKey::Return, ret)])
                .collect(),
            Term::Object(fields) => fields
                .iter()
                .map(|(field, ty)| (PartKey::Field(field.clone()), ty))
                .collect(),
            Term::Enum(_, variants) => variants
                .iter()
                .map(|(variant, payload)| (PartKey::Payload(variant.clone()), payload))
                .collect(),
            Term::ErrorOf(wrapped) => vec![(PartKey::Wrapped, wrapped)],
            Term::Struct(_, args) => args
                .iter()
                .enumerate()
                .map(|(i, arg)| (PartKey::Arg(i), arg))
                .collect(),
            Term::Extern { args, identity, .. } => args
                .iter()
                .enumerate()
                .map(|(i, arg)| (PartKey::Arg(i), arg))
                .chain(
                    identity
                        .iter()
                        .map(|identity| (PartKey::Identity, identity)),
                )
                .collect(),
            Term::FunctionValue { .. }
            | Term::Scalar(_)
            | Term::String
            | Term::Str
            | Term::Unit
            | Term::Never
            | Term::Identity(_) => Vec::new(),
        }
    }

    pub fn part(&self, key: &PartKey) -> Option<&P> {
        self.parts()
            .into_iter()
            .find(|(name, _)| name == key)
            .map(|(_, part)| part)
    }

    /// The same term with each part replaced by `f` of its name and itself,
    /// `f` called in the order of `parts`.
    pub fn map<Q, F>(self, mut f: F) -> Term<Q>
    where
        F: FnMut(PartKey, P) -> Q,
    {
        let mapped: Result<Term<Q>, Infallible> = self.try_map(|key, part| Ok(f(key, part)));
        match mapped {
            Ok(term) => term,
        }
    }

    /// `map` with an `f` that may fail, stopping at the first part it fails at.
    pub fn try_map<Q, E, F>(self, mut f: F) -> Result<Term<Q>, E>
    where
        F: FnMut(PartKey, P) -> Result<Q, E>,
    {
        Ok(match self {
            Term::Ref(mutability, target) => Term::Ref(mutability, f(PartKey::Target, target)?),
            Term::Tuple(components) => Term::Tuple(
                components
                    .into_iter()
                    .enumerate()
                    .map(|(i, component)| f(PartKey::Component(i), component))
                    .collect::<Result<_, E>>()?,
            ),
            Term::Array(len, element) => Term::Array(len, f(PartKey::Element, element)?),
            Term::Slice(element) => Term::Slice(f(PartKey::Element, element)?),
            Term::Option(payload) => Term::Option(f(PartKey::Payload(Name::new(SOME)), payload)?),
            Term::Result { ok, err } => {
                let ok = f(PartKey::Payload(Name::new(OK)), ok)?;
                let err = f(PartKey::Payload(Name::new(ERR)), err)?;
                Term::Result { ok, err }
            }
            Term::Function { params, ret } => {
                let params = params
                    .into_iter()
                    .enumerate()
                    .map(|(i, param)| f(PartKey::Param(i), param))
                    .collect::<Result<_, E>>()?;
                let ret = f(PartKey::Return, ret)?;
                Term::Function { params, ret }
            }
            Term::FunctionValue { arity } => Term::FunctionValue { arity },
            Term::Object(fields) => Term::Object(
                fields
                    .into_iter()
                    .map(|(field, ty)| {
                        let ty = f(PartKey::Field(field.clone()), ty)?;
                        Ok((field, ty))
                    })
                    .collect::<Result<_, E>>()?,
            ),
            Term::Enum(name, variants) => Term::Enum(
                name,
                variants
                    .into_iter()
                    .map(|(variant, payload)| {
                        let payload = f(PartKey::Payload(variant.clone()), payload)?;
                        Ok((variant, payload))
                    })
                    .collect::<Result<_, E>>()?,
            ),
            Term::ErrorOf(wrapped) => Term::ErrorOf(f(PartKey::Wrapped, wrapped)?),
            Term::Struct(module, args) => Term::Struct(
                module,
                args.into_iter()
                    .enumerate()
                    .map(|(i, arg)| f(PartKey::Arg(i), arg))
                    .collect::<Result<_, E>>()?,
            ),
            Term::Extern {
                name,
                args,
                identity,
            } => {
                let args = args
                    .into_iter()
                    .enumerate()
                    .map(|(i, arg)| f(PartKey::Arg(i), arg))
                    .collect::<Result<_, E>>()?;
                let identity = match identity {
                    Some(identity) => Some(f(PartKey::Identity, identity)?),
                    None => None,
                };
                Term::Extern {
                    name,
                    args,
                    identity,
                }
            }
            Term::Scalar(scalar) => Term::Scalar(scalar),
            Term::String => Term::String,
            Term::Str => Term::Str,
            Term::Unit => Term::Unit,
            Term::Never => Term::Never,
            Term::Identity(name) => Term::Identity(name),
        })
    }
}

impl Constructor {
    /// Object, enum, tuple, array, option and result, `Error⟨E⟩`
    /// among the enums as an `Error`. The parts of a structural
    /// constructor are joined without casts, and every part of any other
    /// constructor is invariant.
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Constructor::Object
                | Constructor::Enum(_)
                | Constructor::ErrorOf
                | Constructor::Tuple(_)
                | Constructor::Array(_)
                | Constructor::Option
                | Constructor::Result
        )
    }

    /// The part that holds `variant`'s payload. `Some` and `None` share
    /// `Option`'s one payload.
    pub fn payload_part(&self, variant: &Name) -> Option<PartKey> {
        match (self, variant.as_str()) {
            (Constructor::Option, SOME | NONE) => Some(PartKey::Payload(Name::new(SOME))),
            (Constructor::Result, OK | ERR) | (Constructor::Enum(_), _) => {
                Some(PartKey::Payload(variant.clone()))
            }
            _ => None,
        }
    }
}

/// A closed type: a term whose parts are closed types.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ty(Rc<Term<Ty>>);

/// Why a term is not a closed type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TyError {
    /// A reference whose target is a reference.
    RefTarget,
    /// A tuple of no component, which is `Unit`.
    EmptyTuple,
    /// An identity at a part other than an extern-defined type's identity, or
    /// that part holding anything but an identity.
    IdentityPlace,
}

/// A type each of whose positions holds a term or is not decided: a closed
/// type, every position of which holds one, or a value a use reads.
pub trait HasTerm: Clone + Eq + Sized {
    /// The term at the outermost position, where one is decided.
    fn decided(&self) -> Option<&Term<Self>>;

    /// The type whose outermost term is `term`, refused where `check_term`
    /// refuses it.
    fn from_term(term: Term<Self>) -> Result<Self, TyError>;
}

/// The parts of `term` are checked already, so checking its own parts checks
/// the whole tree. A part inside a reference's target is not restricted:
/// `&N⟨&U⟩` is a type. A position not decided is refused nowhere.
pub fn check_term<P>(term: &Term<P>) -> Result<(), TyError>
where
    P: HasTerm,
{
    if let Term::Ref(_, target) = term
        && matches!(target.decided(), Some(Term::Ref(..)))
    {
        return Err(TyError::RefTarget);
    }
    if matches!(term, Term::Tuple(components) if components.is_empty()) {
        return Err(TyError::EmptyTuple);
    }
    for (key, part) in term.parts() {
        let Some(part) = part.decided() else {
            continue;
        };
        let is_identity = matches!(part, Term::Identity(_));
        if is_identity != (key == PartKey::Identity) {
            return Err(TyError::IdentityPlace);
        }
    }
    Ok(())
}

impl Ty {
    pub fn new(term: Term<Ty>) -> Result<Ty, TyError> {
        check_term(&term)?;
        Ok(Ty(Rc::new(term)))
    }

    pub fn term(&self) -> &Term<Ty> {
        &self.0
    }
}

impl HasTerm for Ty {
    fn decided(&self) -> Option<&Term<Ty>> {
        Some(self.term())
    }

    fn from_term(term: Term<Ty>) -> Result<Ty, TyError> {
        Ty::new(term)
    }
}

impl<P> fmt::Display for Term<P>
where
    P: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Ref(Mutability::Shared, target) => write!(f, "&{target}"),
            Term::Ref(Mutability::Exclusive, target) => write!(f, "&mut {target}"),
            Term::Tuple(components) if components.len() == 1 => write!(f, "({},)", components[0]),
            Term::Tuple(components) => {
                f.write_str("(")?;
                write_listed(f, components)?;
                f.write_str(")")
            }
            Term::Array(len, element) => write!(f, "[{element}; {len}]"),
            Term::Slice(element) => write!(f, "[{element}]"),
            Term::Option(payload) => write!(f, "Option<{payload}>"),
            Term::Result { ok, err } => write!(f, "Result<{ok}, {err}>"),
            Term::Function { params, ret } => {
                f.write_str("Fn(")?;
                write_listed(f, params)?;
                write!(f, " -> {ret})")
            }
            Term::FunctionValue { arity } => write!(f, "Fn/{arity}"),
            Term::Object(fields) => {
                f.write_str("{")?;
                write_listed(f, fields.iter().map(|(field, ty)| format!("{field}: {ty}")))?;
                f.write_str("}")
            }
            Term::Enum(name, variants) => {
                write!(f, "{name}{{")?;
                let variants = variants
                    .iter()
                    .map(|(variant, payload)| format!("{variant}({payload})"));
                write_listed(f, variants)?;
                f.write_str("}")
            }
            Term::ErrorOf(wrapped) => write!(f, "Error<{wrapped}>"),
            Term::Struct(module, args) => {
                write!(f, "{module}")?;
                if !args.is_empty() {
                    f.write_str("<")?;
                    write_listed(f, args)?;
                    f.write_str(">")?;
                }
                Ok(())
            }
            Term::Extern {
                name,
                args,
                identity,
            } => {
                write!(f, "{name}")?;
                let parts: Vec<&P> = args.iter().chain(identity).collect();
                if !parts.is_empty() {
                    f.write_str("<")?;
                    write_listed(f, parts)?;
                    f.write_str(">")?;
                }
                Ok(())
            }
            Term::Scalar(scalar) => write!(f, "{scalar}"),
            Term::String => f.write_str("String"),
            Term::Str => f.write_str("str"),
            Term::Unit => f.write_str("()"),
            Term::Never => f.write_str("!"),
            Term::Identity(name) => write!(f, "{name}"),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.term(), f)
    }
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn write_listed<I>(f: &mut fmt::Formatter<'_>, items: I) -> fmt::Result
where
    I: IntoIterator,
    I::Item: fmt::Display,
{
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    #[test]
    fn lesser_mutability_combines_correctly() {
        assert_eq!(
            lesser(Some(Mutability::Shared), Some(Mutability::Exclusive)),
            Some(Mutability::Shared)
        );
        assert_eq!(
            lesser(Some(Mutability::Exclusive), Some(Mutability::Exclusive)),
            Some(Mutability::Exclusive)
        );
        assert_eq!(lesser(Some(Mutability::Exclusive), None), None);
        assert_eq!(
            lesser(Some(Mutability::Shared), None),
            Some(Mutability::Shared)
        );
    }

    fn ty(term: Term<Ty>) -> Ty {
        Ty::new(term).expect("the term is a closed type")
    }

    fn name(text: &str) -> Name {
        Name::new(text)
    }

    fn int(int: IntTy) -> Ty {
        ty(Term::Scalar(Scalar::Int(int)))
    }

    fn identity(text: &str) -> Ty {
        ty(Term::Identity(name(text)))
    }

    fn extern_ty(text: &str, args: Vec<Ty>, identity: Option<Ty>) -> Ty {
        ty(Term::Extern {
            name: name(text),
            args,
            identity,
        })
    }

    /// One type, with what the shape table gives it, and its notation.
    struct Shape {
        ty: Ty,
        constructor: Constructor,
        members: Vec<Member>,
        parts: Vec<PartKey>,
        structural: bool,
        notation: &'static str,
    }

    fn shapes() -> Vec<Shape> {
        let u8_ = int(IntTy::U8);
        let u16_ = int(IntTy::U16);
        let bool_ = ty(Term::Scalar(Scalar::Bool));
        let field = |text| Member::Field(name(text));
        let variant = |text| Member::Variant(name(text));
        let payload = |text| PartKey::Payload(name(text));
        vec![
            Shape {
                ty: ty(Term::Ref(Mutability::Exclusive, u8_.clone())),
                constructor: Constructor::Ref(Mutability::Exclusive),
                members: vec![],
                parts: vec![PartKey::Target],
                structural: false,
                notation: "&mut u8",
            },
            Shape {
                ty: ty(Term::Tuple(vec![u8_.clone(), u16_.clone()])),
                constructor: Constructor::Tuple(2),
                members: vec![],
                parts: vec![PartKey::Component(0), PartKey::Component(1)],
                structural: true,
                notation: "(u8, u16)",
            },
            Shape {
                ty: ty(Term::Tuple(vec![u8_.clone()])),
                constructor: Constructor::Tuple(1),
                members: vec![],
                parts: vec![PartKey::Component(0)],
                structural: true,
                notation: "(u8,)",
            },
            Shape {
                ty: ty(Term::Array(4, u8_.clone())),
                constructor: Constructor::Array(4),
                members: vec![],
                parts: vec![PartKey::Element],
                structural: true,
                notation: "[u8; 4]",
            },
            Shape {
                ty: ty(Term::Slice(u8_.clone())),
                constructor: Constructor::Slice,
                members: vec![],
                parts: vec![PartKey::Element],
                structural: false,
                notation: "[u8]",
            },
            Shape {
                ty: ty(Term::Option(u8_.clone())),
                constructor: Constructor::Option,
                members: vec![variant("None"), variant("Some")],
                parts: vec![payload("Some")],
                structural: true,
                notation: "Option<u8>",
            },
            Shape {
                ty: ty(Term::Result {
                    ok: u8_.clone(),
                    err: u16_.clone(),
                }),
                constructor: Constructor::Result,
                members: vec![variant("Err"), variant("Ok")],
                parts: vec![payload("Ok"), payload("Err")],
                structural: true,
                notation: "Result<u8, u16>",
            },
            Shape {
                ty: ty(Term::Function {
                    params: vec![u8_.clone(), u16_.clone()],
                    ret: bool_.clone(),
                }),
                constructor: Constructor::Function(2),
                members: vec![],
                parts: vec![PartKey::Param(0), PartKey::Param(1), PartKey::Return],
                structural: false,
                notation: "Fn(u8, u16 -> bool)",
            },
            Shape {
                ty: ty(Term::FunctionValue { arity: 2 }),
                constructor: Constructor::Function(2),
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "Fn/2",
            },
            Shape {
                ty: ty(Term::Object(
                    [(name("y"), u16_.clone()), (name("x"), u8_.clone())].into(),
                )),
                constructor: Constructor::Object,
                members: vec![field("x"), field("y")],
                parts: vec![PartKey::Field(name("x")), PartKey::Field(name("y"))],
                structural: true,
                notation: "{x: u8, y: u16}",
            },
            Shape {
                ty: ty(Term::Enum(
                    name("E"),
                    [(name("B"), u8_.clone()), (name("A"), ty(Term::Never))].into(),
                )),
                constructor: Constructor::Enum(name("E")),
                members: vec![variant("A"), variant("B")],
                parts: vec![payload("A"), payload("B")],
                structural: true,
                notation: "E{A(!), B(u8)}",
            },
            Shape {
                ty: ty(Term::ErrorOf(u8_.clone())),
                constructor: Constructor::ErrorOf,
                members: vec![],
                parts: vec![PartKey::Wrapped],
                structural: true,
                notation: "Error<u8>",
            },
            Shape {
                ty: ty(Term::Struct(
                    name("Pair"),
                    vec![
                        u8_.clone(),
                        ty(Term::Ref(Mutability::Shared, ty(Term::Str))),
                    ],
                )),
                constructor: Constructor::Struct(name("Pair")),
                members: vec![],
                parts: vec![PartKey::Arg(0), PartKey::Arg(1)],
                structural: false,
                notation: "Pair<u8, &str>",
            },
            Shape {
                ty: extern_ty("F", vec![u8_.clone()], Some(identity("ι"))),
                constructor: Constructor::Extern(name("F")),
                members: vec![],
                parts: vec![PartKey::Arg(0), PartKey::Identity],
                structural: false,
                notation: "F<u8, ι>",
            },
            Shape {
                ty: bool_,
                constructor: Constructor::Scalar(Scalar::Bool),
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "bool",
            },
            Shape {
                ty: ty(Term::String),
                constructor: Constructor::String,
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "String",
            },
            Shape {
                ty: ty(Term::Str),
                constructor: Constructor::Str,
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "str",
            },
            Shape {
                ty: ty(Term::Unit),
                constructor: Constructor::Unit,
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "()",
            },
            Shape {
                ty: ty(Term::Never),
                constructor: Constructor::Never,
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "!",
            },
            Shape {
                ty: identity("ι"),
                constructor: Constructor::Identity(name("ι")),
                members: vec![],
                parts: vec![],
                structural: false,
                notation: "ι",
            },
        ]
    }

    /// The position of `term`'s variant in `Term`. A variant added to `Term`
    /// stops this from compiling until `shapes` is given one of it.
    fn variant_index(term: &Term<Ty>) -> usize {
        match term {
            Term::Ref(..) => 0,
            Term::Tuple(_) => 1,
            Term::Array(..) => 2,
            Term::Slice(_) => 3,
            Term::Option(_) => 4,
            Term::Result { .. } => 5,
            Term::Function { .. } => 6,
            Term::FunctionValue { .. } => 7,
            Term::Object(_) => 8,
            Term::Enum(..) => 9,
            Term::Struct(..) => 10,
            Term::Extern { .. } => 11,
            Term::Scalar(_) => 12,
            Term::String => 13,
            Term::Str => 14,
            Term::Unit => 15,
            Term::Never => 16,
            Term::Identity(_) => 17,
            Term::ErrorOf(_) => 18,
        }
    }

    #[test]
    fn shapes_cover_every_variant_of_term() {
        let covered: BTreeSet<usize> = shapes()
            .iter()
            .map(|shape| variant_index(shape.ty.term()))
            .collect();
        assert_eq!(covered, (0..19).collect::<BTreeSet<usize>>());
    }

    #[test]
    fn each_constructor_has_its_members_and_parts() {
        for shape in shapes() {
            let term = shape.ty.term();
            assert_eq!(term.constructor(), shape.constructor, "{}", shape.ty);
            assert_eq!(
                term.members().into_iter().collect::<Vec<_>>(),
                shape.members,
                "{}",
                shape.ty
            );
            let parts: Vec<PartKey> = term.parts().into_iter().map(|(key, _)| key).collect();
            assert_eq!(parts, shape.parts, "{}", shape.ty);
        }
    }

    #[test]
    fn map_names_each_part_as_parts_does_and_keeps_the_rest() {
        for shape in shapes() {
            let mut named = Vec::new();
            let mapped = shape.ty.term().clone().map(|key, part| {
                named.push(key);
                part
            });
            assert_eq!(named, shape.parts, "{}", shape.ty);
            assert_eq!(&mapped, shape.ty.term(), "{}", shape.ty);
        }
    }

    #[test]
    fn part_finds_each_part_by_its_name_and_no_other() {
        for shape in shapes() {
            let term = shape.ty.term();
            for (key, part) in term.parts() {
                assert_eq!(term.part(&key), Some(part), "{}", shape.ty);
            }
            assert_eq!(term.part(&PartKey::Arg(7)), None, "{}", shape.ty);
        }
    }

    #[test]
    fn types_built_apart_are_equal_by_constructor_members_and_parts() {
        let u8_ = int(IntTy::U8);
        assert_eq!(
            ty(Term::Option(u8_.clone())),
            ty(Term::Option(int(IntTy::U8)))
        );
        assert_ne!(
            ty(Term::Option(u8_.clone())),
            ty(Term::Option(int(IntTy::U16)))
        );
        assert_ne!(
            ty(Term::Array(4, u8_.clone())),
            ty(Term::Array(5, u8_.clone()))
        );
        assert_ne!(
            ty(Term::Ref(Mutability::Shared, u8_.clone())),
            ty(Term::Ref(Mutability::Exclusive, u8_.clone()))
        );
        assert_ne!(
            ty(Term::Object([(name("x"), u8_.clone())].into())),
            ty(Term::Object([(name("y"), u8_.clone())].into()))
        );
    }

    #[test]
    fn an_object_has_one_representation_whatever_order_its_fields_come_in() {
        let fields = [(name("x"), int(IntTy::U8)), (name("y"), int(IntTy::U16))];
        let forward: BTreeMap<Name, Ty> = fields.clone().into_iter().collect();
        let backward: BTreeMap<Name, Ty> = fields.into_iter().rev().collect();
        assert_eq!(ty(Term::Object(forward)), ty(Term::Object(backward)));
    }

    #[test]
    fn a_function_value_is_not_the_written_function_type_of_its_arity() {
        let value = ty(Term::FunctionValue { arity: 1 });
        let written = ty(Term::Function {
            params: vec![int(IntTy::U8)],
            ret: int(IntTy::U8),
        });
        assert_eq!(value.term().constructor(), written.term().constructor());
        assert_ne!(value, written);
        assert_eq!(value, ty(Term::FunctionValue { arity: 1 }));
    }

    #[test]
    fn a_tuple_of_no_component_is_refused() {
        assert_eq!(Ty::new(Term::Tuple(Vec::new())), Err(TyError::EmptyTuple));
    }

    #[test]
    fn a_reference_to_a_reference_is_refused_but_not_a_reference_inside_a_target() {
        let shared = ty(Term::Ref(Mutability::Shared, int(IntTy::U8)));
        assert_eq!(
            Ty::new(Term::Ref(Mutability::Exclusive, shared.clone())),
            Err(TyError::RefTarget)
        );
        let holder = extern_ty("N", vec![shared], None);
        assert!(Ty::new(Term::Ref(Mutability::Shared, holder)).is_ok());
    }

    #[test]
    fn an_identity_lies_only_at_an_extern_defined_types_identity() {
        assert!(
            Ty::new(Term::Extern {
                name: name("F"),
                args: vec![],
                identity: Some(identity("ι")),
            })
            .is_ok()
        );
        assert_eq!(
            Ty::new(Term::Extern {
                name: name("F"),
                args: vec![identity("ι")],
                identity: None,
            }),
            Err(TyError::IdentityPlace)
        );
        assert_eq!(
            Ty::new(Term::Extern {
                name: name("F"),
                args: vec![],
                identity: Some(int(IntTy::U8)),
            }),
            Err(TyError::IdentityPlace)
        );
        assert_eq!(
            Ty::new(Term::Option(identity("ι"))),
            Err(TyError::IdentityPlace)
        );
    }

    #[test]
    fn identities_are_equal_exactly_when_their_names_are() {
        assert_eq!(identity("ι₀"), identity("ι₀"));
        assert_ne!(identity("ι₀"), identity("ι₁"));
    }

    #[test]
    fn object_enum_tuple_array_option_and_result_alone_are_structural() {
        for shape in shapes() {
            assert_eq!(
                shape.constructor.is_structural(),
                shape.structural,
                "{}",
                shape.ty
            );
        }
    }

    #[test]
    fn each_variant_has_the_part_its_payload_lies_in() {
        let payload = |text| Some(PartKey::Payload(name(text)));
        assert_eq!(
            Constructor::Option.payload_part(&name("Some")),
            payload("Some")
        );
        assert_eq!(
            Constructor::Option.payload_part(&name("None")),
            payload("Some")
        );
        assert_eq!(Constructor::Option.payload_part(&name("Ok")), None);
        assert_eq!(Constructor::Result.payload_part(&name("Ok")), payload("Ok"));
        assert_eq!(
            Constructor::Result.payload_part(&name("Err")),
            payload("Err")
        );
        assert_eq!(Constructor::Result.payload_part(&name("Some")), None);
        assert_eq!(
            Constructor::Enum(name("E")).payload_part(&name("A")),
            payload("A")
        );
        assert_eq!(
            Constructor::Struct(name("Pair")).payload_part(&name("A")),
            None
        );
    }

    #[test]
    fn the_payload_part_of_a_variant_is_a_part_of_its_term() {
        for shape in shapes() {
            let term = shape.ty.term();
            for member in term.members() {
                if let Member::Variant(variant) = member {
                    let key = term
                        .constructor()
                        .payload_part(&variant)
                        .expect("a variant has a payload part");
                    assert!(term.part(&key).is_some(), "{} {variant}", shape.ty);
                }
            }
        }
    }

    #[test]
    fn display_writes_the_specs_notation_in_ascii() {
        for shape in shapes() {
            assert_eq!(shape.ty.to_string(), shape.notation);
        }
    }
}
