//! Closed types, and the shape of a term that closed types and kinds share
//! (l.39–74).
//!
//! A type is a finite term: one constructor, its members and its parts, each
//! part again a type (l.41). Two types are equal when their constructors,
//! members and parts are (l.66). Each type has one representation, so `Eq` is
//! that equality.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::fmt;
use std::rc::Rc;

#[cfg(test)]
mod tests;

/// The names of `Option`'s and `Result`'s variants (l.49–50).
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

/// A reference's mutability, the weaker first (l.62).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Mutability {
    Shared,
    Exclusive,
}

/// Rust's integer types (l.56).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

impl IntTy {
    pub fn name(self) -> &'static str {
        match self {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
            IntTy::Isize => "isize",
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
            IntTy::U128 => "u128",
            IntTy::Usize => "usize",
        }
    }
}

/// Rust's float types (l.56).
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

/// Rust's scalar types (l.56).
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
/// `Ty`, an equivalence class in a kind (l.43–60, l.125).
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
    /// lambdas' binding groups, and Values names those lambdas (l.193).
    FunctionValue {
        arity: usize,
    },
    /// `{f̄: T̄}`, by field.
    Object(BTreeMap<Name, P>),
    /// `E{V̄(T̄)}`, by variant.
    Enum(Name, BTreeMap<Name, P>),
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
    /// The name of the storage an extern-defined type's values reach (l.352):
    /// a type with no parts, lying only at that type's identity.
    Identity(Name),
}

/// What remains of a term without its members and parts (l.66).
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
/// correspond (l.165).
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variance {
    Covariant,
    Invariant,
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
    /// and `Result`'s `Ok` and `Err` (l.49–53).
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
    /// Object, enum, tuple, array, option and result (l.68).
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Constructor::Object
                | Constructor::Enum(_)
                | Constructor::Tuple(_)
                | Constructor::Array(_)
                | Constructor::Option
                | Constructor::Result
        )
    }

    /// Every part of a structural constructor is covariant, and every part of
    /// any other constructor is invariant (l.74).
    pub fn variance(&self) -> Variance {
        if self.is_structural() {
            Variance::Covariant
        } else {
            Variance::Invariant
        }
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
    /// A reference whose target is a reference (l.64).
    RefTarget,
    /// A tuple of no component, which is `Unit` (l.59).
    EmptyTuple,
    /// An identity at a part other than an extern-defined type's identity, or
    /// that part holding anything but an identity (l.55, l.352).
    IdentityPlace,
}

impl Ty {
    /// The parts of `term` are closed types already, so checking `term`'s own
    /// parts checks the whole tree. A part inside a reference's target is not
    /// restricted: `&N⟨&U⟩` is a type (l.64).
    pub fn new(term: Term<Ty>) -> Result<Ty, TyError> {
        if let Term::Ref(_, target) = &term
            && matches!(target.term(), Term::Ref(..))
        {
            return Err(TyError::RefTarget);
        }
        if matches!(&term, Term::Tuple(components) if components.is_empty()) {
            return Err(TyError::EmptyTuple);
        }
        for (key, part) in term.parts() {
            let is_identity = matches!(part.term(), Term::Identity(_));
            if is_identity != (key == PartKey::Identity) {
                return Err(TyError::IdentityPlace);
            }
        }
        Ok(Ty(Rc::new(term)))
    }

    pub fn term(&self) -> &Term<Ty> {
        &self.0
    }
}

/// The spec's notation, with a function value written `Fn/n` for its arity,
/// since the spec writes none for it.
impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.term() {
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
            Term::Option(payload) => write!(f, "Option⟨{payload}⟩"),
            Term::Result { ok, err } => write!(f, "Result⟨{ok}, {err}⟩"),
            Term::Function { params, ret } => {
                f.write_str("Fn(")?;
                write_listed(f, params)?;
                write!(f, " → {ret})")
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
            Term::Struct(module, args) => {
                write!(f, "{module}")?;
                if !args.is_empty() {
                    f.write_str("⟨")?;
                    write_listed(f, args)?;
                    f.write_str("⟩")?;
                }
                Ok(())
            }
            Term::Extern {
                name,
                args,
                identity,
            } => {
                write!(f, "{name}")?;
                let parts: Vec<&Ty> = args.iter().chain(identity).collect();
                if !parts.is_empty() {
                    f.write_str("⟨")?;
                    write_listed(f, parts)?;
                    f.write_str("⟩")?;
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
