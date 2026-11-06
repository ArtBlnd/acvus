//! Types: constructors, members and parts.

use std::collections::{BTreeMap, BTreeSet};

use super::{ClosureId, Name};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Mutability {
    /// `&`
    Shared,
    /// `&mut`
    Exclusive,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Scalar {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// Its width is the host's, and it is a type other than the integer of
    /// that width.
    Usize,
    F32,
    F64,
    Bool,
    Char,
}

/// One layer of a type: its constructor and its members, with a `P` at each
/// of its parts.
///
/// The members and parts are held by the constructor they belong to, so no
/// combination of members or parts the constructor does not have can be
/// made. One shape serves several places by its `P`: a type the text writes
/// is `Shape<Ty>`, and a decided class is `Shape<ClassId>`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Shape<P> {
    /// `&T`, `&mut T`. Part: the target.
    Ref(Mutability, P),
    /// `(T₁, …, Tₙ)`. Parts: each component.
    Tuple(Vec<P>),
    /// `[T; n]`. Part: the element.
    Array(usize, P),
    /// `[T]`. Part: the element.
    Slice(P),
    /// `Option⟨T⟩`. Part: the payload of `Some`.
    Option(P),
    /// `Result⟨T, E⟩`. Parts: the payloads of `Ok` and `Err`.
    Result(P, P),
    /// `Error{V̄(T̄)}`. Members: its variants. Parts: each payload.
    Error(BTreeMap<Name, P>),
    /// `Error⟨E⟩`. Part: `E`.
    ErrorOf(P),
    /// `Fn(P̄ → R)`. Parts: each parameter, the result.
    Fn(Vec<P>, P),
    /// `λ̄`. Members: its closures. No parts, as each use makes them anew
    /// (O26).
    FnValue(BTreeSet<ClosureId>),
    /// `{f̄: T̄}`. Members: its fields. Parts: each field's type.
    Object(BTreeMap<Name, P>),
    /// `E{V̄(T̄)}`. Members: its variants. Parts: each payload.
    Enum(Name, BTreeMap<Name, P>),
    /// `Self⟨T̄⟩`, named by its module. Parts: one type parameter for each
    /// field, in the order the struct declares them.
    Struct(Name, Vec<P>),
    /// `N⟨T̄, ι⟩`. Parts: each argument, and the brand type.
    Extern(Name, Vec<P>, P),
    Scalar(Scalar),
    String,
    Str,
    /// `()`
    Unit,
    /// `!`
    Never,
}

impl<P> Shape<P> {
    /// Each part, with the key that names it, in the order the shape holds
    /// them. Two shapes of one layer give their keys in one order.
    pub fn places(&self) -> Vec<(PartKey, &P)> {
        let payload = |variant: &str| PartKey::Payload(Name::new(variant));
        match self {
            Shape::Ref(_, target) => vec![(PartKey::Target, target)],
            Shape::Tuple(components) => components
                .iter()
                .enumerate()
                .map(|(index, component)| (PartKey::Component(index), component))
                .collect(),
            Shape::Array(_, elem) | Shape::Slice(elem) => vec![(PartKey::Elem, elem)],
            Shape::Option(some) => vec![(payload("Some"), some)],
            Shape::Result(ok, err) => vec![(payload("Ok"), ok), (payload("Err"), err)],
            Shape::Error(variants) | Shape::Enum(_, variants) => variants
                .iter()
                .map(|(variant, payload)| (PartKey::Payload(variant.clone()), payload))
                .collect(),
            Shape::ErrorOf(error) => vec![(PartKey::Arg(0), error)],
            Shape::Fn(params, result) => params
                .iter()
                .enumerate()
                .map(|(index, param)| (PartKey::Param(index), param))
                .chain([(PartKey::Result, result)])
                .collect(),
            Shape::Object(fields) => fields
                .iter()
                .map(|(field, ty)| (PartKey::Field(field.clone()), ty))
                .collect(),
            Shape::Struct(_, args) => args
                .iter()
                .enumerate()
                .map(|(index, arg)| (PartKey::Arg(index), arg))
                .collect(),
            Shape::Extern(_, args, brand) => args
                .iter()
                .enumerate()
                .map(|(index, arg)| (PartKey::Arg(index), arg))
                .chain([(PartKey::Brand, brand)])
                .collect(),
            Shape::FnValue(_)
            | Shape::Scalar(_)
            | Shape::String
            | Shape::Str
            | Shape::Unit
            | Shape::Never => Vec::new(),
        }
    }

    /// The part `key` names, where the shape has a place for it.
    pub fn place(&self, key: &PartKey) -> Option<&P> {
        self.places()
            .into_iter()
            .find(|(place, _)| place == key)
            .map(|(_, part)| part)
    }

    /// The same constructor and members, with `f` of each part in its place.
    pub fn map<Q, F>(&self, mut f: F) -> Shape<Q>
    where
        F: FnMut(&P) -> Q,
    {
        match self {
            Shape::Ref(mutability, target) => Shape::Ref(*mutability, f(target)),
            Shape::Tuple(components) => Shape::Tuple(components.iter().map(f).collect()),
            Shape::Array(len, elem) => Shape::Array(*len, f(elem)),
            Shape::Slice(elem) => Shape::Slice(f(elem)),
            Shape::Option(some) => Shape::Option(f(some)),
            Shape::Result(ok, err) => {
                let ok = f(ok);
                Shape::Result(ok, f(err))
            }
            Shape::Error(variants) => Shape::Error(
                variants
                    .iter()
                    .map(|(variant, payload)| (variant.clone(), f(payload)))
                    .collect(),
            ),
            Shape::ErrorOf(error) => Shape::ErrorOf(f(error)),
            Shape::Fn(params, result) => {
                let params = params.iter().map(&mut f).collect();
                Shape::Fn(params, f(result))
            }
            Shape::FnValue(closures) => Shape::FnValue(closures.clone()),
            Shape::Object(fields) => Shape::Object(
                fields
                    .iter()
                    .map(|(field, ty)| (field.clone(), f(ty)))
                    .collect(),
            ),
            Shape::Enum(name, variants) => Shape::Enum(
                name.clone(),
                variants
                    .iter()
                    .map(|(variant, payload)| (variant.clone(), f(payload)))
                    .collect(),
            ),
            Shape::Struct(module, args) => Shape::Struct(module.clone(), args.iter().map(f).collect()),
            Shape::Extern(name, args, brand) => {
                let args = args.iter().map(&mut f).collect();
                Shape::Extern(name.clone(), args, f(brand))
            }
            Shape::Scalar(scalar) => Shape::Scalar(*scalar),
            Shape::String => Shape::String,
            Shape::Str => Shape::Str,
            Shape::Unit => Shape::Unit,
            Shape::Never => Shape::Never,
        }
    }

    /// The constructor and members alone. Two shapes are of one layer where
    /// these are equal, whatever their parts.
    pub fn layer(&self) -> Shape<()> {
        self.map(|_| ())
    }
}

/// A finite term, as the text or a signature writes it.
///
/// It is made into classes where it is recorded (O13), and solving sees only
/// the classes after that: each node a decided class, and each hole one
/// undecided class, however many times the signature writes it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Ty {
    /// A hole of a signature, `T`.
    Hole(Name),
    Node(Box<Shape<Ty>>),
}

/// A constructor other than a reference: what remains of the notation
/// without its members and parts.
///
/// A reference is held by [`ClassNode::Ref`], together with its target's
/// constructor. No reference is the target of a reference (O23), so there is
/// no reference here.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Constructor {
    Tuple(usize),
    Array(usize),
    Slice,
    Option,
    Result,
    Error,
    ErrorOf,
    /// The number of parameters.
    Fn(usize),
    FnValue,
    Object,
    Enum(Name),
    Struct(Name),
    Extern(Name),
    Scalar(Scalar),
    String,
    Str,
    Unit,
    Never,
}

/// What a change goes by (Equivalence classes).
///
/// A reference's comes from its own class and its target's class, both
/// decided. Where either is not decided there is no class node: no class node
/// is partly decided.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ClassNode {
    Ref(Mutability, Constructor),
    Plain(Constructor),
}

/// The name of a part.
///
/// An undecided class's constraints name its parts before it has a shape, so
/// the names stand apart from the places of a [`Shape`]. A projection names a
/// part by one of these. A key that carries a field or a variant names that
/// member as well.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PartKey {
    /// Named by `*r`.
    Target,
    /// Named by `t.0`.
    Component(usize),
    /// Named by `a[i]`.
    Elem,
    /// Named by `p.f`.
    Field(Name),
    /// Of `Some`, `Ok`, `Err`, or a variant of an enum or an `Error`; named
    /// by `if let V(x) = o`.
    Payload(Name),
    Param(usize),
    Result,
    /// A type parameter of a struct, an argument of an extern-defined type,
    /// or the `E` of `Error⟨E⟩`.
    Arg(usize),
    Brand,
}
