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

impl Mutability {
    /// Exclusive where both are, and shared otherwise: what two references
    /// join as (Joins), and what a lend through a reference gives (O23).
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Mutability::Exclusive, Mutability::Exclusive) => Mutability::Exclusive,
            _ => Mutability::Shared,
        }
    }
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

impl<P> Shape<P>
where
    P: Clone,
{
    /// The head two heads join as, or none where they are of two head classes
    /// (Joins). `!` is the least head.
    ///
    /// A place both heads have keeps this head's part: that the two parts are
    /// one type is the judgment of each bound that enters the class, not of
    /// the join.
    pub fn join(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Shape::Never, head) | (head, Shape::Never) => Some(head.clone()),
            (Shape::Ref(mine, target), Shape::Ref(theirs, _)) => Some(Shape::Ref(mine.join(*theirs), target.clone())),
            (Shape::Object(mine), Shape::Object(theirs)) => Some(Shape::Object(united(mine, theirs))),
            (Shape::Error(mine), Shape::Error(theirs)) => Some(Shape::Error(united(mine, theirs))),
            (Shape::Enum(name, mine), Shape::Enum(other_name, theirs)) if name == other_name => {
                Some(Shape::Enum(name.clone(), united(mine, theirs)))
            }
            (Shape::FnValue(mine), Shape::FnValue(theirs)) => Some(Shape::FnValue(mine.union(theirs).copied().collect())),
            (mine, theirs) if mine.layer() == theirs.layer() => Some(mine.clone()),
            _ => None,
        }
    }

    /// Whether this head reaches `head` by entering it: their join is `head`.
    pub fn enters(&self, head: &Self) -> bool {
        self.join(head).is_some_and(|joined| joined.layer() == head.layer())
    }
}

fn united<P>(mine: &BTreeMap<Name, P>, theirs: &BTreeMap<Name, P>) -> BTreeMap<Name, P>
where
    P: Clone,
{
    let mut united = mine.clone();
    for (name, part) in theirs {
        united.entry(name.clone()).or_insert_with(|| part.clone());
    }
    united
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::id::Arena;

    const SHARED: Mutability = Mutability::Shared;
    const EXCLUSIVE: Mutability = Mutability::Exclusive;

    fn object(fields: &[&str]) -> Shape<u32> {
        Shape::Object(fields.iter().map(|field| (Name::new(*field), 0)).collect())
    }

    #[test]
    fn a_mutability_is_exclusive_only_where_both_are() {
        assert_eq!(EXCLUSIVE.join(EXCLUSIVE), EXCLUSIVE);
        assert_eq!(EXCLUSIVE.join(SHARED), SHARED);
        assert_eq!(SHARED.join(EXCLUSIVE), SHARED);
        assert_eq!(SHARED.join(SHARED), SHARED);
    }

    #[test]
    fn never_is_the_least_head() {
        let u8_head = Shape::<u32>::Scalar(Scalar::U8);
        assert_eq!(Shape::Never.join(&u8_head), Some(u8_head.clone()));
        assert_eq!(u8_head.join(&Shape::Never), Some(u8_head.clone()));
        assert!(Shape::Never.enters(&u8_head));
        assert!(!u8_head.enters(&Shape::Never));
    }

    #[test]
    fn references_join_as_shared_where_either_is_and_keep_the_first_target() {
        let joined = Shape::Ref(EXCLUSIVE, 1).join(&Shape::Ref(SHARED, 2));
        assert_eq!(joined, Some(Shape::Ref(SHARED, 1)));
        assert!(Shape::Ref(EXCLUSIVE, 1).enters(&Shape::Ref(SHARED, 2)));
        assert!(!Shape::Ref(SHARED, 1).enters(&Shape::Ref(EXCLUSIVE, 2)));
    }

    #[test]
    fn members_are_united_and_enter_only_a_head_that_has_them_all() {
        let (x, x_and_y) = (object(&["x"]), object(&["x", "y"]));
        assert_eq!(x.join(&x_and_y), Some(x_and_y.clone()));
        assert!(x.enters(&x_and_y));
        assert!(!x_and_y.enters(&x));

        let mut closures = Arena::<ClosureId, ()>::default();
        let (first, second) = (closures.push(()), closures.push(()));
        let values = |ids: &[ClosureId]| Shape::<u32>::FnValue(ids.iter().copied().collect());
        assert_eq!(values(&[first]).join(&values(&[second])), Some(values(&[first, second])));
    }

    #[test]
    fn enums_of_two_names_are_two_head_classes() {
        let of = |name: &str| Shape::<u32>::Enum(Name::new(name), BTreeMap::new());
        assert_eq!(of("E").join(&of("F")), None);
        assert_eq!(of("E").join(&of("E")), Some(of("E")));
    }

    #[test]
    fn heads_of_two_head_classes_do_not_join() {
        let scalar = |scalar| Shape::<u32>::Scalar(scalar);
        assert_eq!(scalar(Scalar::U8).join(&scalar(Scalar::U16)), None);
        assert_eq!(Shape::Tuple(vec![1]).join(&Shape::Tuple(vec![1, 2])), None);
        assert_eq!(Shape::Array(1, 1).join(&Shape::Array(2, 1)), None);
        assert_eq!(Shape::Slice(1).join(&Shape::Option(1)), None);
        assert_eq!(Shape::Tuple(vec![1, 2]).join(&Shape::Tuple(vec![3, 4])), Some(Shape::Tuple(vec![1, 2])));
    }
}
