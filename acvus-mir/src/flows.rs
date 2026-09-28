//! What a function type says its call joins (ownership §5, extern-fn §4):
//! for each output of a call, the inputs whose loans it may take.
//!
//! A flow names the ends of a call, not positions: `Param(i)` is the
//! argument at `i` whatever its type, and a flow is `Aligned` (position `k`
//! of the output from position `k` of the input), `Labelled` (each segment
//! of the output from the input segments its map names) or `Any` (every
//! position of the output from every position of the input). The positions
//! an end has are its concrete type's (`ty::positions`), counted where a
//! call is checked. A `Labelled` flow names the declaration its ends are
//! read off, whose segments (`decl::labels`) a call reads beside the
//! concrete types.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use acvus_utils::QualifiedRef;

use crate::decl::Declarations;
use crate::ty::{Concrete, LifetimesOf, ObjectTy, Poly, Ty, TyId, TyKind, Types, positions};

/// One end of a flow: a value the call reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlowEnd {
    /// The call's result. An output only.
    Result,
    /// The argument at this index. As an input, every position it holds,
    /// what its references point at read from the storage they name at the
    /// call. As an output, what its `&mut` positions and its captures name:
    /// the storage the callee may write into.
    Param(usize),
    /// The callee value's one position, what it captured. As an input,
    /// every loan it reaches through its captures; as an output, what it
    /// captured mutably. A named function captures nothing.
    Captures,
}

/// How the positions of a flow's two ends meet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Alignment {
    /// Position `k` of the output takes position `k` of the input; the two
    /// ends have one shape.
    Aligned,
    /// Each segment of the output takes the input segments its map names.
    Labelled(Box<Labelled>),
    /// Every position of the output takes every position of the input.
    Any,
}

/// One segment of a declared type, in the order `segments` reads them
/// (`decl::labels`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Segment {
    /// A reference's own position.
    Reference,
    /// One lifetime an extern-defined type declares.
    Lifetime,
    /// A function value's one position (ownership §5).
    FunctionValue,
    /// Every position of what instantiates this type variable.
    TypeVar(u32),
}

impl Segment {
    /// Whether position `k` of `self` takes position `k` of `from` alone:
    /// both are one type variable's part (`decl::labels`). Otherwise every
    /// position of `self` takes every position of `from`.
    pub fn aligned_with(self, from: Segment) -> bool {
        matches!((self, from), (Segment::TypeVar(to), Segment::TypeVar(from)) if to == from)
    }
}

/// A `Labelled` flow: the declaration its ends are read off, and for each
/// segment of the output the input segments it takes. A segment the map
/// does not name takes nothing from this input.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Labelled {
    /// The declaration whose signature's ends the segments are read off.
    pub declared: Declared,
    pub map: Box<[Box<[usize]>]>,
}

/// An extern fn's declared signature: the scheme's own, or one concrete
/// instance's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Declared {
    pub function: QualifiedRef,
    pub instance: Option<usize>,
}

impl Declared {
    /// The declared type of the end `end` of this signature, in the
    /// declarations' table: the result, or the parameter at its index.
    ///
    /// # Panics
    /// The declaration, its instance or the end does not exist, or `end`
    /// is the callee's captures: a `Labelled` flow is read off an extern's
    /// signature, whose ends are its parameters and its result.
    pub fn end_ty(self, decls: &Declarations, end: FlowEnd) -> TyId<Poly> {
        let scheme = decls
            .scheme(self.function)
            .expect("a Labelled flow names a declared function");
        let signature = match self.instance {
            None => scheme.ty,
            Some(instance) => {
                scheme
                    .instances
                    .as_ref()
                    .and_then(|instances| instances.concrete.get(instance))
                    .expect("a Labelled flow names a concrete instance its declaration has")
                    .ty
            }
        };
        let TyKind::Fn { params, ret, .. } = decls.types().kind(signature) else {
            panic!("a declared function's signature is a function type")
        };
        match end {
            FlowEnd::Result => *ret,
            FlowEnd::Param(index) => {
                params
                    .get(index)
                    .expect("a Labelled flow's end is a parameter of its signature")
                    .ty
            }
            FlowEnd::Captures => panic!("a Labelled flow's ends are an extern's parameters and result"),
        }
    }
}

/// The segments of the declared type `ty`, in order: a reference's own
/// position before its target's, an extern-defined type's lifetimes before
/// its arguments', a structural type's parts in order, and one element's
/// for a sequence. A function value's one position is one segment.
///
/// # Panics
/// `ty` holds a `Handle` or an `Order`, which no declaration writes.
pub fn segments(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>) -> Vec<Segment> {
    let mut out = Vec::new();
    push_segments(poly, decls, ty, &mut out);
    out
}

fn push_segments(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>, out: &mut Vec<Segment>) {
    acvus_utils::grow(|| push_segments_level(poly, decls, ty, out))
}

fn push_segments_level(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>, out: &mut Vec<Segment>) {
    match poly.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Str => {}
        TyKind::Fn { .. } => out.push(Segment::FunctionValue),
        TyKind::Order | TyKind::Handle(_) => panic!("a declared type holds no Handle and no Order"),
        TyKind::Var(var) => out.push(Segment::TypeVar(*var)),
        TyKind::Ref(_, target) => {
            out.push(Segment::Reference);
            push_segments(poly, decls, target.ty(), out);
        }
        TyKind::UserDefined { id, type_args, .. } => {
            out.extend(std::iter::repeat_n(Segment::Lifetime, decls.lifetimes_of(*id)));
            for arg in type_args {
                push_segments(poly, decls, arg.ty(), out);
            }
        }
        TyKind::Array(element, _) | TyKind::Slice(element) | TyKind::Option(element) => {
            push_segments(poly, decls, *element, out)
        }
        TyKind::Result(ok, err) => {
            push_segments(poly, decls, *ok, out);
            push_segments(poly, decls, *err, out);
        }
        TyKind::Tuple(items) => {
            for item in items {
                push_segments(poly, decls, *item, out);
            }
        }
        TyKind::Object(object) => {
            for (_, field) in object.fields() {
                push_segments(poly, decls, *field, out);
            }
        }
        TyKind::Enum { variants, .. } => {
            for payload in variants.iter().filter_map(|(_, payload)| *payload) {
                push_segments(poly, decls, payload, out);
            }
        }
    }
}

/// The positions of a value of `ty` each segment of `decl_ty` covers, in
/// the order `segments` gives them. `ty` is an instance of `decl_ty`: a
/// type variable's part covers every position of what instantiates it.
///
/// # Panics
/// `ty` is not an instance of `decl_ty`. A call's concrete type is its
/// declaration's instantiated, so the two agree by construction.
pub fn segment_positions(
    poly: &Types<Poly>,
    decl_ty: TyId<Poly>,
    types: &Types<Concrete>,
    ty: Ty,
    decls: &Declarations,
) -> Vec<Range<usize>> {
    let mut walk = SegmentWalk {
        poly,
        types,
        decls,
        at: 0,
        out: Vec::new(),
    };
    walk.lay(decl_ty, ty);
    assert_eq!(
        walk.at,
        positions(types, decls, ty),
        "a declared type's segments lie within its instance's positions"
    );
    walk.out
}

struct SegmentWalk<'a> {
    poly: &'a Types<Poly>,
    types: &'a Types<Concrete>,
    decls: &'a Declarations,
    at: usize,
    out: Vec<Range<usize>>,
}

impl SegmentWalk<'_> {
    fn lay(&mut self, decl_ty: TyId<Poly>, ty: Ty) {
        acvus_utils::grow(|| self.lay_level(decl_ty, ty))
    }

    fn cover(&mut self, width: usize) {
        self.out.push(self.at..self.at + width);
        self.at += width;
    }

    fn lay_level(&mut self, decl_ty: TyId<Poly>, ty: Ty) {
        const DISAGREE: &str = "a call's concrete type is an instance of its declared type";
        match (self.poly.kind(decl_ty), self.types.kind(ty)) {
            (TyKind::Var(_), _) => self.cover(positions(self.types, self.decls, ty)),
            (TyKind::Order | TyKind::Handle(_), _) => {
                panic!("a declared type holds no Handle and no Order")
            }
            (
                TyKind::Int(_)
                | TyKind::Float
                | TyKind::Char
                | TyKind::String
                | TyKind::Bool
                | TyKind::Unit
                | TyKind::Never
                | TyKind::Str,
                _,
            ) => {}
            (TyKind::Fn { .. }, TyKind::Fn { .. }) => self.cover(1),
            (TyKind::Ref(_, decl_target), TyKind::Ref(_, target)) => {
                self.cover(1);
                self.lay(decl_target.ty(), target.ty());
            }
            (
                TyKind::UserDefined {
                    id: decl_id,
                    type_args: decl_args,
                    ..
                },
                TyKind::UserDefined { id, type_args, .. },
            ) => {
                assert_eq!(decl_id, id, "{DISAGREE}");
                assert_eq!(decl_args.len(), type_args.len(), "{DISAGREE}");
                for _ in 0..self.decls.lifetimes_of(*id) {
                    self.cover(1);
                }
                for (decl_arg, arg) in decl_args.iter().zip(type_args.iter()) {
                    self.lay(decl_arg.ty(), arg.ty());
                }
            }
            (
                TyKind::Array(decl_element, _) | TyKind::Slice(decl_element),
                TyKind::Array(element, _) | TyKind::Slice(element),
            )
            | (TyKind::Option(decl_element), TyKind::Option(element)) => self.lay(*decl_element, *element),
            (TyKind::Result(decl_ok, decl_err), TyKind::Result(ok, err)) => {
                self.lay(*decl_ok, *ok);
                self.lay(*decl_err, *err);
            }
            (TyKind::Tuple(decl_items), TyKind::Tuple(items)) => {
                assert_eq!(decl_items.len(), items.len(), "{DISAGREE}");
                for (decl_item, item) in decl_items.iter().zip(items.iter()) {
                    self.lay(*decl_item, *item);
                }
            }
            (TyKind::Object(decl_object), TyKind::Object(object)) => {
                assert!(
                    matches!(
                        (decl_object, object),
                        (ObjectTy::Structural(..), ObjectTy::Structural(..))
                    ) || matches!(
                        (decl_object, object),
                        (ObjectTy::Declared(decl_id, _), ObjectTy::Declared(id, _)) if decl_id == id
                    ),
                    "{DISAGREE}"
                );
                let (decl_fields, fields) = (decl_object.fields(), object.fields());
                assert_eq!(decl_fields.len(), fields.len(), "{DISAGREE}");
                for ((decl_name, decl_field), (name, field)) in decl_fields.iter().zip(fields.iter()) {
                    assert_eq!(decl_name, name, "{DISAGREE}");
                    self.lay(*decl_field, *field);
                }
            }
            (
                TyKind::Enum {
                    name: decl_name,
                    variants: decl_variants,
                    ..
                },
                TyKind::Enum { name, variants, .. },
            ) => {
                assert_eq!(decl_name, name, "{DISAGREE}");
                assert_eq!(decl_variants.len(), variants.len(), "{DISAGREE}");
                for ((decl_tag, decl_payload), (tag, payload)) in decl_variants.iter().zip(variants.iter()) {
                    assert_eq!(decl_tag, tag, "{DISAGREE}");
                    match (decl_payload, payload) {
                        (Some(decl_payload), Some(payload)) => self.lay(*decl_payload, *payload),
                        (None, None) => {}
                        _ => panic!("{DISAGREE}"),
                    }
                }
            }
            _ => panic!("{DISAGREE}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Flow {
    pub to: FlowEnd,
    pub from: FlowEnd,
    pub alignment: Alignment,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Source {
    pub from: FlowEnd,
    pub alignment: Alignment,
}

/// The flows of one function type.
///
/// `Every` is every output from every input, `Any`: what a declaration that
/// states nothing about its flows gets (extern-fn §4). It is its own value
/// because a declaration states it without counting the parameters.
/// `Listed` holds the flows sorted, each pair of ends once per alignment,
/// an `Any` flow standing for the `Aligned` or `Labelled` one it covers, so
/// two equal sets are equal values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Flows {
    Every,
    Listed(Vec<Flow>),
}

impl Default for Flows {
    /// What a declaration that states nothing about its flows gets.
    fn default() -> Self {
        Flows::Every
    }
}

impl Flows {
    pub fn none() -> Self {
        Flows::Listed(Vec::new())
    }

    pub fn of(flows: impl IntoIterator<Item = Flow>) -> Self {
        let mut listed: Vec<Flow> = flows.into_iter().collect();
        listed.sort();
        listed.dedup();
        let covered: Vec<Flow> = listed
            .iter()
            .filter(|flow| flow.alignment != Alignment::Any)
            .filter(|narrow| {
                listed.contains(&Flow {
                    to: narrow.to,
                    from: narrow.from,
                    alignment: Alignment::Any,
                })
            })
            .cloned()
            .collect();
        listed.retain(|flow| !covered.contains(flow));
        Flows::Listed(listed)
    }

    /// Both sets' flows: where two function types join (types §2).
    pub fn join(&self, other: &Flows) -> Flows {
        match (self, other) {
            (Flows::Every, _) | (_, Flows::Every) => Flows::Every,
            (Flows::Listed(a), Flows::Listed(b)) => Flows::of(a.iter().chain(b).cloned()),
        }
    }

    /// Whether every flow of `other` is one of these.
    pub fn covers(&self, other: &Flows) -> bool {
        match (self, other) {
            (Flows::Every, _) => true,
            (Flows::Listed(_), Flows::Every) => false,
            (Flows::Listed(_), Flows::Listed(b)) => b
                .iter()
                .all(|flow| self.admits(flow.to, flow.from, &flow.alignment)),
        }
    }

    /// Whether `to` may take from `from` at `alignment`: an `Any` flow
    /// admits an `Aligned` or a `Labelled` one between the same ends, and
    /// a `Labelled` one admits only itself, since which positions it maps
    /// depends on the ends' types.
    pub fn admits(&self, to: FlowEnd, from: FlowEnd, alignment: &Alignment) -> bool {
        match self {
            Flows::Every => true,
            Flows::Listed(listed) => listed.iter().any(|flow| {
                flow.to == to
                    && flow.from == from
                    && (flow.alignment == Alignment::Any || flow.alignment == *alignment)
            }),
        }
    }

    /// The inputs `to` takes from at a call with `arity` arguments.
    pub fn into_end(&self, to: FlowEnd, arity: usize) -> Vec<Source> {
        match self {
            Flows::Every => (0..arity)
                .map(FlowEnd::Param)
                .chain([FlowEnd::Captures])
                .map(|from| Source {
                    from,
                    alignment: Alignment::Any,
                })
                .collect(),
            Flows::Listed(listed) => listed
                .iter()
                .filter(|flow| flow.to == to)
                .map(|flow| Source {
                    from: flow.from,
                    alignment: flow.alignment.clone(),
                })
                .collect(),
        }
    }
}

/// A function type's flows during inference: known, or a variable the
/// solver joins where two function types join (types §2).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FlowTerm<V: crate::ty::Phase> {
    Known(Flows),
    Var(V::FlowVar),
}

impl<V: crate::ty::Phase> From<Flows> for FlowTerm<V> {
    fn from(flows: Flows) -> Self {
        FlowTerm::Known(flows)
    }
}

impl<V: crate::ty::Phase> FlowTerm<V> {
    pub fn map<W: crate::ty::Phase>(
        &self,
        on_flow: &mut impl FnMut(V::FlowVar) -> FlowTerm<W>,
    ) -> FlowTerm<W> {
        match self {
            FlowTerm::Known(flows) => FlowTerm::Known(flows.clone()),
            FlowTerm::Var(v) => on_flow(*v),
        }
    }

    pub fn try_map<W: crate::ty::Phase, E>(
        &self,
        on_flow: &mut impl FnMut(V::FlowVar) -> Result<FlowTerm<W>, E>,
    ) -> Result<FlowTerm<W>, E> {
        match self {
            FlowTerm::Known(flows) => Ok(FlowTerm::Known(flows.clone())),
            FlowTerm::Var(v) => on_flow(*v),
        }
    }
}

impl FlowTerm<crate::ty::Concrete> {
    pub fn get(&self) -> &Flows {
        match self {
            FlowTerm::Known(flows) => flows,
            FlowTerm::Var(v) => match *v {},
        }
    }
}

impl FlowTerm<crate::ty::Poly> {
    pub fn get(&self) -> &Flows {
        match self {
            FlowTerm::Known(flows) => flows,
            FlowTerm::Var(v) => match *v {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow(to: FlowEnd, from: FlowEnd, alignment: Alignment) -> Flow {
        Flow {
            to,
            from,
            alignment,
        }
    }

    #[test]
    fn an_any_flow_stands_for_the_aligned_one_between_the_same_ends() {
        let joined = Flows::of([
            flow(FlowEnd::Result, FlowEnd::Param(0), Alignment::Aligned),
            flow(FlowEnd::Result, FlowEnd::Param(0), Alignment::Any),
        ]);
        assert_eq!(
            joined,
            Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), Alignment::Any)])
        );
    }

    #[test]
    fn a_join_covers_both_sides_and_every_covers_all() {
        let a = Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), Alignment::Aligned)]);
        let b = Flows::of([flow(FlowEnd::Result, FlowEnd::Param(1), Alignment::Any)]);
        let joined = a.join(&b);
        assert!(joined.covers(&a) && joined.covers(&b));
        assert!(!a.covers(&b));
        assert!(Flows::Every.covers(&joined));
        assert!(!joined.covers(&Flows::Every));
    }
}
