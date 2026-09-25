//! What a function type says its call joins (RFC-0079 rule 5): for each
//! output of a call, the inputs whose loans it may take.
//!
//! A flow names the ends of a call, not positions: `Param(i)` is the
//! argument at `i` whatever its type, and a flow is `Aligned` (position `k`
//! of the output from position `k` of the input), `Labelled` (each position
//! of the output from the input positions whose labels reach its label,
//! RFC-0096) or `Any` (every position of the output from every position of
//! the input). A function type is met in the solver before its parameters'
//! types resolve, and the flows it carries then are already what the MIR
//! reads: the positions an end has are the resolved type's
//! (`analysis::loans::positions`), counted where a call is checked. An end
//! of a type variable `T` flowing `Aligned` to another end of `T` is rule
//! 5's label `(T, k)`: the input's `k`-th position reaches only the
//! output's `k`-th. A `Labelled` flow carries the shape of both ends as the
//! extern's Rust signature lays them out (`Laid`), which a call walks beside
//! the ends' resolved types to find each label's positions.

use serde::{Deserialize, Serialize};

use crate::ty::{Phase, TyTerm};

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
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Alignment {
    /// Position `k` of the output takes position `k` of the input; the two
    /// ends have one shape.
    Aligned,
    /// Each position of the output takes the input positions its map names
    /// (RFC-0096 rule 1).
    Labelled(Box<Labelled>),
    /// Every position of the output takes every position of the input.
    Any,
}

/// The shape of one end of a `Labelled` flow, as the extern's Rust
/// signature lays it out (RFC-0079 rule 2, RFC-0096 rule 2). Its
/// *segments* are, in order: one per `Ref` (the reference's own
/// position), one per region parameter of a `User`, and one per `Var` and
/// `Unread`, which stand for every position of the part of the end's type
/// they are matched with.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Laid {
    NoPosition,
    /// `&T`: one segment for the reference, then `T`'s.
    Ref(Box<Laid>),
    Option(Box<Laid>),
    Result(Box<Laid>, Box<Laid>),
    Tuple(Vec<Laid>),
    /// `[T; N]` or `[T]`: the element's.
    Array(Box<Laid>),
    /// An extension type: one segment per region parameter, then its type
    /// arguments', in declaration order.
    User { regions: usize, args: Vec<Laid> },
    /// A type variable: one segment, every position of its value.
    Var,
    /// A type the macro does not read, as a type argument of an extension
    /// type: one segment, every position of it.
    Unread,
}

impl Laid {
    pub fn segment_count(&self) -> usize {
        match self {
            Laid::NoPosition => 0,
            Laid::Var | Laid::Unread => 1,
            Laid::Ref(pointee) => 1 + pointee.segment_count(),
            Laid::Option(inner) | Laid::Array(inner) => inner.segment_count(),
            Laid::Result(ok, err) => ok.segment_count() + err.segment_count(),
            Laid::Tuple(parts) => parts.iter().map(Laid::segment_count).sum(),
            Laid::User { regions, args } => regions + args.iter().map(Laid::segment_count).sum::<usize>(),
        }
    }

    /// Whether this shape is `ty`'s, above the parts `Var` and `Unread`
    /// stand for: the declared form of what a call matches against the
    /// resolved type (`analysis::loans::segments`).
    pub fn fits<V: Phase>(&self, ty: &TyTerm<V>) -> bool {
        match (self, ty) {
            (Laid::Var | Laid::Unread, _) => true,
            (Laid::NoPosition, ty) => has_no_position(ty),
            (Laid::Ref(pointee), TyTerm::Ref(_, inner)) => pointee.fits(&inner.ty()),
            (Laid::Option(inner), TyTerm::Option(ty))
            | (Laid::Array(inner), TyTerm::Array(ty, _) | TyTerm::Slice(ty)) => inner.fits(ty),
            (Laid::Result(ok, err), TyTerm::Result(ok_ty, err_ty)) => ok.fits(ok_ty) && err.fits(err_ty),
            (Laid::Tuple(parts), TyTerm::Tuple(items)) => {
                parts.len() == items.len() && parts.iter().zip(items).all(|(part, item)| part.fits(item))
            }
            (
                Laid::User { regions, args },
                TyTerm::UserDefined {
                    type_args,
                    region_params,
                    ..
                },
            ) => {
                regions == region_params
                    && args.len() == type_args.len()
                    && args.iter().zip(type_args).all(|(arg, ty)| arg.fits(&ty.ty()))
            }
            _ => false,
        }
    }
}

/// Whether a value of `ty` has no position whatever its variables are.
fn has_no_position<V: Phase>(ty: &TyTerm<V>) -> bool {
    match ty {
        TyTerm::Ref(..) | TyTerm::Fn { .. } | TyTerm::Handle(_) | TyTerm::Var(_) => false,
        TyTerm::UserDefined {
            type_args,
            region_params,
            ..
        } => *region_params == 0 && type_args.iter().all(|arg| has_no_position(&arg.ty())),
        TyTerm::Array(inner, _) | TyTerm::Option(inner) | TyTerm::Slice(inner) => has_no_position(inner),
        TyTerm::Result(ok, err) => has_no_position(ok) && has_no_position(err),
        TyTerm::Tuple(items) => items.iter().all(has_no_position),
        TyTerm::Object(fields) => fields.values().all(has_no_position),
        TyTerm::Enum { variants, .. } => variants.values().flatten().all(|payload| has_no_position(payload)),
        TyTerm::Int(_)
        | TyTerm::Float
        | TyTerm::Char
        | TyTerm::String
        | TyTerm::Bool
        | TyTerm::Unit
        | TyTerm::Never
        | TyTerm::Order
        | TyTerm::Str
        | TyTerm::Error(_) => true,
    }
}

/// Where one segment of an output takes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Take {
    /// The input segment.
    pub segment: usize,
    /// Position `k` of the output segment from position `k` of the input
    /// one: both are one type variable's. Otherwise every position of the
    /// output segment takes every position of the input one.
    pub aligned: bool,
}

/// A `Labelled` flow: both ends' shapes and, for each segment of the
/// output, the input segments it takes from (RFC-0096 rule 1). A segment
/// no `Take` names takes nothing from this input.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Labelled {
    pub to: Laid,
    pub from: Laid,
    pub takes: Vec<Vec<Take>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
/// `Every` is every output from every input, `Any`: the union RFC-0064
/// gave a call without a summary, and what an extern's type states until
/// its lifetimes are read (RFC-0079 rule 6). It is its own value because a
/// declaration states it without counting the parameters; `Listed` holds
/// the flows sorted, each pair of ends once per alignment, an `Any` flow
/// standing for the `Aligned` or `Labelled` one it covers, so two equal
/// sets are equal values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// Both sets' flows: where two function types meet (RFC-0079 rule 5).
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
/// solver joins where two function types meet (RFC-0079 rule 5).
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
