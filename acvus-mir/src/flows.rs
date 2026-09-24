//! What a function type says its call joins (RFC-0079 rule 5): for each
//! output of a call, the inputs whose loans it may take.
//!
//! A flow names the ends of a call, not positions: `Param(i)` is the
//! argument at `i` whatever its type, and a flow is `Aligned` (position `k`
//! of the output from position `k` of the input) or `Any` (every position of
//! the output from every position of the input). A function type is met in
//! the solver before its parameters' types resolve, and the flows it carries
//! then are already what the MIR reads: the positions an end has are the
//! resolved type's (`analysis::loans::positions`), counted where a call is
//! checked. An end of a type variable `T` flowing `Aligned` to another end of
//! `T` is rule 5's label `(T, k)`: the input's `k`-th position reaches only
//! the output's `k`-th.

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Alignment {
    /// Position `k` of the output takes position `k` of the input; the two
    /// ends have one shape.
    Aligned,
    /// Every position of the output takes every position of the input.
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Flow {
    pub to: FlowEnd,
    pub from: FlowEnd,
    pub alignment: Alignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
/// the flows sorted, each pair of ends once, an `Any` flow standing for the
/// `Aligned` one it covers, so two equal sets are equal values.
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
            .filter(|flow| flow.alignment == Alignment::Aligned)
            .filter(|aligned| {
                listed.contains(&Flow {
                    alignment: Alignment::Any,
                    ..**aligned
                })
            })
            .copied()
            .collect();
        listed.retain(|flow| !covered.contains(flow));
        Flows::Listed(listed)
    }

    /// Both sets' flows: where two function types meet (RFC-0079 rule 5).
    pub fn join(&self, other: &Flows) -> Flows {
        match (self, other) {
            (Flows::Every, _) | (_, Flows::Every) => Flows::Every,
            (Flows::Listed(a), Flows::Listed(b)) => Flows::of(a.iter().chain(b).copied()),
        }
    }

    /// Whether every flow of `other` is one of these.
    pub fn covers(&self, other: &Flows) -> bool {
        match (self, other) {
            (Flows::Every, _) => true,
            (Flows::Listed(_), Flows::Every) => false,
            (Flows::Listed(_), Flows::Listed(b)) => b
                .iter()
                .all(|flow| self.admits(flow.to, flow.from, flow.alignment)),
        }
    }

    /// Whether `to` may take from `from` at `alignment`: an `Any` flow
    /// admits an `Aligned` one between the same ends.
    pub fn admits(&self, to: FlowEnd, from: FlowEnd, alignment: Alignment) -> bool {
        match self {
            Flows::Every => true,
            Flows::Listed(listed) => listed.iter().any(|flow| {
                flow.to == to
                    && flow.from == from
                    && (flow.alignment == Alignment::Any || flow.alignment == alignment)
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
                    alignment: flow.alignment,
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
