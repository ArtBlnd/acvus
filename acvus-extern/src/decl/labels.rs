//! A declaration's lifetimes (ownership §5, extern-fn §4), labelled from the
//! names its Rust signature writes. The attribute reads, for each end, the
//! lifetime its outermost `&` is written at, if it is one, and the names
//! written anywhere else in it; this is all it reads of the tokens. A type's
//! segments are the compiler's own (`segments`, over the lowered type), so no
//! order is read off a type's spelling.
//!
//! Every name is one label across the signature; the names inside one end
//! (all but its outermost `&`'s) are one label. A label so taken for several
//! lifetimes is taken for the loans of each, a flow wider than Rust's, never
//! narrower: Rust checks the body against the same names, so what the body
//! returns or stores under a name is what the label says. The outermost `&`
//! is the type's first segment wherever it is written, so it keeps its own
//! label, and what lies inside it outlives it.

use std::collections::HashMap;

use acvus_mir::decl::{EndLabels, LabelledSignature, Lifetime, Outlives, Segment, SegmentLabel, segments};
use acvus_mir::ty::FlowEnd;

use crate::decl::DeclTy;
use crate::decl::poly::Lowered;

/// An end of a call: one of its arguments, by position among them, or its
/// result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum At {
    Param(usize),
    Result,
}

/// An end as the attribute read it: its declared type, the name its outermost
/// `&` is written at where its type is one, and every other name written in
/// it. An elided lifetime is named by the attribute: freshly in an argument,
/// as Rust elides it in the result.
pub struct EndNames {
    pub at: At,
    pub ty: DeclTy,
    pub outer: Option<&'static str>,
    pub inner: &'static [&'static str],
}

/// The labels of each end, and the order of the lifetimes they name: the
/// inside of each outermost `&` outliving it, which its type implies. A
/// signature writes no bound between lifetimes.
pub fn labelled(ends: &[EndNames]) -> LabelledSignature {
    let mut classes = Classes::default();
    for end in ends {
        classes.union_all(end.inner);
    }

    let mut lowered = Lowered::new();
    let mut used = Vec::new();
    let mut implied = Vec::new();
    let mut labelled_ends = Vec::with_capacity(ends.len());
    for end in ends {
        let ty = lowered.lower(&end.ty);
        let inner = end.inner.first().map(|name| classes.find(name));
        let mut inside = None;
        let mut labels = Vec::new();
        for (at, segment) in segments(&lowered.types, &lowered, ty).into_iter().enumerate() {
            // The outermost `&` is the first segment, where the end is one.
            let outermost = if at == 0 { end.outer } else { None };
            let class = match (segment, outermost) {
                (Segment::TypeVar(var), _) => {
                    labels.push(SegmentLabel::TypeVar(var));
                    continue;
                }
                (Segment::Reference, Some(outer)) => classes.find(outer),
                (Segment::Reference | Segment::Lifetime | Segment::FunctionValue, _) => {
                    // A segment no name is written at is elided in a path:
                    // the end's own, inside.
                    *inside.get_or_insert_with(|| inner.unwrap_or_else(|| classes.fresh()))
                }
            };
            used.push(class);
            labels.push(SegmentLabel::Lifetime(class));
        }
        if let (Some(outer), Some(inside)) = (end.outer, inside) {
            implied.push(Outlives { longer: inside, shorter: classes.find(outer) });
        }
        labelled_ends.push(EndLabels { end: flow_end(end.at), labels: labels.into() });
    }

    let mut orders: Vec<Outlives> = Vec::new();
    // An order between one lifetime and itself says nothing, and the
    // compiler takes an order only between lifetimes the labels name.
    for order in implied {
        let named = used.contains(&order.longer) && used.contains(&order.shorter);
        if order.longer != order.shorter && named && !orders.contains(&order) {
            orders.push(order);
        }
    }
    LabelledSignature { ends: labelled_ends.into(), outlives: orders.into() }
}

fn flow_end(at: At) -> FlowEnd {
    match at {
        At::Param(param) => FlowEnd::Param(param),
        At::Result => FlowEnd::Result,
    }
}

/// The names, each in one class, and the classes, each one lifetime.
#[derive(Default)]
struct Classes {
    of: HashMap<&'static str, u32>,
    parent: Vec<u32>,
}

impl Classes {
    fn fresh(&mut self) -> Lifetime {
        let class = self.parent.len() as u32;
        self.parent.push(class);
        Lifetime(class)
    }

    fn class(&mut self, name: &'static str) -> u32 {
        if let Some(&class) = self.of.get(name) {
            return class;
        }
        let Lifetime(class) = self.fresh();
        self.of.insert(name, class);
        class
    }

    fn root(&self, mut class: u32) -> u32 {
        while self.parent[class as usize] != class {
            class = self.parent[class as usize];
        }
        class
    }

    fn find(&mut self, name: &'static str) -> Lifetime {
        let class = self.class(name);
        Lifetime(self.root(class))
    }

    fn union_all(&mut self, names: &[&'static str]) {
        let Some((first, rest)) = names.split_first() else {
            return;
        };
        let Lifetime(root) = self.find(first);
        for name in rest {
            let Lifetime(other) = self.find(name);
            self.parent[other as usize] = root;
        }
    }
}

#[cfg(test)]
mod tests;
