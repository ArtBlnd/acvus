//! Changes: what a boundary's class reaches its upper bound by (O11).
//!
//! A change goes by what it sees of each end: a class node and members. Of a
//! reference it sees the mutability alone where the target shows no class node:
//! the argument's target is not decided yet, or the type given writes a hole there,
//! which takes whatever target comes. Ways to go are taken in this order, each only
//! where the one before it does not go:
//!
//! | Order | What goes | Chosen by |
//! |---|---|---|
//! | 1 | one stage of coercion | what each end shows |
//! | 2 | a coercion, then a chain of views | class nodes |
//! | 3 | a coercion, then a chain with a value conversion | whole types, once the argument's is decided at every depth |
//!
//! A view is a conversion between references whose parameter writes nothing its
//! class node does not show, as `&Vec<T> -> &[T]`. Any value with that class node
//! fits it, so choosing it early is never contradicted, and what a part takes
//! through it comes from the text. Any other conversion runs a function the text
//! does not call, so it is chosen only once the argument's type is known whole, and
//! it decides none of the argument's types. Within one order, exactly one way must
//! go.
//!
//! No coercion follows a conversion. A change alters nothing below what it sees:
//! each part there is one type with the part it corresponds to (O9), through each
//! conversion's signature, made anew at the boundary (O12).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::Adjustment;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{ClassKind, Template};
use crate::primitive::decl::Conversion;
use crate::primitive::record::Coercion;
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{ClassNode, Constructor, Mutability, Shape, TargetNode, Ty};
use crate::primitive::{ItemId, Name, TemplateId};
use crate::store::classes::Classes;
use crate::store::template::Fate;

use super::Outcome;
use super::bound::{decided_by, shape_of};

/// A class node, and the members of the head.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Point {
    node: ClassNode,
    layer: Shape<()>,
}

impl Point {
    /// The point of `head`, `target` being its target's head where it is a
    /// reference. No class node is partly decided.
    fn new<P, Q>(head: &Shape<P>, target: Option<&Shape<Q>>) -> Option<Self> {
        let node = match head {
            Shape::Ref(mutability, _) => {
                let target = match target? {
                    Shape::Ref(of_target, _) => TargetNode::Ref(*of_target),
                    target => TargetNode::Plain(target.constructor()?),
                };
                ClassNode::Ref(*mutability, target)
            }
            head => ClassNode::Plain(head.constructor()?),
        };
        Some(Self {
            node,
            layer: head.layer(),
        })
    }
}

/// What a change sees of one of its ends.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Seen {
    At(Point),
    /// A reference whose target shows no class node.
    Reference(Mutability),
}

fn point_of_shape<K>(classes: &Classes<K>, shape: &Shape<K::Id>) -> Option<Point>
where
    K: ClassKind,
{
    let target = match shape {
        Shape::Ref(_, target) => Some(shape_of(classes, *target)?),
        _ => None,
    };
    Point::new(shape, target)
}

/// What a change sees of `shape` as its lower end: a reference whose target is not
/// decided yet shows its mutability alone.
fn seen_from<K>(classes: &Classes<K>, shape: &Shape<K::Id>) -> Option<Seen>
where
    K: ClassKind,
{
    match shape {
        Shape::Ref(mutability, target) if shape_of(classes, *target).is_none() => Some(Seen::Reference(*mutability)),
        shape => point_of_shape(classes, shape).map(Seen::At),
    }
}

/// What a change sees of `shape` as its upper end, the type given: a reference
/// whose target is a hole shows its mutability alone, whether the hole is decided
/// yet or not, as a hole takes whatever comes.
fn seen_to<K>(classes: &Classes<K>, shape: &Shape<K::Id>) -> Option<Seen>
where
    K: ClassKind,
{
    match shape {
        Shape::Ref(mutability, target) if !classes.is_external(*target) => Some(Seen::Reference(*mutability)),
        shape => point_of_shape(classes, shape).map(Seen::At),
    }
}

/// `shape` with the reference just below its own read through: `&ₘ &U` as `&U`.
/// The inner reference is copied out, so it must be a shared one, and the outer
/// one may be either. There is none where `shape` is no reference to a reference
/// decided as shared.
fn read_through_once<K>(classes: &Classes<K>, shape: &Shape<K::Id>) -> Option<Shape<K::Id>>
where
    K: ClassKind,
{
    let Shape::Ref(_, target) = shape else {
        return None;
    };
    match shape_of(classes, *target)? {
        Shape::Ref(Mutability::Shared, inner) => Some(Shape::Ref(Mutability::Shared, *inner)),
        _ => None,
    }
}

/// What one stage of coercion comes to.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Stage {
    /// It goes by these coercions, none being the empty change.
    Goes(BTreeSet<Coercion>),
    /// A function value type going to a function type, whose instances are not
    /// recorded yet (O30).
    ToFnType,
    /// Only the target's class node, not decided yet, can tell.
    Waits,
    NoChange,
}

fn reborrow(from: Mutability, into: Mutability) -> Stage {
    match (from, into) {
        (Mutability::Shared, Mutability::Exclusive) => Stage::NoChange,
        (Mutability::Exclusive, Mutability::Shared) => Stage::Goes(BTreeSet::from([Coercion::Reborrow])),
        _ => Stage::Goes(BTreeSet::new()),
    }
}

/// The one stage of coercion from what is seen of one end to what is seen of the
/// other (Changes).
fn coercion(from: &Seen, to: &Seen) -> Stage {
    match (from, to) {
        (Seen::At(from), Seen::At(to)) => stage(from, to),
        (
            Seen::At(Point {
                node: ClassNode::Plain(Constructor::Never),
                ..
            }),
            Seen::Reference(_),
        ) => Stage::Goes(BTreeSet::from([Coercion::Never])),
        (
            Seen::At(Point {
                node: ClassNode::Ref(from, _),
                ..
            })
            | Seen::Reference(from),
            Seen::Reference(into),
        ) => reborrow(*from, *into),
        (Seen::At(_), Seen::Reference(_)) => Stage::NoChange,
        (
            Seen::Reference(from),
            Seen::At(Point {
                node: ClassNode::Ref(into, _),
                ..
            }),
        ) => match reborrow(*from, *into) {
            Stage::Goes(_) => Stage::Waits,
            refused => refused,
        },
        (Seen::Reference(_), Seen::At(_)) => Stage::NoChange,
    }
}

/// The one stage of coercion between two points.
fn stage(from: &Point, to: &Point) -> Stage {
    let goes = |coercions: &[Coercion]| Stage::Goes(coercions.iter().copied().collect());
    match (&from.node, &to.node) {
        (ClassNode::Plain(Constructor::Never), ClassNode::Plain(Constructor::Never)) => goes(&[]),
        (ClassNode::Plain(Constructor::Never), _) => goes(&[Coercion::Never]),
        (ClassNode::Ref(from_mutability, from_target), ClassNode::Ref(to_mutability, to_target)) => {
            let Stage::Goes(mut coercions) = reborrow(*from_mutability, *to_mutability) else {
                return Stage::NoChange;
            };
            match (from_target, to_target) {
                (TargetNode::Plain(Constructor::Array(_)), TargetNode::Plain(Constructor::Slice)) => {
                    coercions.insert(Coercion::Unsize);
                }
                (same, other) if same == other => {}
                _ => return Stage::NoChange,
            }
            Stage::Goes(coercions)
        }
        (ClassNode::Plain(Constructor::FnValue), ClassNode::Plain(Constructor::Fn(_))) => Stage::ToFnType,
        (ClassNode::Plain(same), ClassNode::Plain(other)) if same == other => {
            let same_members = from.layer == to.layer;
            match same {
                Constructor::Object if in_places_of(&to.layer, &from.layer).is_none() => Stage::NoChange,
                Constructor::Object if !same_members => goes(&[Coercion::DropFields]),
                Constructor::Enum(_) | Constructor::Error if in_places_of(&from.layer, &to.layer).is_none() => {
                    Stage::NoChange
                }
                Constructor::Enum(_) | Constructor::Error if !same_members => goes(&[Coercion::AddVariants]),
                _ if same_members => goes(&[]),
                _ => Stage::NoChange,
            }
        }
        _ => Stage::NoChange,
    }
}

/// The parts of `fewer` with the part `more` has in the same place, where `more`
/// has a place for each.
fn in_places_of<P>(fewer: &Shape<P>, more: &Shape<P>) -> Option<Vec<(P, P)>>
where
    P: Clone,
{
    fewer
        .places()
        .into_iter()
        .map(|(key, part)| more.place(&key).map(|other| (part.clone(), other.clone())))
        .collect()
}

/// Each pair of parts below what a change by `coercions` from `from` to `to` sees,
/// the second corresponding to the first (O9). `shape_at` gives the head at a
/// part, where there is one.
fn below<P, F>(from: &Shape<P>, to: &Shape<P>, coercions: &BTreeSet<Coercion>, shape_at: F) -> Vec<(P, P)>
where
    P: Clone,
    F: Fn(&P) -> Option<Shape<P>>,
{
    match (from, to) {
        (Shape::Never, _) => Vec::new(),
        (Shape::Ref(_, from), Shape::Ref(_, to)) if coercions.contains(&Coercion::Unsize) => {
            let (Some(Shape::Array(_, from)), Some(Shape::Slice(to))) = (shape_at(from), shape_at(to)) else {
                unreachable!("an unsizing goes from a reference to an array to one to a slice");
            };
            vec![(from, to)]
        }
        (Shape::Ref(_, from), Shape::Ref(_, to)) => vec![(from.clone(), to.clone())],
        (Shape::Object(_), Shape::Object(_)) => in_places_of(to, from)
            .unwrap_or_default()
            .into_iter()
            .map(|(kept, field)| (field, kept))
            .collect(),
        _ => in_places_of(from, to).unwrap_or_default(),
    }
}

/// Whether a coercion may go from `head` to `to`, as rung 2 asks of a head an upper
/// bound gives, before the class has parts of its own. The head is the class's own
/// where `within` is none, and the target of a reference of mutability `within`
/// otherwise. A reference's own target is a part, made anew, so of a reference
/// only its mutability is seen. No conversion chooses a head.
pub(super) fn may_coerce<K>(classes: &Classes<K>, head: &Shape<K::Id>, within: Option<Mutability>, to: &Shape<K::Id>) -> bool
where
    K: ClassKind,
{
    let from = match (within, head) {
        (Some(mutability), head) => Point::new(&Shape::Ref(mutability, ()), Some(head)).map(Seen::At),
        (None, Shape::Ref(mutability, _)) => Some(Seen::Reference(*mutability)),
        (None, head) => Point::new(head, None::<&Shape<()>>).map(Seen::At),
    };
    let (Some(from), Some(to)) = (from, seen_to(classes, to)) else {
        return false;
    };
    coercion(&from, &to) != Stage::NoChange
}

/// Whether `bound` is a function value going to a function type, which waits for
/// instances not recorded yet.
pub(super) fn awaits_instances<K>(classes: &Classes<K>, bound: &Bound<K>) -> bool
where
    K: ClassKind<Bound = Bound<K>>,
{
    let (Some(from), Some(to)) = (shape_of(classes, bound.lower()), shape_of(classes, bound.upper())) else {
        return false;
    };
    let (Some(from), Some(to)) = (seen_from(classes, from), seen_to(classes, to)) else {
        return false;
    };
    coercion(&from, &to) == Stage::ToFnType
}

/// A way to go: one stage of coercion, then a chain of conversions.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Route {
    /// The references read through, `&ⁿ U` going as `&U`, before anything else.
    read_through: usize,
    coercions: BTreeSet<Coercion>,
    chain: Vec<Name>,
}

/// A conversion's signature, written once and made anew at each boundary that
/// takes it (O12, O13).
#[derive(Clone, Debug)]
struct Declared {
    template: Classes<Template>,
    from: TemplateId,
    to: TemplateId,
    from_point: Point,
    to_point: Point,
    parameter: Ty,
    result: Ty,
    view: bool,
}

/// A conversion writes no class node at one of its ends, so no change could reach
/// it or go on from it (O12).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NoClassNode;

/// Whether a conversion is a view: both its ends are references, and its parameter
/// writes nothing its class node does not show, which is its target's constructor
/// with no members and a distinct hole at each part.
fn is_view(parameter: &Ty, result: &Ty) -> bool {
    let (Ty::Node(parameter), Ty::Node(result)) = (parameter, result) else {
        return false;
    };
    let (Shape::Ref(_, Ty::Node(target)), Shape::Ref(..)) = (&**parameter, &**result) else {
        return false;
    };
    let holds_members = matches!(
        **target,
        Shape::Object(_) | Shape::Enum(..) | Shape::Error(_) | Shape::FnValue(_)
    );
    let mut holes = BTreeSet::new();
    !holds_members
        && target
            .places()
            .into_iter()
            .all(|(_, part)| matches!(part, Ty::Hole(hole) if holes.insert(hole.clone())))
}

/// What a chain of conversions comes to at a boundary.
enum Chosen {
    One(Route),
    /// Only the argument's whole type can tell, and it is not decided yet.
    Waits,
    NoChange,
}

fn exactly_one(mut routes: Vec<Route>) -> Chosen {
    match (routes.pop(), routes.is_empty()) {
        (Some(route), true) => Chosen::One(route),
        _ => Chosen::NoChange,
    }
}

/// Every conversion declared.
#[derive(Clone, Debug, Default)]
pub struct Conversions {
    declared: BTreeMap<Name, Declared>,
}

impl Conversions {
    /// Writes `conversion` as a template, with one class for each type it writes.
    pub fn declare(&mut self, name: Name, conversion: &Conversion, scopes: &Scopes, by: ItemId) -> Result<(), NoClassNode> {
        let mut template = Classes::default();
        let mut written = BTreeMap::new();
        let from = template.write(scopes.root(), &conversion.from, by, &mut written);
        let to = template.write(scopes.root(), &conversion.to, by, &mut written);
        let point = |class| point_of_shape(&template, shape_of(&template, class)?);
        let from_point = point(from).ok_or(NoClassNode)?;
        let to_point = point(to).ok_or(NoClassNode)?;
        let declared = Declared {
            from,
            to,
            from_point,
            to_point,
            parameter: conversion.from.clone(),
            result: conversion.to.clone(),
            view: is_view(&conversion.from, &conversion.to),
            template,
        };
        self.declared.insert(name, declared);
        Ok(())
    }

    /// The one chain that goes from `from` to `to`, `argument` being what is passed
    /// into it: a chain of views where any goes, and a chain with a value
    /// conversion otherwise, once the argument is decided at every depth.
    fn choose<K>(&self, classes: &Classes<K>, argument: &Shape<K::Id>, from: &Point, to: &Point) -> Chosen
    where
        K: ClassKind,
    {
        let (views, values): (Vec<Route>, Vec<Route>) = self
            .routes(from, to)
            .into_iter()
            .partition(|route| route.chain.iter().all(|name| self.declared[name].view));
        if !views.is_empty() {
            return exactly_one(views);
        }
        if values.is_empty() {
            return Chosen::NoChange;
        }
        let Some(argument) = written_shape(classes, argument) else {
            return Chosen::Waits;
        };
        exactly_one(values.into_iter().filter(|route| self.fits(&argument, route)).collect())
    }

    /// Every chain from `from` to `to`, after one stage of coercion. No chain comes
    /// back to a point it has been at, or it would make another way out of one that
    /// is there already.
    fn routes(&self, from: &Point, to: &Point) -> Vec<Route> {
        let mut routes = Vec::new();
        for (name, declared) in &self.declared {
            if let Stage::Goes(coercions) = stage(from, &declared.from_point) {
                let mut seen = BTreeSet::from([from.clone()]);
                let mut chain = Vec::new();
                self.follow(name, declared, to, &coercions, &mut chain, &mut seen, &mut routes);
            }
        }
        routes
    }

    /// Takes the conversion `name`, and whatever chains on from where it ends.
    #[expect(clippy::too_many_arguments, reason = "a walk's state, which only this walk reads")]
    fn follow(
        &self,
        name: &Name,
        declared: &Declared,
        to: &Point,
        coercions: &BTreeSet<Coercion>,
        chain: &mut Vec<Name>,
        seen: &mut BTreeSet<Point>,
        routes: &mut Vec<Route>,
    ) {
        if !seen.insert(declared.to_point.clone()) {
            return;
        }
        chain.push(name.clone());
        if declared.to_point == *to {
            routes.push(Route {
                read_through: 0,
                coercions: coercions.clone(),
                chain: chain.clone(),
            });
        } else {
            for (next_name, next) in &self.declared {
                if next.from_point == declared.to_point {
                    self.follow(next_name, next, to, coercions, chain, seen, routes);
                }
            }
        }
        chain.pop();
        seen.remove(&declared.to_point);
    }

    /// Whether every parameter of `route`'s chain takes what comes to it, starting
    /// from `argument`, a type with no hole.
    fn fits(&self, argument: &Ty, route: &Route) -> bool {
        let ty_shape = |ty: &Ty| match ty {
            Ty::Node(shape) => Some((**shape).clone()),
            Ty::Hole(_) => None,
        };
        let mut current = argument.clone();
        let mut coercions = route.coercions.clone();
        for name in &route.chain {
            let declared = &self.declared[name];
            let (Ty::Node(from), Ty::Node(parameter)) = (&current, &declared.parameter) else {
                unreachable!("both ends of a conversion write a class node, and so does the argument");
            };
            let mut binding = BTreeMap::new();
            let fits = below(from, parameter, &coercions, ty_shape)
                .iter()
                .all(|(part, pattern)| bind(pattern, part, &mut binding));
            if !fits {
                return false;
            }
            current = instantiate(&declared.result, &binding);
            coercions = BTreeSet::new();
        }
        true
    }
}

/// The type of `class` written out, where it is decided at every depth. A class
/// refused as its own part has no finite type, and gives none.
fn written_whole<K>(classes: &Classes<K>, class: K::Id) -> Option<Ty>
where
    K: ClassKind,
{
    fn within<K>(classes: &Classes<K>, class: K::Id, path: &mut BTreeSet<K::Id>) -> Option<Ty>
    where
        K: ClassKind,
    {
        let root = classes.find(class).id();
        if !path.insert(root) {
            return None;
        }
        let parts = shape_of(classes, root)?.map(|part| within(classes, *part, path));
        path.remove(&root);
        if parts.places().iter().any(|(_, part)| part.is_none()) {
            return None;
        }
        Some(Ty::Node(Box::new(parts.map(|part| part.clone().expect("every part was written")))))
    }
    within(classes, class, &mut BTreeSet::new())
}

/// The type `shape` heads written out, where each of its parts is decided at every
/// depth.
fn written_shape<K>(classes: &Classes<K>, shape: &Shape<K::Id>) -> Option<Ty>
where
    K: ClassKind,
{
    let parts = shape.map(|part| written_whole(classes, *part));
    if parts.places().iter().any(|(_, part)| part.is_none()) {
        return None;
    }
    Some(Ty::Node(Box::new(parts.map(|part| part.clone().expect("every part was written")))))
}

/// Whether `ty` is an instance of `pattern`, each hole of `pattern` bound to one
/// type. A hole of `ty` is a part a result left open, which takes whatever comes.
fn bind(pattern: &Ty, ty: &Ty, binding: &mut BTreeMap<Name, Ty>) -> bool {
    match (pattern, ty) {
        (_, Ty::Hole(_)) => true,
        (Ty::Hole(hole), ty) => match binding.get(hole) {
            Some(bound) => same(bound, ty),
            None => {
                binding.insert(hole.clone(), ty.clone());
                true
            }
        },
        (Ty::Node(pattern), Ty::Node(ty)) => {
            pattern.layer() == ty.layer()
                && pattern
                    .places()
                    .into_iter()
                    .zip(ty.places())
                    .all(|((_, pattern), (_, ty))| bind(pattern, ty, binding))
        }
    }
}

/// Whether two types are one, a hole being whatever the other is.
fn same(first: &Ty, second: &Ty) -> bool {
    match (first, second) {
        (Ty::Hole(_), _) | (_, Ty::Hole(_)) => true,
        (Ty::Node(first), Ty::Node(second)) => {
            first.layer() == second.layer()
                && first
                    .places()
                    .into_iter()
                    .zip(second.places())
                    .all(|((_, first), (_, second))| same(first, second))
        }
    }
}

fn instantiate(ty: &Ty, binding: &BTreeMap<Name, Ty>) -> Ty {
    match ty {
        Ty::Hole(hole) => binding.get(hole).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Node(shape) => Ty::Node(Box::new(shape.map(|part| instantiate(part, binding)))),
    }
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Makes each change that can be made, and says whether it made any.
    pub(super) fn make_changes(&mut self, classes: &mut Classes<K>, scopes: &Scopes, conversions: &Conversions) -> bool {
        let mut made = false;
        for (bound, at) in classes.between_decided() {
            if matches!(bound, Bound::Change { .. }) && self.make_change(classes, scopes, conversions, bound, at) {
                classes.finish(&bound, at);
                made = true;
            }
        }
        self.judge(classes, scopes);
        made
    }

    /// Makes the change `bound` asks for, and says whether it is done with.
    fn make_change(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        conversions: &Conversions,
        bound: Bound<K>,
        at: ItemId,
    ) -> bool {
        let (Some(written), Some(to)) = (shape_of(classes, bound.lower()), shape_of(classes, bound.upper())) else {
            return false;
        };
        let Some(to) = seen_to(classes, to) else {
            return false;
        };
        // The argument as it is written, then with one more reference read through
        // each time: an index is loosened only where the types do not match as they
        // are (Subtyping the language gives).
        let mut from_shape = written.clone();
        let mut read_through = 0;
        let route = loop {
            let Some(from) = seen_from(classes, &from_shape) else {
                return false;
            };
            match coercion(&from, &to) {
                Stage::Goes(coercions) => {
                    let route = Route {
                        read_through,
                        coercions,
                        chain: Vec::new(),
                    };
                    break Some(route);
                }
                Stage::ToFnType | Stage::Waits => return false,
                Stage::NoChange => {}
            }
            let Some(inner) = read_through_once(classes, &from_shape) else {
                break None;
            };
            from_shape = inner;
            read_through += 1;
        };
        let route = match route {
            Some(route) => route,
            // Nothing the language gives goes alone, so a declared conversion is
            // looked at. What the language gives is taken on the way into it, the
            // reading through of references among it, as few as will do.
            None => {
                let mut passed = written.clone();
                let mut read_through = 0;
                loop {
                    let Some(from) = seen_from(classes, &passed) else {
                        return false;
                    };
                    let chosen = match (&from, &to) {
                        (Seen::At(from), Seen::At(to)) => conversions.choose(classes, &passed, from, to),
                        // Its class node may yet let a conversion go.
                        (Seen::Reference(_), Seen::At(_)) => Chosen::Waits,
                        // No conversion reaches a hole, which takes only what comes.
                        (_, Seen::Reference(_)) => Chosen::NoChange,
                    };
                    match chosen {
                        Chosen::One(route) => break Route { read_through, ..route },
                        Chosen::Waits => return false,
                        Chosen::NoChange => {}
                    }
                    let Some(inner) = read_through_once(classes, &passed) else {
                        self.refuse_no_change(classes, &bound, at);
                        return true;
                    };
                    passed = inner;
                    read_through += 1;
                }
            }
        };
        self.take(classes, scopes, conversions, &bound, at, &route);
        true
    }

    fn refuse_no_change(&mut self, classes: &Classes<K>, bound: &Bound<K>, at: ItemId) {
        let mut items = decided_by(classes, bound.lower());
        items.extend(decided_by(classes, bound.upper()));
        items.insert(at);
        self.refusals.push(Refusal {
            cause: Cause::NoChange,
            items,
        });
    }

    /// Takes `route` at the boundary: records it, makes each conversion's signature
    /// anew, and connects the parts below each step.
    fn take(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        conversions: &Conversions,
        bound: &Bound<K>,
        at: ItemId,
        route: &Route,
    ) {
        let adjustments = std::iter::repeat_n(Adjustment::Deref, route.read_through)
            .chain(route.coercions.iter().copied().map(Adjustment::Coerce))
            .chain(route.chain.iter().cloned().map(Adjustment::Convert));
        for adjustment in adjustments {
            self.adjustments.entry(at).or_default().push(adjustment);
        }
        let scope = classes.get(classes.find(bound.lower())).scope;
        let Some(mut current) = shape_of(classes, bound.lower()).cloned() else {
            unreachable!("a change is made between decided classes");
        };
        for _ in 0..route.read_through {
            let Some(inner) = read_through_once(classes, &current) else {
                unreachable!("the route counts the references that can be read through");
            };
            current = inner;
        }
        let mut coercions = route.coercions.clone();
        for name in &route.chain {
            let declared = &conversions.declared[name];
            let copied = classes.copy_from(&declared.template, scopes, scope, at, |_| Fate::Anew);
            self.connect(classes, scopes, &current, copied.classes[&declared.from], &coercions, at);
            coercions = BTreeSet::new();
            let Some(result) = shape_of(classes, copied.classes[&declared.to]).cloned() else {
                unreachable!("a conversion writes a class node at its result");
            };
            current = result;
        }
        self.connect(classes, scopes, &current, bound.upper(), &coercions, at);
    }

    fn connect(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        from: &Shape<K::Id>,
        to: K::Id,
        coercions: &BTreeSet<Coercion>,
        at: ItemId,
    ) {
        let Some(to) = shape_of(classes, to) else {
            unreachable!("a change is made between decided classes");
        };
        let pairs = below(from, to, coercions, |class: &K::Id| shape_of(classes, *class).cloned());
        for (part, corresponding) in pairs {
            self.one_type(classes, scopes, part, corresponding, at);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::operation::settle::Settled;
    use crate::operation::testing::{EXCLUSIVE, SHARED, U8, World};
    use crate::primitive::adjustment::Adjustments;
    use crate::primitive::class::Live;
    use crate::primitive::id::Arena;
    use crate::primitive::ty::{Scalar, Ty};
    use crate::primitive::{ClassId, ClosureId};

    const U16: Shape<ClassId> = Shape::Scalar(Scalar::U16);

    fn object(world: &mut World, fields: &[&str], declared: bool) -> ClassId {
        let mut held = BTreeMap::new();
        for field in fields {
            let part = if declared { world.external(U8) } else { world.text(U8) };
            held.insert(Name::new(*field), part);
        }
        if declared {
            world.external(Shape::Object(held))
        } else {
            world.text(Shape::Object(held))
        }
    }

    fn coerced(at: ItemId, coercions: &[Coercion]) -> Adjustments {
        let adjustments = coercions.iter().copied().map(Adjustment::Coerce).collect();
        BTreeMap::from([(at, adjustments)])
    }

    fn refused_at(world: &World, cause: Cause, at: ItemId) {
        assert_eq!(world.outcome.refusals.len(), 1, "{:?}", world.outcome.refusals);
        assert_eq!(world.outcome.refusals[0].cause, cause);
        assert!(world.outcome.refusals[0].items.contains(&at));
    }

    #[test]
    fn the_empty_change_records_no_adjustment() {
        let mut world = World::new();
        let argument = world.text(U8);
        let parameter = world.external(U8);
        world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, Adjustments::new());
        assert_eq!(world.classes.between_decided().len(), 0);
    }

    /// `take(return 0);`, with `take(u8)`.
    #[test]
    fn never_goes_to_any_type() {
        let mut world = World::new();
        let argument = world.text(Shape::Never);
        let parameter = world.external(U8);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Never]));
    }

    #[test]
    fn never_going_to_never_is_the_empty_change() {
        let mut world = World::new();
        let argument = world.text(Shape::Never);
        let parameter = world.external(Shape::Never);
        world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    #[test]
    fn nothing_but_never_goes_to_never() {
        let mut world = World::new();
        let argument = world.text(U8);
        let parameter = world.external(Shape::Never);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn a_reference_goes_from_exclusive_to_shared_and_not_back() {
        let mut world = World::new();
        let x = world.text(U8);
        let (exclusive, _) = world.lend(x, EXCLUSIVE);
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let (_, at) = world.pass(exclusive, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Reborrow]));

        let mut world = World::new();
        let x = world.text(U8);
        let (shared, _) = world.lend(x, SHARED);
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(EXCLUSIVE, declared));
        let (_, at) = world.pass(shared, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `&mut [T; n]` to `&[T]` takes both in the one stage, and the element below
    /// is one type.
    #[test]
    fn an_array_behind_a_reference_goes_to_a_slice() {
        for (mutability, expected) in [
            (SHARED, &[Coercion::Unsize][..]),
            (EXCLUSIVE, &[Coercion::Reborrow, Coercion::Unsize][..]),
        ] {
            let mut world = World::new();
            let elem = world.text(U8);
            let array = world.text(Shape::Array(2, elem));
            let (lent, _) = world.lend(array, mutability);
            let hole = world.undecided();
            let slice = world.external(Shape::Slice(hole));
            let parameter = world.external(Shape::Ref(SHARED, slice));
            let (_, at) = world.pass(lent, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            world.no_refusals();
            assert_eq!(world.outcome.adjustments, coerced(at, expected));
            assert!(world.same(elem, hole));
        }
    }

    /// An array by value has no unsizing: its size is part of the value.
    #[test]
    fn an_array_not_behind_a_reference_does_not_go_to_a_slice() {
        let mut world = World::new();
        let elem = world.text(U8);
        let array = world.text(Shape::Array(2, elem));
        let hole = world.undecided();
        let slice = world.external(Shape::Slice(hole));
        let (_, at) = world.pass(array, slice);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn a_slice_behind_a_reference_does_not_go_to_an_array() {
        let mut world = World::new();
        let elem = world.undecided();
        let slice = world.text(Shape::Slice(elem));
        let (lent, _) = world.lend(slice, SHARED);
        let declared = world.external(U8);
        let array = world.external(Shape::Array(2, declared));
        let parameter = world.external(Shape::Ref(SHARED, array));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn an_object_goes_to_one_with_only_some_of_its_fields() {
        let mut world = World::new();
        let argument = object(&mut world, &["x", "y"], false);
        let parameter = object(&mut world, &["x"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::DropFields]));
    }

    #[test]
    fn an_object_does_not_go_to_one_with_a_field_it_lacks() {
        let mut world = World::new();
        let argument = object(&mut world, &["x"], false);
        let parameter = object(&mut world, &["x", "y"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// The fields that stay are parts below the class node, so no change alters
    /// them: `u16` does not reach a declared `u8` there.
    #[test]
    fn a_field_that_stays_is_one_type_with_the_field_it_goes_to() {
        let mut world = World::new();
        let sixteen = world.text(U16);
        let y = world.text(U8);
        let argument = world.text(Shape::Object(BTreeMap::from([
            (Name::new("x"), sixteen),
            (Name::new("y"), y),
        ])));
        let parameter = object(&mut world, &["x"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
    }

    /// Members of a target are not in a reference's class node, so what lies
    /// behind it is one type: the fields of what is lent cannot be dropped.
    #[test]
    fn an_object_behind_a_reference_keeps_its_fields() {
        let mut world = World::new();
        let argument = object(&mut world, &["x", "y"], false);
        let (lent, _) = world.lend(argument, SHARED);
        let target = object(&mut world, &["x"], true);
        let parameter = world.external(Shape::Ref(SHARED, target));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// The same holds of variants: a reference points at the value as it is, and
    /// no change makes another value for it to point at.
    #[test]
    fn an_enum_behind_a_reference_keeps_its_variants() {
        let mut world = World::new();
        let payload = world.external(U8);
        let one = BTreeMap::from([(Name::new("A"), payload)]);
        let two = BTreeMap::from([(Name::new("A"), payload), (Name::new("B"), payload)]);
        let argument = world.external(Shape::Enum(Name::new("E"), one));
        let (lent, _) = world.lend(argument, SHARED);
        let target = world.external(Shape::Enum(Name::new("E"), two));
        let parameter = world.external(Shape::Ref(SHARED, target));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    #[test]
    fn an_enum_or_an_error_goes_to_one_with_more_variants_and_not_fewer() {
        let enumerate = |variants: &[&str], part: ClassId| -> BTreeMap<Name, ClassId> {
            variants.iter().map(|variant| (Name::new(*variant), part)).collect()
        };
        type Kind = fn(BTreeMap<Name, ClassId>) -> Shape<ClassId>;
        let kinds: [Kind; 2] = [|variants| Shape::Enum(Name::new("E"), variants), Shape::Error];
        for kind in kinds {
            let mut world = World::new();
            let payload = world.external(U8);
            let argument = world.external(kind(enumerate(&["A"], payload)));
            let parameter = world.external(kind(enumerate(&["A", "B"], payload)));
            let (_, at) = world.pass(argument, parameter);
            assert_eq!(world.settle_fully(), Settled::Finished);
            world.no_refusals();
            assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::AddVariants]));

            let mut world = World::new();
            let payload = world.external(U8);
            let argument = world.external(kind(enumerate(&["A", "B"], payload)));
            let parameter = world.external(kind(enumerate(&["A"], payload)));
            let (_, at) = world.pass(argument, parameter);
            assert_eq!(world.settle_fully(), Settled::Finished);
            refused_at(&world, Cause::NoChange, at);
        }
    }

    #[test]
    fn enums_of_two_names_do_not_change_into_one_another() {
        let mut world = World::new();
        let argument = world.external(Shape::Enum(Name::new("E"), BTreeMap::new()));
        let parameter = world.external(Shape::Enum(Name::new("F"), BTreeMap::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// Where a function value is passed its closures are neither added to nor taken
    /// from: their parameters carry bounds, so it is invariant there.
    #[test]
    fn a_function_value_goes_only_to_one_with_the_same_closures() {
        let mut closures = Arena::<ClosureId, ()>::default();
        let (first, second) = (closures.push(()), closures.push(()));
        let values = |ids: &[ClosureId]| Shape::<ClassId>::FnValue(ids.iter().copied().collect());

        let mut world = World::new();
        let argument = world.external(values(&[first, second]));
        let parameter = world.external(values(&[first, second]));
        world.pass(argument, parameter);
        assert_eq!(world.settle_fully(), Settled::Finished);
        world.no_refusals();
        assert_eq!(world.outcome.adjustments, Adjustments::new());

        let mut world = World::new();
        let argument = world.external(values(&[first]));
        let parameter = world.external(values(&[first, second]));
        let (_, at) = world.pass(argument, parameter);
        assert_eq!(world.settle_fully(), Settled::Finished);
        refused_at(&world, Cause::NoChange, at);

        let mut world = World::new();
        let argument = world.external(values(&[first, second]));
        let parameter = world.external(values(&[first]));
        let (_, at) = world.pass(argument, parameter);
        assert_eq!(world.settle_fully(), Settled::Finished);
        refused_at(&world, Cause::NoChange, at);
    }

    /// Its instances are not recorded yet (O30), so it is not made, and settling
    /// says so.
    #[test]
    fn a_function_value_going_to_a_function_type_waits() {
        let mut closures = Arena::<ClosureId, ()>::default();
        let closure = closures.push(());
        let mut world = World::new();
        let argument = world.external(Shape::FnValue(BTreeSet::from([closure])));
        let result = world.external(U8);
        let parameter = world.external(Shape::Fn(Vec::new(), result));
        let (boundary, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::AwaitsInstances);

        world.no_refusals();
        assert_eq!(
            world.classes.between_decided(),
            BTreeSet::from([(
                Bound::Change {
                    lower: boundary,
                    upper: parameter
                },
                at
            )])
        );
    }

    #[test]
    fn heads_of_two_constructors_do_not_change_into_one_another() {
        let mut world = World::new();
        let argument = world.text(U16);
        let parameter = world.external(U8);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `let a; g(&a);`, with `g(&[u8])`: the boundary's head is `&`, and what its
    /// target is is not known, so no change can be chosen. The head is given by hand
    /// here, to one upper bounds would not give: it is the wait that is looked at.
    #[test]
    fn a_change_waits_for_the_target_of_a_reference() {
        let mut world = World::new();
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        let (boundary, at) = world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(world.classes.is_decided(boundary));
        assert_eq!(world.classes.between_decided().len(), 1);
        assert_eq!(world.outcome.adjustments, Adjustments::new());

        let elem = world.external(U8);
        world.decide(a, Shape::Array(2, elem));
        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Unsize]));
    }

    /// `a` is decided as a reference, and its target is not: a class node is not
    /// partly decided, however much of it would do.
    #[test]
    fn no_class_node_is_partly_decided() {
        let mut world = World::new();
        let target = world.undecided();
        let argument = world.external(Shape::Ref(SHARED, target));
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let boundary = world.external(Shape::Ref(SHARED, target));
        let at = world.record(Bound::Change {
            lower: boundary,
            upper: parameter,
        });
        world.enter(argument, boundary);

        world.settle();

        assert_eq!(
            world.classes.between_decided(),
            BTreeSet::from([(
                Bound::Change {
                    lower: boundary,
                    upper: parameter
                },
                at
            )])
        );
    }

    /// Whatever the coercion, a part takes the declared part it corresponds to as
    /// an upper bound, and stays undecided for what enters it to decide. Taken the
    /// other way round, the two would be one class, and the part decided already.
    #[test]
    fn a_part_takes_the_declared_part_as_an_upper_bound_under_every_coercion() {
        type Heads = fn(ClassId, ClassId) -> (Shape<ClassId>, Shape<ClassId>);
        fn named(entries: &[(&str, ClassId)]) -> BTreeMap<Name, ClassId> {
            entries.iter().map(|(name, part)| (Name::new(*name), *part)).collect()
        }
        let kinds: [Heads; 5] = [
            |part, declared| (Shape::Option(part), Shape::Option(declared)),
            |part, declared| (Shape::Tuple(vec![part]), Shape::Tuple(vec![declared])),
            |part, declared| {
                (
                    Shape::Object(named(&[("A", part), ("B", part)])),
                    Shape::Object(named(&[("A", declared)])),
                )
            },
            |part, declared| {
                (
                    Shape::Enum(Name::new("E"), named(&[("A", part)])),
                    Shape::Enum(Name::new("E"), named(&[("A", declared), ("B", declared)])),
                )
            },
            |part, declared| {
                (
                    Shape::Error(named(&[("A", part)])),
                    Shape::Error(named(&[("A", declared), ("B", declared)])),
                )
            },
        ];
        for kind in kinds {
            let mut world = World::new();
            let part = world.undecided();
            let declared = world.external(U8);
            let (from, to) = kind(part, declared);
            let argument = world.text(from);
            let parameter = world.external(to);
            world.pass(argument, parameter);

            world.settle();

            world.no_refusals();
            assert!(!world.classes.is_decided(part));
            assert!(world.classes.upper_items(part).contains_key(&world.classes.find(declared).id()));

            assert_eq!(world.settle_fully(), Settled::Finished);

            world.no_refusals();
            assert_eq!(world.shape(part), U8);
        }
    }

    /// The same behind a reference, where the element is what lies below the
    /// class node.
    #[test]
    fn an_element_takes_the_declared_element_as_an_upper_bound_when_unsized() {
        let mut world = World::new();
        let elem = world.undecided();
        let array = world.text(Shape::Array(2, elem));
        let (lent, _) = world.lend(array, SHARED);
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(elem));
        assert!(world.classes.upper_items(elem).contains_key(&world.classes.find(declared).id()));

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(elem), U8);
    }

    fn vec_of(element: Ty) -> Ty {
        node(Shape::Struct(Name::new("Vec"), vec![element]))
    }

    fn node(shape: Shape<Ty>) -> Ty {
        Ty::Node(Box::new(shape))
    }

    fn hole() -> Ty {
        Ty::Hole(Name::new("T"))
    }

    /// `as_slice: &Vec<T> -> &[T]`.
    fn declare_as_slice(world: &mut World) {
        let from = node(Shape::Ref(SHARED, vec_of(hole())));
        let to = node(Shape::Ref(SHARED, node(Shape::Slice(hole()))));
        world.declare("as_slice", from, to);
    }

    /// `as_slice_mut: &mut Vec<T> -> &mut [T]`.
    fn declare_as_slice_mut(world: &mut World) {
        let from = node(Shape::Ref(EXCLUSIVE, vec_of(hole())));
        let to = node(Shape::Ref(EXCLUSIVE, node(Shape::Slice(hole()))));
        world.declare("as_slice_mut", from, to);
    }

    fn vec_class(world: &mut World, element: ClassId) -> ClassId {
        world.external(Shape::Struct(Name::new("Vec"), vec![element]))
    }

    fn reference_to_slice(world: &mut World, mutability: Mutability, element: ClassId) -> ClassId {
        let slice = world.external(Shape::Slice(element));
        world.external(Shape::Ref(mutability, slice))
    }

    fn conversions(coercions: &[Coercion], names: &[&str]) -> Vec<Adjustment> {
        coercions
            .iter()
            .copied()
            .map(Adjustment::Coerce)
            .chain(names.iter().map(|name| Adjustment::Convert(Name::new(*name))))
            .collect()
    }

    /// Example 5: `let v = new(); sum(&v);`, with `new() -> Vec<T>`, `sum(&[u8])`
    /// and `as_slice`. Its `T` is made anew: the parameter's side makes it one with
    /// `v`'s element, and the result's side gives it the upper bound `u8`.
    #[test]
    fn a_conversion_connects_a_hole_on_both_its_sides() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let element = world.undecided();
        let result = vec_class(&mut world, element);
        let v = world.meets(result);
        let (lent, _) = world.lend(v, SHARED);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(element), U8);
        assert!(!world.same(element, declared));
        assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["as_slice"]))]));
    }

    /// Example 13: `let mut v = bytes(); sum(&mut v);`, with `bytes() -> Vec<u8>` and
    /// `sum(&[u8])`. `&mut -> &` then `as_slice` is the one change: `as_slice_mut`
    /// ends at `&mut [u8]`, and no coercion follows it.
    #[test]
    fn a_coercion_goes_before_a_conversion_and_none_after() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        declare_as_slice_mut(&mut world);
        let byte = world.external(U8);
        let result = vec_class(&mut world, byte);
        let v = world.meets(result);
        let (lent, _) = world.lend(v, EXCLUSIVE);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(
            world.outcome.adjustments,
            BTreeMap::from([(at, conversions(&[Coercion::Reborrow], &["as_slice"]))])
        );
    }

    #[test]
    fn no_coercion_follows_a_conversion() {
        let mut world = World::new();
        declare_as_slice_mut(&mut world);
        let byte = world.external(U8);
        let result = vec_class(&mut world, byte);
        let v = world.meets(result);
        let (lent, _) = world.lend(v, EXCLUSIVE);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// A class takes a head an upper bound gives, and never one only a conversion
    /// goes from: `let a; g(&a);` does not make `a` a `Vec`.
    #[test]
    fn a_head_only_a_conversion_goes_from_is_not_given() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert!(matches!(world.shape(a), Shape::Slice(_)));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `A -> B -> C`, an auto-conversion, each signature made anew.
    #[test]
    fn conversions_chain() {
        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("a_to_b", named("A"), named("B"));
        world.declare("b_to_c", named("B"), named("C"));
        let argument = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let parameter = world.external(Shape::Struct(Name::new("C"), Vec::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["a_to_b", "b_to_c"]))]));
    }

    /// `Vec<u16>` to `&[u8]`: the conversion goes, and what lies below it does not
    /// change, so the element is refused there.
    #[test]
    fn a_conversion_does_not_change_what_lies_below_it() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let sixteen = world.external(U16);
        let result = vec_class(&mut world, sixteen);
        let v = world.meets(result);
        let (lent, _) = world.lend(v, SHARED);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
    }

    /// Each boundary makes the signature anew, so its `T` is its own.
    #[test]
    fn each_boundary_has_its_own_copy_of_a_conversion() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        for head in [U8, U16] {
            let element = world.external(head.clone());
            let result = vec_class(&mut world, element);
            let v = world.meets(result);
            let (lent, _) = world.lend(v, SHARED);
            let declared = world.external(head);
            let parameter = reference_to_slice(&mut world, SHARED, declared);
            world.pass(lent, parameter);
        }

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
    }

    /// Exactly one change must go (O11).
    #[test]
    fn two_ways_to_go_are_no_change() {
        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("one", named("A"), named("B"));
        world.declare("other", named("A"), named("B"));
        let argument = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let parameter = world.external(Shape::Struct(Name::new("B"), Vec::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `A -> B`, `B -> A`: going round is no second way from `A` to `A`, and the
    /// search ends.
    #[test]
    fn a_chain_does_not_come_back_to_where_it_has_been() {
        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("a_to_b", named("A"), named("B"));
        world.declare("b_to_a", named("B"), named("A"));
        let argument = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let same = world.external(Shape::Struct(Name::new("A"), Vec::new()));
        world.pass(argument, same);
        let other = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let nowhere = world.external(Shape::Struct(Name::new("C"), Vec::new()));
        let (_, at) = world.pass(other, nowhere);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `take(return 0);`, with `take(&[u8])` and a conversion that ends there. A
    /// coercion goes from `!` to anything, so no conversion is looked at.
    #[test]
    fn never_goes_by_the_coercion_alone() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let argument = world.text(Shape::Never);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Never]));
    }

    /// A signature writes the class node of both its ends (O12).
    #[test]
    fn a_conversion_with_no_class_node_is_not_declared() {
        let scopes = Scopes::new();
        let mut items = Arena::<ItemId, ()>::default();
        let mut declared = Conversions::default();
        let to = node(Shape::Slice(hole()));
        let conversion = Conversion { from: hole(), to };
        assert_eq!(
            declared.declare(Name::new("from_a_hole"), &conversion, &scopes, items.push(())),
            Err(NoClassNode)
        );
        let target_is_a_hole = node(Shape::Ref(SHARED, hole()));
        let conversion = Conversion {
            from: target_is_a_hole,
            to: node(Shape::Unit),
        };
        assert_eq!(
            declared.declare(Name::new("through_a_hole"), &conversion, &scopes, items.push(())),
            Err(NoClassNode)
        );
    }

    /// A coercion goes before the chain, and nowhere in it: `A -> &mut V` and
    /// `&V -> C` do not chain, though `&mut V` could reach `&V`.
    #[test]
    fn no_coercion_goes_between_conversions() {
        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("a_to_v", named("A"), node(Shape::Ref(EXCLUSIVE, named("V"))));
        world.declare("v_to_c", node(Shape::Ref(SHARED, named("V"))), named("C"));
        let argument = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let parameter = world.external(Shape::Struct(Name::new("C"), Vec::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// Example 12: `take(Some(&mut x));`, with `take(Option<&T>)`. The change
    /// between the two `Option`s is empty, and alters no part.
    #[test]
    fn a_part_is_not_changed_at_a_boundary() {
        let mut world = World::new();
        let x = world.text(U8);
        let (exclusive, _) = world.lend(x, EXCLUSIVE);
        let some = world.text(Shape::Option(exclusive));
        let hole = world.undecided();
        let reference = world.external(Shape::Ref(SHARED, hole));
        let parameter = world.external(Shape::Option(reference));
        let (_, at) = world.pass(some, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
    }

    /// Example 6: `let mut buf = new(); push(&mut buf, 1u16); consume(buf);`, with
    /// `new() -> Option<T>`, `push(&mut Option<T>, T)` and `consume(Option<u8>)`.
    /// The boundary does not change what enters a part: `buf` is `Option<u16>`,
    /// and it is `consume` that refuses it, whichever class is visited first.
    #[test]
    fn a_boundary_does_not_change_what_enters_a_part() {
        for reversed in [false, true] {
            let mut world = World::new();
            let new_hole = world.undecided();
            let result = world.external(Shape::Option(new_hole));
            let buf = world.meets(result);

            let push_hole = world.undecided();
            let push_option = world.external(Shape::Option(push_hole));
            let push_first = world.external(Shape::Ref(EXCLUSIVE, push_option));
            let (lent, _) = world.lend(buf, EXCLUSIVE);
            world.pass(lent, push_first);
            let sixteen = world.text(U16);
            world.enter(sixteen, push_hole);

            let declared_u8 = world.external(U8);
            let consume = world.external(Shape::Option(declared_u8));
            let (_, consume_at) = world.pass(buf, consume);
            if reversed {
                world.open.reverse();
            }

            assert_eq!(world.settle_fully(), Settled::Finished);

            refused_at(&world, Cause::Mismatch, consume_at);
            assert!(world.same(new_hole, push_hole));
            assert_eq!(world.shape(buf), Shape::Option(new_hole));
            assert_eq!(world.shape(new_hole), U16);
        }
    }

    /// Example 4: `let a = new(); fill_all(&mut a);`,
    /// with `new() -> Option<T>` and `fill_all(&mut Option<u8>)`. The part has the
    /// upper bound `u8`, and takes it once nothing else moves.
    #[test]
    fn a_part_a_boundary_reaches_has_the_part_it_corresponds_to_as_an_upper_bound() {
        let mut world = World::new();
        let hole = world.undecided();
        let result = world.external(Shape::Option(hole));
        let a = world.meets(result);
        let (lent, _) = world.lend(a, EXCLUSIVE);
        let declared_u8 = world.external(U8);
        let declared = world.external(Shape::Option(declared_u8));
        let parameter = world.external(Shape::Ref(EXCLUSIVE, declared));
        world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(hole));
        let uppers: Vec<ClassId> = world.classes.upper_items(hole).into_keys().collect();
        assert_eq!(uppers, vec![world.classes.find(declared_u8).id()]);
        assert_eq!(world.classes.between_decided(), crate::primitive::class::Bounds::<Live>::new());

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(hole), U8);
    }

    /// Example 7: `let a; g(&a); h(&a);`, with `g(&[u8])`, `h(&Vec<u8>)` and
    /// `as_slice`, and `let x; f(x); g(x);`, with `f(A)`, `g(B)` and `a_to_b: A -> B`.
    /// A head goes to a boundary by a coercion alone, so in each, no one head goes to
    /// both, and the class is refused.
    #[test]
    fn no_conversion_chooses_a_head() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let a = world.undecided();
        let (first, _) = world.lend(a, SHARED);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        world.pass(first, parameter);
        let (second, _) = world.lend(a, SHARED);
        let declared = world.external(U8);
        let vector = vec_class(&mut world, declared);
        let parameter = world.external(Shape::Ref(SHARED, vector));
        world.pass(second, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(!world.classes.is_decided(a));
        assert!(world.outcome.refusals.iter().all(|refusal| refusal.cause == Cause::UndecidedClass));
        assert_eq!(world.outcome.adjustments, Adjustments::new());

        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("a_to_b", named("A"), named("B"));
        let x = world.undecided();
        let a = world.external(Shape::Struct(Name::new("A"), Vec::new()));
        let b = world.external(Shape::Struct(Name::new("B"), Vec::new()));
        world.pass(x, a);
        world.pass(x, b);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(!world.classes.is_decided(x));
        assert!(world.outcome.refusals.iter().all(|refusal| refusal.cause == Cause::UndecidedClass));
    }

    /// `let v = Vec::new(); read(&mut v);`, with `read(&mut [u8])` and
    /// `as_slice_mut`. A view is chosen by class nodes, so the element is what
    /// `read` writes.
    #[test]
    fn a_view_lets_a_parameter_decide_what_lies_below_it() {
        let mut world = World::new();
        declare_as_slice_mut(&mut world);
        let element = world.undecided();
        let result = vec_class(&mut world, element);
        let v = world.meets(result);
        let (lent, _) = world.lend(v, EXCLUSIVE);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, EXCLUSIVE, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(element), U8);
        assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["as_slice_mut"]))]));
    }

    #[test]
    fn a_view_is_a_conversion_between_references_whose_parameter_shows_all_it_writes() {
        let mut world = World::new();
        let named = |name: &str, parts: Vec<Ty>| node(Shape::Struct(Name::new(name), parts));
        let reference = |ty: Ty| node(Shape::Ref(SHARED, ty));
        let u = || Ty::Hole(Name::new("U"));
        let fields = BTreeMap::from([(Name::new("x"), hole())]);
        let cases = [
            ("as_slice", reference(vec_of(hole())), reference(node(Shape::Slice(hole()))), true),
            ("pair", reference(named("Pair", vec![hole(), u()])), reference(node(Shape::Str)), true),
            ("same_twice", reference(named("Pair", vec![hole(), hole()])), reference(node(Shape::Str)), false),
            ("written_below", reference(vec_of(node(Shape::Scalar(Scalar::U8)))), reference(node(Shape::Str)), false),
            ("members", reference(node(Shape::Object(fields))), reference(node(Shape::Str)), false),
            ("by_value", vec_of(hole()), reference(node(Shape::Slice(hole()))), false),
            ("to_a_value", reference(vec_of(hole())), node(Shape::Slice(hole())), false),
        ];
        for (name, from, to, view) in cases {
            world.declare(name, from, to);
            assert_eq!(world.conversions.declared[&Name::new(name)].view, view, "{name}");
        }
    }

    /// `as_bytes: &Vec<u8> -> &str` and `as_words: &Vec<u16> -> &str` write below
    /// their class node, so they wait for the argument's whole type, and are chosen
    /// by it.
    #[test]
    fn a_value_conversion_is_chosen_by_the_whole_type() {
        for (element, declared, expected) in [
            (U16, &["as_bytes", "as_words"][..], Some("as_words")),
            (U8, &["as_bytes", "as_words"][..], Some("as_bytes")),
            (U16, &["as_bytes"][..], None),
        ] {
            let mut world = World::new();
            for name in declared {
                let scalar = if *name == "as_bytes" { Scalar::U8 } else { Scalar::U16 };
                let from = node(Shape::Ref(SHARED, vec_of(node(Shape::Scalar(scalar)))));
                world.declare(name, from, node(Shape::Ref(SHARED, node(Shape::Str))));
            }
            let element = world.external(element);
            let result = vec_class(&mut world, element);
            let v = world.meets(result);
            let (lent, _) = world.lend(v, SHARED);
            let text = world.external(Shape::Str);
            let parameter = world.external(Shape::Ref(SHARED, text));
            let (_, at) = world.pass(lent, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            match expected {
                Some(name) => {
                    world.no_refusals();
                    assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &[name]))]));
                }
                None => refused_at(&world, Cause::NoChange, at),
            }
        }
    }

    /// `let a = 1u8; let b; f((a, b));`, with `f(P)`. A value conversion waits for
    /// `b`, and gives it nothing: it falls back to `!` as if no conversion were
    /// declared. `pair: (T, T) -> P` then does not fit `(u8, !)`, and `(T, U) -> P`
    /// does.
    #[test]
    fn a_value_conversion_decides_none_of_its_arguments_types() {
        let u = || Ty::Hole(Name::new("U"));
        let p = || node(Shape::Struct(Name::new("P"), Vec::new()));
        for (second, fits) in [(hole(), false), (u(), true)] {
            let mut world = World::new();
            world.declare("pair", node(Shape::Tuple(vec![hole(), second])), p());
            let a = world.text(U8);
            let b = world.undecided();
            let tuple = world.text(Shape::Tuple(vec![a, b]));
            let parameter = world.external(Shape::Struct(Name::new("P"), Vec::new()));
            let (_, at) = world.pass(tuple, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            assert_eq!(world.shape(b), Shape::Never);
            assert!(!world.same(a, b));
            if fits {
                world.no_refusals();
                assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["pair"]))]));
            } else {
                refused_at(&world, Cause::NoChange, at);
            }
        }
    }

    /// `f((1u8, 'c'))`, with `f(P)` and `pair: (T, T) -> P`: the conversion is not
    /// taken, rather than taken and refused.
    #[test]
    fn a_value_conversion_its_argument_does_not_fit_is_not_taken() {
        let mut world = World::new();
        let p = node(Shape::Struct(Name::new("P"), Vec::new()));
        world.declare("pair", node(Shape::Tuple(vec![hole(), hole()])), p);
        let (a, b) = (world.text(U8), world.text(Shape::Scalar(Scalar::Char)));
        let tuple = world.text(Shape::Tuple(vec![a, b]));
        let parameter = world.external(Shape::Struct(Name::new("P"), Vec::new()));
        let (_, at) = world.pass(tuple, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
        assert!(!world.same(a, b));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `A -> B -> C` as values, the argument's type being known whole.
    #[test]
    fn a_chain_carries_its_parts_through_every_signature() {
        let mut world = World::new();
        let of = |name: &str| node(Shape::Struct(Name::new(name), vec![hole()]));
        world.declare("box_to_opt", of("Box"), of("Opt"));
        world.declare("opt_to_list", of("Opt"), of("List"));
        let element = world.text(U8);
        let argument = world.text(Shape::Struct(Name::new("Box"), vec![element]));
        let declared = world.external(U8);
        let parameter = world.external(Shape::Struct(Name::new("List"), vec![declared]));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(
            world.outcome.adjustments,
            BTreeMap::from([(at, conversions(&[], &["box_to_opt", "opt_to_list"]))])
        );
    }

    /// The same chain from `Box<?x>`: the conversions wait for `?x`, which nothing
    /// decides but its fallback, and `List<!>` is refused where `List<u8>` is given.
    #[test]
    fn a_chain_of_value_conversions_gives_no_part_a_type() {
        let mut world = World::new();
        let of = |name: &str| node(Shape::Struct(Name::new(name), vec![hole()]));
        world.declare("box_to_opt", of("Box"), of("Opt"));
        world.declare("opt_to_list", of("Opt"), of("List"));
        let element = world.undecided();
        let argument = world.text(Shape::Struct(Name::new("Box"), vec![element]));
        let declared = world.external(U8);
        let parameter = world.external(Shape::Struct(Name::new("List"), vec![declared]));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert_eq!(world.shape(element), Shape::Never);
        refused_at(&world, Cause::Mismatch, at);
    }

    /// A coercion goes first, and a conversion only where none does: one declared
    /// from `&[T; 2]` to `&[T]` is never taken.
    #[test]
    fn a_coercion_is_taken_before_any_conversion() {
        let mut world = World::new();
        let array = node(Shape::Ref(SHARED, node(Shape::Array(2, hole()))));
        let slice = node(Shape::Ref(SHARED, node(Shape::Slice(hole()))));
        world.declare("array_as_slice", array, slice);
        let element = world.text(U8);
        let value = world.text(Shape::Array(2, element));
        let (lent, _) = world.lend(value, SHARED);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Unsize]));
    }

    /// `f(&x)`, with `f(&T)` and the like: a hole at a reference's target takes the
    /// target that comes, and only the reference itself changes.
    #[test]
    fn a_hole_at_the_target_takes_the_target_that_comes() {
        type Case = (&'static str, Mutability, Mutability, fn(&mut World) -> ClassId, Option<&'static [Coercion]>);
        let cases: [Case; 5] = [
            ("f(&x)", SHARED, SHARED, |w| w.text(U8), Some(&[])),
            ("f(&mut x)", EXCLUSIVE, SHARED, |w| w.text(U8), Some(&[Coercion::Reborrow])),
            ("f(&[1u8, 2])", SHARED, SHARED, |w| {
                let element = w.text(U8);
                w.text(Shape::Array(2, element))
            }, Some(&[])),
            ("f(&x), f(&mut T)", SHARED, EXCLUSIVE, |w| w.text(U8), None),
            ("f(&return 0), f(&T)", SHARED, SHARED, |w| w.text(Shape::Never), Some(&[])),
        ];
        for (name, lent_as, given_as, place, expected) in cases {
            let mut world = World::new();
            let place = place(&mut world);
            let (lent, _) = world.lend(place, lent_as);
            let hole = world.undecided();
            let parameter = world.external(Shape::Ref(given_as, hole));
            let (_, at) = world.pass(lent, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished, "{name}");

            match expected {
                Some(coercions) => {
                    world.no_refusals();
                    assert!(world.same(place, hole), "{name}");
                    let expected = if coercions.is_empty() { Adjustments::new() } else { coerced(at, coercions) };
                    assert_eq!(world.outcome.adjustments, expected, "{name}");
                }
                None => refused_at(&world, Cause::NoChange, at),
            }
        }
    }

    /// `f(x)`, with `f(&T)`: a value is no reference, and no conversion reaches a
    /// hole.
    #[test]
    fn what_is_no_reference_does_not_go_to_a_hole_behind_one() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let element = world.external(U8);
        let vector = vec_class(&mut world, element);
        let argument = world.meets(vector);
        let hole = world.undecided();
        let parameter = world.external(Shape::Ref(SHARED, hole));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `take(return 0);`, with `take(&T)`.
    #[test]
    fn never_goes_to_a_hole_behind_a_reference() {
        let mut world = World::new();
        let argument = world.text(Shape::Never);
        let hole = world.undecided();
        let parameter = world.external(Shape::Ref(SHARED, hole));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Never]));
    }

    /// `f(g());`, with `g() -> &U` and `f(&T)`: the two targets are one class at
    /// once, before either is decided, as a hole takes whatever comes.
    #[test]
    fn a_target_not_decided_yet_is_one_class_with_the_hole_at_once() {
        let mut world = World::new();
        let target = world.undecided();
        let argument = world.external(Shape::Ref(SHARED, target));
        let hole = world.undecided();
        let parameter = world.external(Shape::Ref(SHARED, hole));
        world.pass(argument, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(target));
        assert!(world.same(target, hole));
    }

    /// `f(&[1u8, 2]);`, with `f(&str)` and the view `as_str: &[T] -> &str`: the one
    /// stage of coercion goes before the chain, into its parameter alone.
    #[test]
    fn a_coercion_goes_before_a_view_and_parts_connect_through_it() {
        let mut world = World::new();
        let slice = node(Shape::Ref(SHARED, node(Shape::Slice(hole()))));
        world.declare("as_str", slice, node(Shape::Ref(SHARED, node(Shape::Str))));
        let element = world.text(U8);
        let array = world.text(Shape::Array(2, element));
        let (lent, _) = world.lend(array, SHARED);
        let text = world.external(Shape::Str);
        let parameter = world.external(Shape::Ref(SHARED, text));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(
            world.outcome.adjustments,
            BTreeMap::from([(at, conversions(&[Coercion::Unsize], &["as_str"]))])
        );
    }

    /// `f(g());`, with `g() -> &T`, `T` decided by what enters it as `Vec<u8>`,
    /// `f(String)` and the value conversion `to_string: &Vec<U> -> String`. The
    /// target is not decided when the change is first looked at, so it waits for it,
    /// and the conversion goes once it is: refused at once, the outcome would depend
    /// on which is looked at first.
    #[test]
    fn a_reference_whose_target_is_not_decided_yet_waits_for_a_conversion() {
        let mut world = World::new();
        let from = node(Shape::Ref(SHARED, vec_of(Ty::Hole(Name::new("U")))));
        world.declare("to_string", from, node(Shape::String));
        let target = world.undecided();
        let argument = world.external(Shape::Ref(SHARED, target));
        let parameter = world.external(Shape::String);
        let (_, at) = world.pass(argument, parameter);
        let element = world.text(U8);
        let vector = world.text(Shape::Struct(Name::new("Vec"), vec![element]));
        world.enter(vector, target);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["to_string"]))]));
    }

    /// `A -> B`, `B -> A` and `A -> C`: from `A` to `C` goes `a_to_c` alone, as a
    /// chain that comes back to `A` is no second way.
    #[test]
    fn a_chain_round_a_cycle_is_no_second_way() {
        let mut world = World::new();
        let named = |name: &str| node(Shape::Struct(Name::new(name), Vec::new()));
        world.declare("a_to_b", named("A"), named("B"));
        world.declare("b_to_a", named("B"), named("A"));
        world.declare("a_to_c", named("A"), named("C"));
        let argument = world.text(Shape::Struct(Name::new("A"), Vec::new()));
        let parameter = world.external(Shape::Struct(Name::new("C"), Vec::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, BTreeMap::from([(at, conversions(&[], &["a_to_c"]))]));
    }

    /// `let r = &x; f(&r);`, the reference to a reference `&r` passed to each kind of
    /// parameter. The argument's class, the item of its boundary, and `r` and `x`.
    fn reference_to_reference(
        world: &mut World,
        of_outer: Mutability,
        of_inner: Mutability,
        parameter: ClassId,
    ) -> (ItemId, ClassId, ClassId) {
        let x = world.text(U8);
        let (lent_x, _) = world.lend(x, of_inner);
        let r = world.meets(lent_x);
        let (lent_r, _) = world.lend(r, of_outer);
        let (_, at) = world.pass(lent_r, parameter);
        (at, r, x)
    }

    fn read_through(at: ItemId, times: usize) -> Adjustments {
        Adjustments::from([(at, vec![Adjustment::Deref; times])])
    }

    /// `f(&r)` with `f(&T)` and `r: &u8`: `&&u8` matches `&T` as it is written, so
    /// `T` is `&u8` and no reference is read through.
    #[test]
    fn a_hole_at_a_target_takes_a_reference_as_it_is() {
        let mut world = World::new();
        let hole = world.undecided();
        let parameter = world.external(Shape::Ref(SHARED, hole));
        let (_, r, _) = reference_to_reference(&mut world, SHARED, SHARED, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert!(world.same(hole, r));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `f(&r)` with `f(&u8)` and `r: &u8`: `&&u8` goes to `&u8`, and so does
    /// `&mut &u8`, as the inner reference is copied out.
    #[test]
    fn a_reference_to_a_shared_reference_is_read_through() {
        for of_outer in [SHARED, EXCLUSIVE] {
            let mut world = World::new();
            let declared = world.external(U8);
            let parameter = world.external(Shape::Ref(SHARED, declared));
            let (at, r, x) = reference_to_reference(&mut world, of_outer, SHARED, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            world.no_refusals();
            assert_eq!(world.outcome.adjustments, read_through(at, 1));
            assert_eq!(world.shape(x), U8);
            assert_eq!(world.target(r).0, SHARED);
        }
    }

    /// `f(&r)` with `f(&u8)` and `r: &mut u8`: an exclusive reference is not copied
    /// out, so it is not read through.
    #[test]
    fn a_reference_to_an_exclusive_reference_is_not_read_through() {
        for of_outer in [SHARED, EXCLUSIVE] {
            let mut world = World::new();
            let declared = world.external(U8);
            let parameter = world.external(Shape::Ref(SHARED, declared));
            let (at, ..) = reference_to_reference(&mut world, of_outer, EXCLUSIVE, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            refused_at(&world, Cause::NoChange, at);
        }
    }

    /// `let r = &x; let q = &r; f(&q);` with `f(&u8)`: every reference above the
    /// last is read through in the one step.
    #[test]
    fn each_reference_above_the_last_is_read_through() {
        let mut world = World::new();
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let x = world.text(U8);
        let (lent_x, _) = world.lend(x, SHARED);
        let r = world.meets(lent_x);
        let (lent_r, _) = world.lend(r, SHARED);
        let q = world.meets(lent_r);
        let (lent_q, _) = world.lend(q, SHARED);
        let (_, at) = world.pass(lent_q, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, read_through(at, 2));
    }

    /// `let r = &[1u8, 2]; f(&r);` with `f(&[u8])`: the reference is read through
    /// and the array unsized in the one step, and the element below is one type.
    #[test]
    fn a_reference_is_read_through_and_its_array_unsized_together() {
        let mut world = World::new();
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        let element = world.text(U8);
        let array = world.text(Shape::Array(2, element));
        let (lent_array, _) = world.lend(array, SHARED);
        let r = world.meets(lent_array);
        let (lent_r, _) = world.lend(r, SHARED);
        let (_, at) = world.pass(lent_r, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        let expected = Adjustments::from([(at, vec![Adjustment::Deref, Adjustment::Coerce(Coercion::Unsize)])]);
        assert_eq!(world.outcome.adjustments, expected);
    }

    /// `if c { &r } else { &x }` with `r: &u8` and `x: u8`: no reference is read
    /// through where values meet, so `&&u8` and `&u8` do not join.
    #[test]
    fn no_reference_is_read_through_where_values_meet() {
        let mut world = World::new();
        let x = world.text(U8);
        let (lent_x, _) = world.lend(x, SHARED);
        let r = world.meets(lent_x);
        let (lent_r, _) = world.lend(r, SHARED);
        let y = world.text(U8);
        let (lent_y, _) = world.lend(y, SHARED);
        let place = world.meets(lent_r);
        world.enter(lent_y, place);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(world.outcome.refusals.iter().any(|refusal| refusal.cause == Cause::Mismatch));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `let r = &v; sum(&r);` with `sum(&[u8])` and `as_slice`: the reference is
    /// read through on the way into the conversion, as a reborrow is.
    #[test]
    fn a_reference_is_read_through_on_the_way_into_a_conversion() {
        let mut world = World::new();
        declare_as_slice(&mut world);
        let declared = world.external(U8);
        let parameter = reference_to_slice(&mut world, SHARED, declared);
        let element = world.text(U8);
        let v = vec_class(&mut world, element);
        let (lent_v, _) = world.lend(v, SHARED);
        let r = world.meets(lent_v);
        let (lent_r, _) = world.lend(r, SHARED);
        let (_, at) = world.pass(lent_r, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        let mut expected = vec![Adjustment::Deref];
        expected.extend(conversions(&[], &["as_slice"]));
        assert_eq!(world.outcome.adjustments, Adjustments::from([(at, expected)]));
    }
}
