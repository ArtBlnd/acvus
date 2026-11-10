//! Settling: the order decisions are taken in.
//!
//! A head is decided once, so a class waits until nothing more can enter it, and
//! each rung below decides what the rungs above it cannot. Whichever rung changes
//! anything, solving returns to the top.
//!
//! | Rung | What it does | Decides a head |
//! |---|---|---|
//! | 0. structure | connects parts, and merges cycles, to a fixpoint | no |
//! | 1. lower | a class takes the join of what enters it (O6) | yes |
//! | 2. upper | a class nothing enters takes the one head its upper bounds give it (O19) | yes |
//! | 3. domain | a literal's class no upper bound gives a head takes its row's fallback type (O20) | yes |
//! | 4. never | a class nothing says anything about takes `!` | yes |
//! | end | a class still undecided is refused (O21), and so is a class that is a part of itself (O5) | |
//!
//! Rung 0 comes first because deciding the head of a class that a later union
//! would give more to fixes it too early. The two fallbacks are apart because
//! `!` means nothing at all is known: a class left without a head may yet have
//! one given by what a literal falls back to.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::bound::Bound;
use crate::primitive::class::{ClassKind, Side, State};
use crate::primitive::domain::Domain;
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{PartKey, Shape};
use crate::store::classes::Classes;

use super::Outcome;
use super::bound::{Join, decided_by, join_lowers};
use super::change::{Conversions, awaits_instances};
use super::upper::{Given, given_by_uppers};

/// Where solving stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Settled {
    /// No rung changes anything. What is still undecided has been refused.
    Finished,
    /// A function value goes to a function type, whose instances are not recorded
    /// yet (O30). Solving stops here: they may decide classes that a fallback
    /// taken now would decide otherwise.
    AwaitsInstances,
}

/// What is known of a class's head before the class is decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeadKind {
    Reference,
    /// Anything else. A join cannot turn it into a reference: two head classes
    /// that meet are refused, whichever enters first. `!` alone is this too.
    NoReference,
    Unknown,
}

/// Whether what enters `class` makes it a reference, without deciding it. This
/// is all a lend needs to know to find its target.
fn head_kind<K>(classes: &Classes<K>, class: K::Id) -> HeadKind
where
    K: ClassKind<Bound = Bound<K>>,
{
    let mut seen = BTreeSet::new();
    let mut stack = vec![class];
    let mut no_reference = false;
    let mut unknown = false;
    while let Some(class) = stack.pop() {
        let root = classes.find(class);
        if !seen.insert(root.id()) {
            continue;
        }
        if let State::Decided(decided) = &classes.get(root).state {
            match decided.shape {
                Shape::Ref(..) => return HeadKind::Reference,
                Shape::Never => {}
                _ => no_reference = true,
            }
            continue;
        }
        let mut entered = false;
        for (lower, bounds) in classes.lowers(root) {
            for (bound, _) in bounds {
                match bound {
                    Bound::Lend { .. } => return HeadKind::Reference,
                    Bound::Enter { .. } if lower != root => {
                        entered = true;
                        stack.push(lower.id());
                    }
                    Bound::Enter { .. } | Bound::Change { .. } | Bound::Equal { .. } => {}
                }
            }
        }
        unknown |= !entered;
    }
    match (no_reference, unknown) {
        (true, _) | (false, false) => HeadKind::NoReference,
        (false, true) => HeadKind::Unknown,
    }
}

fn every_root<K>(classes: &Classes<K>) -> Vec<K::Id>
where
    K: ClassKind,
{
    classes.roots().map(|root| root.id()).collect()
}

/// The undecided classes `visit` gives, of those no lower bound enters.
fn nothing_enters<K, F>(classes: &Classes<K>, visit: &F) -> Vec<K::Id>
where
    K: ClassKind<Bound = Bound<K>>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    visit(classes)
        .into_iter()
        .filter(|class| !classes.is_decided(*class) && join_lowers(classes, *class) == Join::Nothing)
        .collect()
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Takes every rung there is, `at` being the item a decision here goes by.
    pub fn settle(&mut self, classes: &mut Classes<K>, scopes: &Scopes, conversions: &Conversions, at: ItemId) -> Settled {
        self.settle_visiting(classes, scopes, conversions, at, &every_root)
    }

    /// As [`Outcome::settle`], visiting classes in the order `visit` gives, which
    /// the outcome should not depend on.
    pub(super) fn settle_visiting<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        conversions: &Conversions,
        at: ItemId,
        visit: &F,
    ) -> Settled
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        loop {
            self.settle_lower(classes, scopes, conversions, at, visit);
            if self.instances_owed(classes) {
                return Settled::AwaitsInstances;
            }
            if self.decide_by_uppers(classes, scopes, at, visit)
                || self.fall_back_to_domains(classes, scopes, at, visit)
                || self.fall_back_to_never(classes, scopes, at, visit)
            {
                continue;
            }
            self.refuse_own_parts(classes, visit);
            self.refuse_undecided(classes, visit);
            return Settled::Finished;
        }
    }

    /// Whether a function value waits to go to a function type.
    fn instances_owed(&self, classes: &Classes<K>) -> bool {
        self.bounds.iter().any(|(bound, _)| awaits_instances(classes, bound))
    }

    /// Rung 2: each class nothing enters takes the one head its upper bounds give
    /// it, where there is one, all at once.
    fn decide_by_uppers<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        at: ItemId,
        visit: &F,
    ) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let mut decided = false;
        for (class, given) in given_by_uppers(classes, &self.bounds, nothing_enters(classes, visit)) {
            if let Given::Head { shape, by } = given
                && !classes.is_decided(class)
            {
                self.decide_with_new_parts(classes, scopes, class, &shape, by, at);
                decided = true;
            }
        }
        decided
    }

    /// The classes whose dimension is neither lower nor upper (O17). A class its
    /// upper bounds give heads to is not among them, even where no one of those
    /// heads will do.
    fn may_fall_back<F>(&self, classes: &Classes<K>, visit: &F) -> Vec<K::Id>
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        given_by_uppers(classes, &self.bounds, nothing_enters(classes, visit))
            .into_iter()
            .filter_map(|(class, given)| (given == Given::Nothing).then_some(class))
            .collect()
    }

    /// Rungs 0 and 1, until neither changes anything.
    pub(super) fn settle_lower<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        conversions: &Conversions,
        at: ItemId,
        visit: &F,
    )
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        loop {
            let connected = self.connect_structure(classes, scopes, conversions);
            let decided = self.decide_by_join(classes, scopes, at, visit);
            if !connected && !decided {
                break;
            }
        }
    }

    /// Rung 0: connects what can be connected with no head decided, and makes
    /// each change whose class nodes are decided, to a fixpoint, and says whether it
    /// did anything.
    fn connect_structure(&mut self, classes: &mut Classes<K>, scopes: &Scopes, conversions: &Conversions) -> bool {
        let mut connected = false;
        loop {
            let merged = self.merge_cycles(classes, scopes);
            let carried = self.carry_parts(classes, scopes);
            let lent = self.connect_lends(classes, scopes);
            let changed = self.make_changes(classes, scopes, conversions);
            if !merged && !carried && !lent && !changed {
                return connected;
            }
            connected = true;
        }
    }

    /// Makes each part of a class one type with the part of that name on each
    /// undecided class it enters, as judging the bound would once both have
    /// heads. It stops at a decided class, which nothing changes: judging the
    /// bound connects the two there.
    fn carry_parts(&mut self, classes: &mut Classes<K>, scopes: &Scopes) -> bool {
        let mut entering = Vec::new();
        for root in classes.roots() {
            for (bound, at) in classes.get(root).state.bounds().into_iter().flatten() {
                if let Bound::Enter { lower, upper } = bound {
                    entering.push((*lower, *upper, *at));
                }
            }
        }
        let mut carried = false;
        for (lower, upper, at) in entering {
            if classes.is_decided(upper) {
                continue;
            }
            let parts: Vec<(PartKey, K::Id)> = match &classes.get(classes.find(lower)).state {
                State::Decided(decided) => decided.shape.places().into_iter().map(|(key, part)| (key, *part)).collect(),
                State::Undecided(constraints) => constraints
                    .parts
                    .iter()
                    .map(|(key, part)| (key.clone(), part.class))
                    .collect(),
            };
            for (key, part) in parts {
                carried |= self.make_one(classes, scopes, upper, key, part, at);
            }
        }
        self.judge(classes, scopes);
        carried
    }

    /// Finds the target of each lend whose place is known to be a reference or
    /// not, as a part of the reference it gives. Its mutability and its head wait
    /// for the place to be decided, and the bound is judged then.
    fn connect_lends(&mut self, classes: &mut Classes<K>, scopes: &Scopes) -> bool {
        let mut lends = Vec::new();
        for root in classes.roots() {
            for (bound, at) in classes.get(root).state.bounds().into_iter().flatten() {
                if let Bound::Lend { lower, upper, .. } = bound {
                    lends.push((*lower, *upper, *at));
                }
            }
        }
        let mut connected = false;
        for (place, reference, at) in lends {
            if classes.is_decided(reference) {
                continue;
            }
            let target = match head_kind(classes, place) {
                HeadKind::Unknown => continue,
                HeadKind::NoReference => Some(place),
                HeadKind::Reference => classes.part(place, PartKey::Target, Side::Carried, at),
            };
            if let Some(target) = target {
                connected |= self.make_one(classes, scopes, reference, PartKey::Target, target, at);
            }
        }
        self.judge(classes, scopes);
        connected
    }

    /// Makes `part` one class with the part `key` names on `class`, and says
    /// whether they were not one before.
    fn make_one(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        class: K::Id,
        key: PartKey,
        part: K::Id,
        at: ItemId,
    ) -> bool {
        let existing = classes.get(classes.find(class)).state.part(&key);
        if existing.is_some_and(|existing| classes.find(existing) == classes.find(part)) {
            return false;
        }
        let Some(named) = classes.part(class, key, Side::Carried, at) else {
            return false;
        };
        let unified = classes.unify(scopes, named, part, at);
        self.release(classes, scopes, unified);
        true
    }

    /// Rung 1: decides every class all that enters it has given a head to. They
    /// are found first and decided together, so that deciding one does not let
    /// another in ahead of what would connect it.
    fn decide_by_join<F>(&mut self, classes: &mut Classes<K>, scopes: &Scopes, at: ItemId, visit: &F) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let mut ready = Vec::new();
        for class in visit(classes) {
            if classes.is_decided(class) {
                continue;
            }
            match join_lowers(classes, class) {
                Join::Head { .. } => ready.push(class),
                Join::Mismatch(refusal) => self.refuse_once(refusal),
                Join::Nothing | Join::Waits => {}
            }
        }
        let mut decided = false;
        for class in ready {
            if classes.is_decided(class) {
                continue;
            }
            if let Join::Head { shape, by } = join_lowers(classes, class) {
                self.decide(classes, scopes, class, shape, by, at);
                decided = true;
            }
        }
        decided
    }

    /// Rung 3: each class a literal has registered a domain in, and that neither
    /// lower nor upper bounds decide, takes its row's fallback type, all at once.
    fn fall_back_to_domains<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        at: ItemId,
        visit: &F,
    ) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let mut falling = Vec::new();
        for class in self.may_fall_back(classes, visit) {
            let State::Undecided(constraints) = &classes.get(classes.find(class)).state else {
                continue;
            };
            let Some(domain) = Domain::intersection(constraints.domains.iter().map(|(domain, _)| domain)) else {
                continue;
            };
            let by: BTreeSet<ItemId> = constraints.domains.iter().map(|(_, by)| *by).collect();
            if domain.0.is_empty() {
                self.refuse_once(Refusal {
                    cause: Cause::EmptyDomain,
                    items: by,
                });
            } else if let Some(family) = domain.family() {
                falling.push((class, Shape::Scalar(family.fallback()), by));
            }
        }
        let mut fell = false;
        for (class, shape, by) in falling {
            if !classes.is_decided(class) {
                self.decide(classes, scopes, class, shape, by, at);
                fell = true;
            }
        }
        fell
    }

    /// Rung 4: each class no literal has registered a domain in, and that neither
    /// lower nor upper bounds decide, takes `!`, all at once.
    fn fall_back_to_never<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        at: ItemId,
        visit: &F,
    ) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let falling: Vec<K::Id> = self
            .may_fall_back(classes, visit)
            .into_iter()
            .filter(|class| {
                matches!(
                    &classes.get(classes.find(*class)).state,
                    State::Undecided(constraints) if constraints.domains.is_empty()
                )
            })
            .collect();
        let mut fell = false;
        for class in falling {
            if !classes.is_decided(class) {
                self.decide(classes, scopes, class, Shape::Never, BTreeSet::new(), at);
                fell = true;
            }
        }
        fell
    }

    /// The end: refuses each class that is a part of itself (O5), naming what
    /// wrote or decided it and what named its parts. The union that closed the
    /// cycle is not named, as which union closes it depends on the order unions
    /// are made in.
    fn refuse_own_parts<F>(&mut self, classes: &Classes<K>, visit: &F)
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        for class in visit(classes) {
            if !classes.is_own_part(class) {
                continue;
            }
            let mut items = decided_by(classes, class);
            if let State::Undecided(constraints) = &classes.get(classes.find(class)).state {
                let named = constraints
                    .parts
                    .values()
                    .flat_map(|part| part.lower.iter().chain(&part.upper).chain(&part.carried).copied());
                items.extend(named);
            }
            self.refuse_once(Refusal {
                cause: Cause::Occurs,
                items,
            });
        }
    }

    /// The end: refuses each class still undecided (O21), naming what was
    /// recorded on it.
    fn refuse_undecided<F>(&mut self, classes: &Classes<K>, visit: &F)
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        for class in visit(classes) {
            let State::Undecided(constraints) = &classes.get(classes.find(class)).state else {
                continue;
            };
            let bounds = constraints.bounds.iter().map(|(_, at)| *at);
            let parts = constraints
                .parts
                .values()
                .flat_map(|part| part.lower.iter().chain(&part.upper).chain(&part.carried).copied());
            let domains = constraints.domains.iter().map(|(_, at)| *at);
            self.refuse_once(Refusal {
                cause: Cause::UndecidedClass,
                items: bounds.chain(parts).chain(domains).collect(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::operation::bound::shape_of;
    use crate::operation::testing::{EXCLUSIVE, SHARED, U8, World};
    use crate::primitive::adjustment::{Adjustment, Adjustments};
    use crate::primitive::id::Arena;
    use crate::primitive::record::Coercion;
    use crate::primitive::ty::{Mutability, Scalar, Ty};
    use crate::primitive::{ClassId, ClosureId, Name};

    fn describe(world: &World, class: ClassId, depth: usize) -> String {
        if depth == 0 {
            return "…".to_owned();
        }
        let Some(shape) = shape_of(&world.classes, class) else {
            return "?".to_owned();
        };
        match shape {
            Shape::Option(payload) => format!("Option<{}>", describe(world, *payload, depth - 1)),
            Shape::Slice(elem) => format!("[{}]", describe(world, *elem, depth - 1)),
            Shape::Array(len, elem) => format!("[{};{len}]", describe(world, *elem, depth - 1)),
            Shape::Struct(name, args) => {
                let args: Vec<String> = args.iter().map(|arg| describe(world, *arg, depth - 1)).collect();
                format!("{}<{}>", name.0, args.join(","))
            }
            Shape::Ref(SHARED, target) => format!("&{}", describe(world, *target, depth - 1)),
            Shape::Ref(EXCLUSIVE, target) => format!("&mut {}", describe(world, *target, depth - 1)),
            Shape::Object(fields) => {
                let names: Vec<String> = fields.keys().map(|name| name.0.clone()).collect();
                format!("{{{}}}", names.join(","))
            }
            Shape::Scalar(scalar) => format!("{scalar:?}").to_lowercase(),
            Shape::Never => "!".to_owned(),
            other => format!("{:?}", other.layer()),
        }
    }

    /// The type of `class`, or what refused the program.
    fn summary(world: &World, class: ClassId) -> String {
        let mut causes: Vec<String> = world.outcome.refusals.iter().map(|refusal| format!("{:?}", refusal.cause)).collect();
        causes.sort();
        causes.dedup();
        if causes.is_empty() {
            describe(world, class, 4)
        } else {
            format!("refused:{}", causes.join("+"))
        }
    }

    fn some_of(world: &mut World, payload: ClassId) -> ClassId {
        world.text(Shape::Option(payload))
    }

    fn meet(world: &mut World, first: ClassId, second: ClassId) -> ClassId {
        let place = world.meets(first);
        world.enter(second, place);
        place
    }

    fn object_of(world: &mut World, name: &str) -> ClassId {
        let field = world.text(U8);
        world.text(Shape::Object(BTreeMap::from([(Name::new(name), field)])))
    }

    fn field(name: &str) -> PartKey {
        PartKey::Field(Name::new(name))
    }

    fn project(world: &mut World, operand: ClassId, key: PartKey, side: Side) -> (ClassId, ItemId) {
        let at = world.items.push("projection");
        let part = world
            .outcome
            .project(&mut world.classes, operand, key, side, at)
            .expect("an undecided class names any part");
        (part, at)
    }

    /// A literal with no suffix: a class with a domain and no lower bound.
    fn literal(world: &mut World, domain: &[Scalar]) -> ClassId {
        let class = world.undecided();
        let at = world.items.push("literal");
        let registered = world.classes.domain(class, Domain(domain.iter().copied().collect()), at);
        assert_eq!(registered, None);
        class
    }

    const INTEGERS: [Scalar; 9] = [
        Scalar::I8,
        Scalar::I16,
        Scalar::I32,
        Scalar::I64,
        Scalar::U8,
        Scalar::U16,
        Scalar::U32,
        Scalar::U64,
        Scalar::Usize,
    ];
    const FLOATS: [Scalar; 2] = [Scalar::F32, Scalar::F64];

    /// The visit order `order` names: as written, reversed, and shuffled.
    fn reorder(open: &mut [ClassId], order: u64) {
        match order {
            0 => {}
            1 => open.reverse(),
            seed => {
                let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                for i in (1..open.len()).rev() {
                    state = state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                    let j = usize::try_from(state >> 33).expect("a shifted word fits") % (i + 1);
                    open.swap(i, j);
                }
            }
        }
    }

    /// The same outcome in every visit order.
    fn in_every_order<F>(build: F, expected: &str, name: &str)
    where
        F: Fn(&mut World) -> ClassId,
    {
        for order in 0..24u64 {
            let mut world = World::new();
            let observed = build(&mut world);
            reorder(&mut world.open, order);
            assert_eq!(world.settle_fully(), Settled::Finished, "{name}, order {order}");
            assert_eq!(summary(&world, observed), expected, "{name}, order {order}");
        }
    }

    #[test]
    fn what_meets_comes_out_the_same_in_every_order() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 7] = [
            ("if c { Some(return) } else { Some(2u8) }", "Option<u8>", |w| {
                let (r, t) = (w.text(Shape::Never), w.text(U8));
                let (a, b) = (some_of(w, r), some_of(w, t));
                meet(w, a, b)
            }),
            ("if c { Some({x}) } else { Some({y}) }", "Option<{x,y}>", |w| {
                let (x, y) = (object_of(w, "x"), object_of(w, "y"));
                let (a, b) = (some_of(w, x), some_of(w, y));
                meet(w, a, b)
            }),
            ("if c { Some(&mut a) } else { Some(&b) }", "Option<&u8>", |w| {
                let (a, b) = (w.text(U8), w.text(U8));
                let (lent_a, lent_b) = (w.lend(a, EXCLUSIVE).0, w.lend(b, SHARED).0);
                let (some_a, some_b) = (some_of(w, lent_a), some_of(w, lent_b));
                meet(w, some_a, some_b)
            }),
            ("if c { Some(Some(return)) } else { Some(Some(2u8)) }", "Option<Option<u8>>", |w| {
                let (r, t) = (w.text(Shape::Never), w.text(U8));
                let (inner_a, inner_b) = (some_of(w, r), some_of(w, t));
                let (a, b) = (some_of(w, inner_a), some_of(w, inner_b));
                meet(w, a, b)
            }),
            ("let v = Some(return); if c { v } else { Some(2u8) }", "Option<u8>", |w| {
                let (r, t) = (w.text(Shape::Never), w.text(U8));
                let a = some_of(w, r);
                let v = w.meets(a);
                let b = some_of(w, t);
                meet(w, v, b)
            }),
            ("q.x = {a}; q = {x: {b}}", "{a,b}", |w| {
                let q = w.undecided();
                let (part, _) = project(w, q, field("x"), Side::Lower);
                let a = object_of(w, "a");
                w.enter(a, part);
                let b = object_of(w, "b");
                let written = w.text(Shape::Object(BTreeMap::from([(Name::new("x"), b)])));
                w.enter(written, q);
                part
            }),
            ("x = if c { x } else { Some(return) }; x = if c { x } else { Some(2u8) }", "Option<u8>", |w| {
                let x = w.undecided();
                let (r, t) = (w.text(Shape::Never), w.text(U8));
                let (a, b) = (some_of(w, r), some_of(w, t));
                let first = meet(w, x, a);
                w.enter(first, x);
                let second = meet(w, x, b);
                w.enter(second, x);
                x
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    #[test]
    fn what_a_lend_targets_meets_as_any_part_does_in_every_order() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 5] = [
            ("if c { &mut x } else { &y }", "&u8", |w| {
                let (x, y) = (w.text(U8), w.text(U8));
                let (a, b) = (w.lend(x, EXCLUSIVE).0, w.lend(y, SHARED).0);
                meet(w, a, b)
            }),
            ("if c { &{x} } else { &{y} }", "&{x,y}", |w| {
                let (x, y) = (object_of(w, "x"), object_of(w, "y"));
                let (a, b) = (w.lend(x, SHARED).0, w.lend(y, SHARED).0);
                meet(w, a, b)
            }),
            ("let a = return; if c { &a } else { &{y} }", "&{y}", |w| {
                let a = w.text(Shape::Never);
                let y = object_of(w, "y");
                let (lent_a, lent_y) = (w.lend(a, SHARED).0, w.lend(y, SHARED).0);
                meet(w, lent_a, lent_y)
            }),
            ("let p = &mut {x}; let q = &{y}; if c { &p } else { &q }", "&{x,y}", |w| {
                let (x, y) = (object_of(w, "x"), object_of(w, "y"));
                let lent_x = w.lend(x, EXCLUSIVE).0;
                let p = w.meets(lent_x);
                let lent_y = w.lend(y, SHARED).0;
                let q = w.meets(lent_y);
                let (lent_p, lent_q) = (w.lend(p, SHARED).0, w.lend(q, SHARED).0);
                meet(w, lent_p, lent_q)
            }),
            ("if c { &{x: 1u8} } else { &1u8 }", "refused:Mismatch+UndecidedClass", |w| {
                let (x, y) = (object_of(w, "x"), w.text(U8));
                let (a, b) = (w.lend(x, SHARED).0, w.lend(y, SHARED).0);
                meet(w, a, b)
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    #[test]
    fn what_the_examples_and_declarations_give_comes_out_the_same_in_every_order() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 5] = [
            ("example 2: if c { Some(1u8) } else { None }", "Option<u8>", |w| {
                let one = w.text(U8);
                let nothing = w.undecided();
                let (a, b) = (some_of(w, one), some_of(w, nothing));
                meet(w, a, b)
            }),
            ("example 24: if c { return 0u8 } else { 1u8 }", "u8", |w| {
                let (r, one) = (w.text(Shape::Never), w.text(U8));
                meet(w, r, one)
            }),
            ("if c { Some(1u8) } else { f() }, f() -> Option<u8>", "Option<u8>", |w| {
                let declared = w.external(U8);
                let result = w.external(Shape::Option(declared));
                let one = w.text(U8);
                let a = some_of(w, one);
                meet(w, a, result)
            }),
            ("if c { Some(1u16) } else { f() }, f() -> Option<u8>", "refused:Mismatch", |w| {
                let declared = w.external(U8);
                let result = w.external(Shape::Option(declared));
                let one = w.text(Shape::Scalar(Scalar::U16));
                let a = some_of(w, one);
                meet(w, a, result)
            }),
            ("if c { f() } else { Some(1u16) }, f() -> Option<u8>", "refused:Mismatch", |w| {
                let declared = w.external(U8);
                let result = w.external(Shape::Option(declared));
                let one = w.text(Shape::Scalar(Scalar::U16));
                let a = some_of(w, one);
                meet(w, result, a)
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    /// `x = Some(x);`. It is refused while its parts are connected, before any
    /// head is decided.
    #[test]
    fn a_class_that_holds_itself_is_refused_in_every_order() {
        in_every_order(
            |w| {
                let x = w.undecided();
                let some = some_of(w, x);
                w.enter(some, x);
                x
            },
            "refused:Occurs",
            "x = Some(x)",
        );
    }

    #[test]
    fn a_literal_nothing_else_speaks_of_falls_back_by_its_row() {
        let mut world = World::new();
        let integer = literal(&mut world, &INTEGERS);
        let float = literal(&mut world, &FLOATS);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(integer), Shape::Scalar(Scalar::I32));
        assert_eq!(world.shape(float), Shape::Scalar(Scalar::F64));
    }

    /// Example 11: `let x = None;`
    #[test]
    fn a_class_nothing_speaks_of_falls_back_to_never() {
        let mut world = World::new();
        let payload = world.undecided();
        let x = world.text(Shape::Option(payload));

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(x), Shape::Option(payload));
        assert_eq!(world.shape(payload), Shape::Never);
    }

    /// Example 22: `f(0, 1);`, with `f(T, T)`. `T` gives its literals no head, as
    /// nothing has entered it, and what enters it decides it after they fall back.
    #[test]
    fn literals_passed_to_a_hole_fall_back_and_decide_it() {
        let mut world = World::new();
        let hole = world.undecided();
        let (zero, one) = (literal(&mut world, &INTEGERS), literal(&mut world, &INTEGERS));
        world.enter(zero, hole);
        world.enter(one, hole);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(hole), Shape::Scalar(Scalar::I32));
    }

    /// Example 10: `let a; a.x = 0;`
    #[test]
    fn a_projection_on_what_falls_back_to_never_is_refused() {
        let mut world = World::new();
        let a = world.undecided();
        let (part, named_at) = project(&mut world, a, field("x"), Side::Lower);
        let zero = literal(&mut world, &INTEGERS);
        world.enter(zero, part);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert_eq!(world.shape(a), Shape::Never);
        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::NotHeld);
        assert!(world.outcome.refusals[0].items.contains(&named_at));
    }

    /// `let y; let z = if c { y } else { 0 };`: `!` waits for what the literal
    /// falls back to, which `z` then gives `y`. All at once, `y` would be `!`,
    /// though what it holds has a type.
    #[test]
    fn never_waits_for_what_a_literal_falls_back_to() {
        let mut world = World::new();
        let y = world.undecided();
        let zero = literal(&mut world, &INTEGERS);
        let z = meet(&mut world, y, zero);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(zero), Shape::Scalar(Scalar::I32));
        assert_eq!(world.shape(y), Shape::Scalar(Scalar::I32));
        assert_eq!(world.shape(z), Shape::Scalar(Scalar::I32));
    }

    fn reference_to_slice(world: &mut World, mutability: Mutability) -> ClassId {
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        world.external(Shape::Ref(mutability, slice))
    }

    #[test]
    fn what_upper_bounds_give_comes_out_the_same_in_every_order() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 26] = [
            ("example 1: let a = 1u32; let b = if c { a } else { 10 };", "u32", |w| {
                let written = w.text(Shape::Scalar(Scalar::U32));
                let a = w.meets(written);
                let ten = literal(w, &INTEGERS);
                let if_class = meet(w, a, ten);
                w.meets(if_class);
                ten
            }),
            ("example 9: let x = 0; take(x);, take(u8)", "u8", |w| {
                let zero = literal(w, &INTEGERS);
                let x = w.meets(zero);
                let parameter = w.external(U8);
                w.pass(x, parameter);
                zero
            }),
            ("example 21: f(0, 1u8);, f(T, T)", "u8", |w| {
                let hole = w.undecided();
                let zero = literal(w, &INTEGERS);
                let one = w.text(U8);
                w.enter(zero, hole);
                w.enter(one, hole);
                zero
            }),
            ("example 23: let a; if let Some(v) = a {} take(a);, take(Option<u8>)", "u8", |w| {
                let a = w.undecided();
                let (v, _) = project(w, a, PartKey::Payload(Name::new("Some")), Side::Upper);
                let declared = w.external(U8);
                let parameter = w.external(Shape::Option(declared));
                w.pass(a, parameter);
                v
            }),
            ("let y; let z = if c { y } else { 0 };", "i32", |w| {
                let y = w.undecided();
                let zero = literal(w, &INTEGERS);
                meet(w, y, zero);
                y
            }),
            ("let y; let x = 0; let z = if c { y } else { x };", "i32", |w| {
                let y = w.undecided();
                let zero = literal(w, &INTEGERS);
                let x = w.meets(zero);
                meet(w, y, x);
                y
            }),
            ("let y; h(y); if c { y } else { return }, h(&T)", "&!", |w| {
                let y = w.undecided();
                let hole = w.undecided();
                let parameter = w.external(Shape::Ref(SHARED, hole));
                w.pass(y, parameter);
                let returns = w.text(Shape::Never);
                meet(w, y, returns);
                y
            }),
            ("let y; f(y); g(y);, f(&T), g(&mut U)", "&mut !", |w| {
                let y = w.undecided();
                let (t, u) = (w.undecided(), w.undecided());
                let (shared, exclusive) = (w.external(Shape::Ref(SHARED, t)), w.external(Shape::Ref(EXCLUSIVE, u)));
                w.pass(y, shared);
                w.pass(y, exclusive);
                y
            }),
            ("let v = new(); let a = first(&v); let z = if c { a } else { 0 };", "i32", |w| {
                let new_hole = w.undecided();
                let result = w.external(Shape::Option(new_hole));
                let v = w.meets(result);
                let (lent, _) = w.lend(v, SHARED);
                let first_hole = w.undecided();
                let option = w.external(Shape::Option(first_hole));
                let parameter = w.external(Shape::Ref(SHARED, option));
                w.pass(lent, parameter);
                let a = w.meets(first_hole);
                let zero = literal(w, &INTEGERS);
                meet(w, a, zero);
                new_hole
            }),
            ("let a; g(&a);, g(&[u8])", "[u8]", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, SHARED);
                let parameter = reference_to_slice(w, SHARED);
                w.pass(lent, parameter);
                a
            }),
            ("let a; let r = &a; let s = r; g(s);, g(&[u8])", "[u8]", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, SHARED);
                let r = w.meets(lent);
                let s = w.meets(r);
                let parameter = reference_to_slice(w, SHARED);
                w.pass(s, parameter);
                a
            }),
            ("let a; g(&mut a);, g(&[u8])", "[u8]", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, EXCLUSIVE);
                let parameter = reference_to_slice(w, SHARED);
                w.pass(lent, parameter);
                a
            }),
            ("let a; g(&mut a);, g(&mut [u8])", "[u8]", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, EXCLUSIVE);
                let parameter = reference_to_slice(w, EXCLUSIVE);
                w.pass(lent, parameter);
                a
            }),
            ("let a; g(&a);, g(&mut [u8])", "refused:UndecidedClass", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, SHARED);
                let parameter = reference_to_slice(w, EXCLUSIVE);
                w.pass(lent, parameter);
                a
            }),
            ("example 8: let a; g(&a); k(&a);, g(&[u8]), k(&str)", "refused:UndecidedClass", |w| {
                let a = w.undecided();
                let (first, _) = w.lend(a, SHARED);
                let parameter = reference_to_slice(w, SHARED);
                w.pass(first, parameter);
                let (second, _) = w.lend(a, SHARED);
                let text = w.external(Shape::Str);
                let parameter = w.external(Shape::Ref(SHARED, text));
                w.pass(second, parameter);
                a
            }),
            ("let x = 0; f(x); g(x);, f(i32), g(u8)", "refused:UndecidedClass", |w| {
                let zero = literal(w, &INTEGERS);
                let x = w.meets(zero);
                let (first, second) = (w.external(Shape::Scalar(Scalar::I32)), w.external(U8));
                w.pass(x, first);
                w.pass(x, second);
                zero
            }),
            ("f(0);, f(String)", "refused:UndecidedClass", |w| {
                let zero = literal(w, &INTEGERS);
                let parameter = w.external(Shape::String);
                w.pass(zero, parameter);
                zero
            }),
            ("let a; f(a); g(a);, f({x}), g({x, y})", "{x,y}", |w| {
                let a = w.undecided();
                let declared = w.external(U8);
                let x = BTreeMap::from([(Name::new("x"), declared)]);
                let x_and_y = BTreeMap::from([(Name::new("x"), declared), (Name::new("y"), declared)]);
                let (first, second) = (w.external(Shape::Object(x)), w.external(Shape::Object(x_and_y)));
                w.pass(a, first);
                w.pass(a, second);
                a
            }),
            ("let p; let v = p.x; take(p);, take(&{x: u8})", "u8", |w| {
                let p = w.undecided();
                let (v, _) = project(w, p, field("x"), Side::Upper);
                let declared = w.external(U8);
                let target = w.external(Shape::Object(BTreeMap::from([(Name::new("x"), declared)])));
                let parameter = w.external(Shape::Ref(SHARED, target));
                w.pass(p, parameter);
                v
            }),
            ("let a; h(Some(&a));, h(Option<&u8>)", "u8", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, SHARED);
                let some = some_of(w, lent);
                let declared = w.external(U8);
                let reference = w.external(Shape::Ref(SHARED, declared));
                let parameter = w.external(Shape::Option(reference));
                w.pass(some, parameter);
                a
            }),
            ("let a; h(Some(&a));, h(Option<&mut u8>)", "refused:UndecidedClass", |w| {
                let a = w.undecided();
                let (lent, _) = w.lend(a, SHARED);
                let some = some_of(w, lent);
                let declared = w.external(U8);
                let reference = w.external(Shape::Ref(EXCLUSIVE, declared));
                let parameter = w.external(Shape::Option(reference));
                w.pass(some, parameter);
                a
            }),
            ("let a; if c { &a } else { &1u8 }", "u8", |w| {
                let a = w.undecided();
                let one = w.text(U8);
                let (first, second) = (w.lend(a, SHARED).0, w.lend(one, SHARED).0);
                meet(w, first, second);
                a
            }),
            ("let b; take(b); let z = if c { b } else { 0 };, take(u8)", "u8", |w| {
                let b = w.undecided();
                let parameter = w.external(U8);
                w.pass(b, parameter);
                let zero = literal(w, &INTEGERS);
                meet(w, b, zero);
                zero
            }),
            ("let y; take(y); if c { y } else { 1u16 }, take(u8)", "refused:UndecidedClass", |w| {
                let y = w.undecided();
                let parameter = w.external(U8);
                w.pass(y, parameter);
                let sixteen = w.text(Shape::Scalar(Scalar::U16));
                meet(w, y, sixteen);
                y
            }),
            ("let p; h(Some(p)); k(p);, h(Option<u8>), k(u16)", "refused:UndecidedClass", |w| {
                let p = w.undecided();
                let some = some_of(w, p);
                let declared = w.external(U8);
                let parameter = w.external(Shape::Option(declared));
                w.pass(some, parameter);
                let other = w.external(Shape::Scalar(Scalar::U16));
                w.pass(p, other);
                p
            }),
            ("let a; f(a); if c { a } else { {x, y} }, f({x})", "refused:UndecidedClass", |w| {
                let a = w.undecided();
                let declared = w.external(U8);
                let parameter = w.external(Shape::Object(BTreeMap::from([(Name::new("x"), declared)])));
                w.pass(a, parameter);
                let (x, y) = (w.text(U8), w.text(U8));
                let both = w.text(Shape::Object(BTreeMap::from([(Name::new("x"), x), (Name::new("y"), y)])));
                meet(w, a, both);
                a
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    #[test]
    fn what_a_conversion_connects_comes_out_the_same_in_every_order() {
        fn node(shape: Shape<Ty>) -> Ty {
            Ty::Node(Box::new(shape))
        }
        fn declare(world: &mut World, name: &str, mutability: Mutability) {
            let hole = || Ty::Hole(Name::new("T"));
            let vector = node(Shape::Struct(Name::new("Vec"), vec![hole()]));
            let from = node(Shape::Ref(mutability, vector));
            let to = node(Shape::Ref(mutability, node(Shape::Slice(hole()))));
            world.declare(name, from, to);
        }
        fn vector(world: &mut World, element: ClassId) -> ClassId {
            world.external(Shape::Struct(Name::new("Vec"), vec![element]))
        }
        fn slice_parameter(world: &mut World) -> ClassId {
            let declared = world.external(U8);
            let slice = world.external(Shape::Slice(declared));
            world.external(Shape::Ref(SHARED, slice))
        }
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 3] = [
            ("example 5: let v = new(); sum(&v);", "Vec<u8>", |w| {
                declare(w, "as_slice", SHARED);
                let element = w.undecided();
                let result = vector(w, element);
                let v = w.meets(result);
                let (lent, _) = w.lend(v, SHARED);
                let parameter = slice_parameter(w);
                w.pass(lent, parameter);
                v
            }),
            ("example 7: let a; g(&a); h(&a);", "refused:UndecidedClass", |w| {
                declare(w, "as_slice", SHARED);
                let a = w.undecided();
                let (first, _) = w.lend(a, SHARED);
                let parameter = slice_parameter(w);
                w.pass(first, parameter);
                let (second, _) = w.lend(a, SHARED);
                let declared = w.external(U8);
                let wanted = vector(w, declared);
                let parameter = w.external(Shape::Ref(SHARED, wanted));
                w.pass(second, parameter);
                a
            }),
            ("example 13: let mut v = bytes(); sum(&mut v);", "Vec<u8>", |w| {
                declare(w, "as_slice", SHARED);
                declare(w, "as_slice_mut", EXCLUSIVE);
                let byte = w.external(U8);
                let result = vector(w, byte);
                let v = w.meets(result);
                let (lent, _) = w.lend(v, EXCLUSIVE);
                let parameter = slice_parameter(w);
                w.pass(lent, parameter);
                v
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    /// Example 9: the literal takes what the boundary's type gives, and the classes
    /// it enters take it from it, as joins.
    #[test]
    fn a_literal_takes_what_its_upper_bounds_give_rather_than_falling_back() {
        let mut world = World::new();
        let zero = literal(&mut world, &INTEGERS);
        let x = world.meets(zero);
        let parameter = world.external(U8);
        let (boundary, _) = world.pass(x, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        for class in [zero, x, boundary] {
            assert_eq!(world.shape(class), U8);
        }
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// Example 19's reason, with no overload: two upper bounds give two heads, both
    /// in the literal's domain, and neither goes to the other. The dimension is
    /// upper, and it alone decides, so `i32`, the fallback, does not reach it.
    #[test]
    fn a_class_given_heads_it_cannot_choose_among_does_not_fall_back() {
        let mut world = World::new();
        let zero = literal(&mut world, &INTEGERS);
        let x = world.meets(zero);
        let (first, second) = (world.external(Shape::Scalar(Scalar::I32)), world.external(U8));
        let (_, first_at) = world.pass(x, first);
        let (_, second_at) = world.pass(x, second);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(!world.classes.is_decided(zero));
        let named: BTreeSet<ItemId> = world.outcome.refusals.iter().flat_map(|refusal| refusal.items.clone()).collect();
        assert!(world.outcome.refusals.iter().all(|refusal| refusal.cause == Cause::UndecidedClass));
        assert!(named.contains(&first_at) && named.contains(&second_at));
    }

    /// A class that heads go to takes the one that goes to them all, and the bound
    /// it goes to by a change records it there alone.
    #[test]
    fn the_head_that_goes_to_every_upper_bound_records_its_change_where_it_makes_one() {
        let mut world = World::new();
        let a = world.undecided();
        let declared = world.external(U8);
        let x = BTreeMap::from([(Name::new("x"), declared)]);
        let x_and_y = BTreeMap::from([(Name::new("x"), declared), (Name::new("y"), declared)]);
        let (first, second) = (world.external(Shape::Object(x)), world.external(Shape::Object(x_and_y)));
        let (_, first_at) = world.pass(a, first);
        world.pass(a, second);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        let expected: Adjustments = [(first_at, vec![Adjustment::Coerce(Coercion::DropFields)])].into_iter().collect();
        assert_eq!(world.outcome.adjustments, expected);
    }

    /// `let p; let v = p.x; take(p);`, with `take(&{x: u8})`. The head of `p` comes
    /// from above, and the projection derefs it, wherever that was found out.
    #[test]
    fn a_projection_derefs_a_head_its_operand_was_given_from_above() {
        let mut world = World::new();
        let p = world.undecided();
        let (v, projected) = project(&mut world, p, field("x"), Side::Upper);
        let declared = world.external(U8);
        let target = world.external(Shape::Object(BTreeMap::from([(Name::new("x"), declared)])));
        let parameter = world.external(Shape::Ref(SHARED, target));
        world.pass(p, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(v), U8);
        assert!(!world.same(v, declared));
        let expected: Adjustments = [(projected, vec![Adjustment::Deref])].into_iter().collect();
        assert_eq!(world.outcome.adjustments, expected);
    }

    /// A head taken from above has parts of its own: what the upper bound holds is
    /// what they are connected to, not what they are.
    #[test]
    fn a_head_given_from_above_has_parts_of_its_own() {
        let mut world = World::new();
        let a = world.undecided();
        let declared = world.external(U8);
        let parameter = world.external(Shape::Option(declared));
        world.pass(a, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        let Shape::Option(payload) = world.shape(a) else {
            panic!("`a` takes the head its upper bound gives");
        };
        assert!(!world.same(payload, declared));
        assert_eq!(world.shape(payload), U8);
        assert!(!world.classes.is_external(a));
    }

    /// Its instances could decide classes, so no fallback is taken while they are
    /// owed.
    #[test]
    fn a_function_type_waiting_for_instances_stops_the_fallbacks() {
        let mut closures = Arena::<ClosureId, ()>::default();
        let closure = closures.push(());
        let mut world = World::new();
        let other = world.undecided();
        let argument = world.external(Shape::FnValue(BTreeSet::from([closure])));
        let result = world.external(U8);
        let parameter = world.external(Shape::Fn(Vec::new(), result));
        world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::AwaitsInstances);

        world.no_refusals();
        assert!(!world.classes.is_decided(other));
    }

    /// A change that waits for its argument's whole type is no reason to stop: what
    /// it waits for is decided by a fallback like any other, as the conversion gives
    /// it nothing. `let b; f((1u8, b));`, with `f(P)` and `pair: (T, U) -> P`.
    #[test]
    fn a_change_waiting_for_its_argument_does_not_stop_the_fallbacks() {
        let mut world = World::new();
        let parts = vec![Ty::Hole(Name::new("T")), Ty::Hole(Name::new("U"))];
        world.declare("pair", ty(Shape::Tuple(parts)), ty(Shape::Struct(Name::new("P"), Vec::new())));
        let a = world.text(U8);
        let b = world.undecided();
        let tuple = world.text(Shape::Tuple(vec![a, b]));
        let parameter = world.external(Shape::Struct(Name::new("P"), Vec::new()));
        world.pass(tuple, parameter);

        world.settle();
        assert_eq!(world.outcome.bounds.len(), 1);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(b), Shape::Never);
        assert_eq!(world.outcome.bounds.len(), 0);
    }

    fn ty(shape: Shape<Ty>) -> Ty {
        Ty::Node(Box::new(shape))
    }

    fn reference_to_hole(world: &mut World, mutability: Mutability) -> (ClassId, ClassId) {
        let hole = world.undecided();
        (world.external(Shape::Ref(mutability, hole)), hole)
    }

    /// A hole at a reference's target takes the target that comes, whatever else
    /// decides the hole and in whichever order.
    #[test]
    fn what_a_hole_at_a_target_takes_comes_out_the_same_in_every_order() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 5] = [
            ("let x = 1u8; f(&x);, f(&T)", "u8", |w| {
                let x = w.text(U8);
                let (lent, _) = w.lend(x, SHARED);
                let (parameter, hole) = reference_to_hole(w, SHARED);
                w.pass(lent, parameter);
                hole
            }),
            ("f(&[1u8, 2]);, f(&T)", "[u8;2]", |w| {
                let element = w.text(U8);
                let array = w.text(Shape::Array(2, element));
                let (lent, _) = w.lend(array, SHARED);
                let (parameter, hole) = reference_to_hole(w, SHARED);
                w.pass(lent, parameter);
                hole
            }),
            ("let x = 1u8; let y = id(&x);, id(&T) -> &T", "&u8", |w| {
                let x = w.text(U8);
                let (lent, _) = w.lend(x, SHARED);
                let (parameter, _) = reference_to_hole(w, SHARED);
                w.pass(lent, parameter);
                w.meets(parameter)
            }),
            ("let y; f(y); g(y);, f(&T), g(&[u8])", "&[u8]", |w| {
                let y = w.undecided();
                let (first, _) = reference_to_hole(w, SHARED);
                w.pass(y, first);
                let second = reference_to_slice(w, SHARED);
                w.pass(y, second);
                y
            }),
            ("g(&[1u8, 2], &s);, g(&T, &T), s: [u8]", "refused:Mismatch", |w| {
                let element = w.text(U8);
                let array = w.text(Shape::Array(2, element));
                let (lent_array, _) = w.lend(array, SHARED);
                let other = w.text(U8);
                let slice = w.text(Shape::Slice(other));
                let (lent_slice, _) = w.lend(slice, SHARED);
                let hole = w.undecided();
                let (first, second) = (w.external(Shape::Ref(SHARED, hole)), w.external(Shape::Ref(SHARED, hole)));
                w.pass(lent_array, first);
                w.pass(lent_slice, second);
                hole
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    /// `let mut v = new(); f(&v); push(&mut v, 1u16);`, with `f(&Vec<u8>)` and a
    /// value conversion `widen: &Vec<u16> -> &Vec<u8>`. A coercion goes, so it is
    /// taken whatever the element turns out to be, and the element is refused there
    /// in every order, never converted.
    #[test]
    fn a_coercion_that_goes_is_taken_before_a_conversion_in_every_order() {
        fn vector(element: Shape<Ty>) -> Ty {
            ty(Shape::Ref(SHARED, ty(Shape::Struct(Name::new("Vec"), vec![ty(element)]))))
        }
        in_every_order(
            |w| {
                w.declare("widen", vector(Shape::Scalar(Scalar::U16)), vector(Shape::Scalar(Scalar::U8)));
                let element = w.undecided();
                let result = w.external(Shape::Struct(Name::new("Vec"), vec![element]));
                let v = w.meets(result);
                let (lent, _) = w.lend(v, SHARED);
                let declared = w.external(U8);
                let wanted = w.external(Shape::Struct(Name::new("Vec"), vec![declared]));
                let parameter = w.external(Shape::Ref(SHARED, wanted));
                w.pass(lent, parameter);
                let sixteen = w.text(Shape::Scalar(Scalar::U16));
                w.enter(sixteen, element);
                element
            },
            "refused:Mismatch",
            "a coercion that goes is taken before a conversion",
        );
    }

    /// O8: a domain holds the head decided, whichever rung decided it.
    #[test]
    fn a_head_outside_the_domain_is_refused() {
        // A lower bound decides it.
        let mut world = World::new();
        let class = literal(&mut world, &FLOATS);
        let written = world.external(U8);
        world.enter(written, class);
        assert_eq!(world.settle_fully(), Settled::Finished);
        assert_eq!(summary(&world, class), "refused:EmptyDomain");

        // Its row's fallback type does not hold its value: `3_000_000_000`.
        let mut world = World::new();
        let class = literal(&mut world, &[Scalar::I64, Scalar::U32, Scalar::U64, Scalar::Usize]);
        assert_eq!(world.settle_fully(), Settled::Finished);
        assert_eq!(summary(&world, class), "refused:EmptyDomain");
    }

    #[test]
    fn a_domain_two_literals_leave_empty_is_refused() {
        let mut world = World::new();
        let integer = literal(&mut world, &INTEGERS);
        let float = literal(&mut world, &FLOATS);
        let at = world.items.push("meets");
        let unified = world.classes.unify(&world.scopes, integer, float, at);
        assert_eq!(unified.refusals, Vec::new());

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert_eq!(summary(&world, integer), "refused:EmptyDomain+UndecidedClass");
    }

    #[test]
    fn a_head_inside_the_domain_is_accepted() {
        let mut world = World::new();
        let class = literal(&mut world, &INTEGERS);
        let written = world.external(U8);
        world.enter(written, class);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(class), U8);
        assert_eq!(world.outcome.pending, Vec::new());
    }

    /// `let r = q; let v = q.x;`, `q` then being `&{x}`: `r` is carried the name,
    /// and only the projection that wrote it takes the deref.
    #[test]
    fn a_name_carried_to_a_class_takes_no_adjustment_there() {
        let mut world = World::new();
        let q = world.undecided();
        let (v, projected) = project(&mut world, q, field("x"), Side::Upper);
        let r = world.meets(q);
        let field_class = world.text(U8);
        let object = world.text(Shape::Object(BTreeMap::from([(Name::new("x"), field_class)])));
        let (lent, _) = world.lend(object, SHARED);
        world.enter(lent, q);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert!(world.same(v, field_class));
        assert_eq!(world.target(r).0, SHARED);
        let expected: Adjustments = [(projected, vec![Adjustment::Deref])].into_iter().collect();
        assert_eq!(world.outcome.adjustments, expected);
    }

    /// A name carried to a class whose head has no place for it is refused once,
    /// at the projection that wrote it, and not again at the bound that carried it.
    #[test]
    fn a_name_carried_where_it_has_no_place_is_refused_at_the_projection_alone() {
        let mut world = World::new();
        let q = world.undecided();
        let (_, projected) = project(&mut world, q, field("x"), Side::Upper);
        world.meets(q);
        let one = world.text(U8);
        world.enter(one, q);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::NotHeld);
        assert!(world.outcome.refusals[0].items.contains(&projected));
    }

    /// `let mut x = &y; x = &x;`: it ends refused, as both still wait, and which
    /// way it should go is not settled.
    #[test]
    fn a_class_that_still_waits_when_solving_ends_is_refused() {
        let mut world = World::new();
        let y = world.text(U8);
        let (lent_y, _) = world.lend(y, SHARED);
        let x = world.meets(lent_y);
        let (lent_x, _) = world.lend(x, SHARED);
        world.enter(lent_x, x);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert_eq!(world.shape(y), U8);
        assert!(!world.classes.is_decided(x));
        assert!(world.outcome.refusals.iter().all(|refusal| refusal.cause == Cause::UndecidedClass));
        assert!(!world.outcome.refusals.is_empty());
    }

    /// `let a; let r = &a;`, `a` turning out to be a reference: its target is
    /// `a`'s, so nothing may connect it to `a` before that is known. Here the head
    /// is given by hand, as the upper bounds will give it.
    #[test]
    fn a_lend_of_a_place_not_known_yet_connects_nothing() {
        let mut world = World::new();
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);

        world.settle();
        assert!(!world.classes.is_decided(lent));

        let target = world.text(U8);
        world.decide(a, Shape::Ref(SHARED, target));
        world.settle();

        world.no_refusals();
        assert_eq!(world.target(lent).0, SHARED);
        assert!(world.same(world.target(lent).1, target));
        assert!(!world.same(world.target(lent).1, a));
    }

    /// What enters a literal's class decides it. Its row's fallback is for a class
    /// nothing enters (O17).
    #[test]
    fn a_literal_something_enters_does_not_fall_back() {
        let mut world = World::new();
        let waited = world.undecided();
        let class = literal(&mut world, &INTEGERS);
        world.enter(waited, class);

        world.settle_fully();

        assert_ne!(shape_of(&world.classes, class), Some(&Shape::Scalar(Scalar::I32)));
    }

    /// `Outcome::settle` is what the tests' visiting order stands in for.
    #[test]
    fn settling_visits_every_class_there_is() {
        let mut world = World::new();
        let q = world.undecided();
        let (part, _) = project(&mut world, q, PartKey::Payload(Name::new("Some")), Side::Upper);
        let one = world.text(U8);
        let some = world.text(Shape::Option(one));
        world.enter(some, q);
        assert!(!world.open.contains(&part));
        let at = world.items.push("settling");

        let settled = world.outcome.settle(&mut world.classes, &world.scopes, &world.conversions, at);

        assert_eq!(settled, Settled::Finished);
        world.no_refusals();
        assert_eq!(world.shape(part), U8);
    }

    /// `!` is the least head, so a join that has come to `!` alone gives nothing
    /// from above: what else enters decides, or a fallback does. A class decided as
    /// `!` still gives `!`, which no literal takes.
    #[test]
    fn a_join_of_never_alone_gives_no_head_from_above() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 4] = [
            ("let z = if c { 0 } else { return };", "i32", |w| {
                let zero = literal(w, &INTEGERS);
                let returns = w.text(Shape::Never);
                meet(w, zero, returns)
            }),
            ("f(0, return);, f(T, T)", "i32", |w| {
                let hole = w.undecided();
                let zero = literal(w, &INTEGERS);
                let returns = w.text(Shape::Never);
                w.enter(zero, hole);
                w.enter(returns, hole);
                hole
            }),
            ("let a; if c { a } else { return }; if c { &a } else { &1u8 }", "u8", |w| {
                let a = w.undecided();
                let returns = w.text(Shape::Never);
                meet(w, a, returns);
                let one = w.text(U8);
                let (first, second) = (w.lend(a, SHARED).0, w.lend(one, SHARED).0);
                meet(w, first, second);
                a
            }),
            ("0 entering a declared !", "refused:UndecidedClass", |w| {
                let zero = literal(w, &INTEGERS);
                let never = w.external(Shape::Never);
                w.enter(zero, never);
                zero
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    /// `let a; f(a); if c { &a } else { &{x: 1u8, y: 2u8} }`, with `f({x: u8})`.
    /// At the reference's target `a` is one type with `{x, y}`, so `{x}`, which only
    /// enters it, is no head that goes there, and `{x, y}` is the one that goes to
    /// both.
    #[test]
    fn a_head_at_a_targets_place_is_the_targets_own() {
        in_every_order(
            |w| {
                let a = w.undecided();
                let declared = w.external(U8);
                let parameter = w.external(Shape::Object(BTreeMap::from([(Name::new("x"), declared)])));
                w.pass(a, parameter);
                let (x, y) = (w.text(U8), w.text(U8));
                let both = w.text(Shape::Object(BTreeMap::from([(Name::new("x"), x), (Name::new("y"), y)])));
                let (first, second) = (w.lend(a, SHARED).0, w.lend(both, SHARED).0);
                meet(w, first, second);
                a
            },
            "{x,y}",
            "a head at a target's place",
        );
    }

    /// What has entered an undecided class so far is not its head, which the class
    /// being decided joins into: a head goes there where it joins with what is
    /// there. A decided head is checked as it is.
    #[test]
    fn a_head_goes_into_an_undecided_class_where_it_joins_with_what_is_there() {
        type Program = (&'static str, &'static str, fn(&mut World) -> ClassId);
        let programs: [Program; 4] = [
            ("let a; let d: {x: u8} = a; if c { a } else { {x: 1u8, y: 2u8} }", "{x}", |w| {
                let a = w.undecided();
                let declared = w.external(U8);
                let written = w.external(Shape::Object(BTreeMap::from([(Name::new("x"), declared)])));
                w.enter(a, written);
                let (x, y) = (w.text(U8), w.text(U8));
                let both = w.text(Shape::Object(BTreeMap::from([(Name::new("x"), x), (Name::new("y"), y)])));
                meet(w, a, both);
                a
            }),
            ("let a; if c { &mut x } else { &a }", "u8", |w| {
                let x = w.text(U8);
                let a = w.undecided();
                let (first, second) = (w.lend(x, EXCLUSIVE).0, w.lend(a, SHARED).0);
                meet(w, first, second);
                a
            }),
            ("let a; if c { a } else { {x: 1u8} }; f(a);, f({x: u8, y: u8})", "{x,y}", |w| {
                let a = w.undecided();
                let one = w.text(U8);
                let x = w.text(Shape::Object(BTreeMap::from([(Name::new("x"), one)])));
                meet(w, a, x);
                let declared = w.external(U8);
                let fields = BTreeMap::from([(Name::new("x"), declared), (Name::new("y"), declared)]);
                let parameter = w.external(Shape::Object(fields));
                w.pass(a, parameter);
                a
            }),
            ("let a; &a into a place declared &mut u8", "refused:UndecidedClass", |w| {
                let a = w.undecided();
                let declared = w.external(U8);
                let place = w.external(Shape::Ref(EXCLUSIVE, declared));
                let (lent, _) = w.lend(a, SHARED);
                w.enter(lent, place);
                a
            }),
        ];
        for (name, expected, build) in programs {
            in_every_order(build, expected, name);
        }
    }

    /// `T = [T]`, written by a declaration and refused as its own part, with `x`
    /// passed to it. A class that is a part of itself has no finite type, so it
    /// gives `x` no head, and settling ends.
    #[test]
    fn a_class_that_is_its_own_part_gives_no_head() {
        let mut world = World::new();
        let hole = world.undecided();
        let slice = world.external(Shape::Slice(hole));
        world.enter(hole, slice);
        world.enter(slice, hole);
        let argument = world.undecided();
        world.pass(argument, slice);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(world.classes.is_own_part(slice));
        assert!(world.outcome.refusals.iter().any(|refusal| refusal.cause == Cause::Occurs));
        assert_eq!(world.shape(argument), Shape::Never);
    }

    /// `let y; let mut a = return; a = f(a, Some(y));`, with
    /// `f(Option<T>, Option<T>) -> Option<&T>`: `T = &T`, and `y` goes to `T` at a
    /// part.
    #[test]
    fn a_part_that_goes_to_a_class_that_is_its_own_part_takes_no_head_from_it() {
        let mut world = World::new();
        let y = world.undecided();
        let a = world.text(Shape::Never);
        let hole = world.undecided();
        let (first, second) = (world.external(Shape::Option(hole)), world.external(Shape::Option(hole)));
        let reference = world.external(Shape::Ref(SHARED, hole));
        let result = world.external(Shape::Option(reference));
        world.pass(a, first);
        let some = some_of(&mut world, y);
        world.pass(some, second);
        world.enter(result, a);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(world.classes.is_own_part(hole));
        assert!(world.outcome.refusals.iter().any(|refusal| refusal.cause == Cause::Occurs));
    }

    /// `let a; g(&a);`, with `g(&S)` and `S = [S]`: at a reference's target too, a
    /// class that is a part of itself gives no head.
    #[test]
    fn a_target_that_is_its_own_part_gives_no_head() {
        let mut world = World::new();
        let hole = world.undecided();
        let slice = world.external(Shape::Slice(hole));
        world.enter(hole, slice);
        world.enter(slice, hole);
        let parameter = world.external(Shape::Ref(SHARED, slice));
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);
        world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(world.classes.is_own_part(slice));
        assert_eq!(world.shape(a), Shape::Never);
    }

    /// A program random settling found, whose cycles are closed by unions that
    /// depend on the visit order: which unions close them decided which items the
    /// occurs refusals named, and how many there were. The refusals name what wrote
    /// each class that is a part of itself, and are the same in every order.
    #[test]
    fn an_occurs_refusal_names_the_same_items_in_every_order() {
        let mut seen = BTreeSet::new();
        for order in 0..24u64 {
            let mut world = World::new();
            let first_hole = world.undecided();
            let option = world.external(Shape::Option(first_hole));
            literal(&mut world, &INTEGERS);
            let second_hole = world.undecided();
            world.pass(option, option);
            let slice = world.external(Shape::Slice(second_hole));
            let alone = world.undecided();
            literal(&mut world, &INTEGERS);
            world.enter(second_hole, slice);
            world.enter(first_hole, slice);
            let place = meet(&mut world, alone, alone);
            meet(&mut world, place, option);
            reorder(&mut world.open, order);

            assert_eq!(world.settle_fully(), Settled::Finished);

            let mut refusals: Vec<String> = world.outcome.refusals.iter().map(|refusal| format!("{refusal:?}")).collect();
            refusals.sort();
            assert!(refusals.iter().any(|refusal| refusal.contains("Occurs")), "{refusals:?}");
            seen.insert(refusals);
        }
        assert_eq!(seen.len(), 1, "{seen:#?}");
    }

    /// `p.f = p; p = p.f; take_u8(p); take_u16(p);`: the cycle makes `p` its own
    /// field, and `p` is left undecided. Nothing decided it, so the occurs refusal
    /// names the projection that named the part.
    #[test]
    fn an_undecided_class_that_is_its_own_part_is_refused_at_what_named_the_part() {
        let mut world = World::new();
        let p = world.undecided();
        let (f, named) = project(&mut world, p, field("f"), Side::Lower);
        world.enter(p, f);
        world.enter(f, p);
        let (first, second) = (world.external(U8), world.external(Shape::Scalar(Scalar::U16)));
        world.pass(p, first);
        world.pass(p, second);

        assert_eq!(world.settle_fully(), Settled::Finished);

        assert!(!world.classes.is_decided(p));
        assert!(world.classes.is_own_part(p));
        let occurs: Vec<&Refusal> = world.outcome.refusals.iter().filter(|refusal| refusal.cause == Cause::Occurs).collect();
        assert_eq!(occurs.len(), 1);
        assert!(occurs[0].items.contains(&named));
    }
}
