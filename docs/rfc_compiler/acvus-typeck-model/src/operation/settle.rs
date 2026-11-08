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
//! | 2. upper | a class takes the one head its upper bounds give it (O19) | not here yet |
//! | 3. domain | a literal's class takes its row's fallback type (O20) | yes |
//! | 4. never | a class nothing says anything about takes `!` | yes |
//! | end | a class still undecided is refused (O21) | |
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
use super::bound::{Join, given, join_lowers};

/// Where solving stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Settled {
    /// No rung changes anything. What is still undecided has been refused.
    Finished,
    /// What upper bounds give is owed, which is not made yet, or a change cannot
    /// be made yet. Solving stops here: a fallback taken now could decide a class
    /// they would decide otherwise.
    AwaitsUpper,
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

/// Whether going up from `class`, a head is there to be given: a decided class,
/// or a class something has entered.
fn upper_gives_a_head<K>(classes: &Classes<K>, class: K::Id) -> bool
where
    K: ClassKind<Bound = Bound<K>>,
{
    let mut seen = BTreeSet::new();
    let mut stack = vec![class];
    while let Some(class) = stack.pop() {
        let root = classes.find(class);
        if !seen.insert(root.id()) {
            continue;
        }
        for (upper, _) in classes.uppers(root) {
            if upper == root {
                continue;
            }
            if classes.get(upper).state.is_decided() {
                return true;
            }
            let entered = classes
                .lowers(upper)
                .into_values()
                .flatten()
                .any(|(bound, _)| given(classes, &bound).is_some());
            if entered {
                return true;
            }
            stack.push(upper.id());
        }
    }
    false
}

fn every_root<K>(classes: &Classes<K>) -> Vec<K::Id>
where
    K: ClassKind,
{
    classes.roots().map(|root| root.id()).collect()
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Takes every rung there is, `at` being the item a decision here goes by.
    pub fn settle(&mut self, classes: &mut Classes<K>, scopes: &Scopes, at: ItemId) -> Settled {
        self.settle_visiting(classes, scopes, at, &every_root)
    }

    /// As [`Outcome::settle`], visiting classes in the order `visit` gives, which
    /// the outcome should not depend on.
    pub(super) fn settle_visiting<F>(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        at: ItemId,
        visit: &F,
    ) -> Settled
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        loop {
            self.settle_lower(classes, scopes, at, visit);
            let owed = !self.bounds.is_empty()
                || visit(classes).into_iter().any(|class| {
                    !classes.is_decided(class)
                        && join_lowers(classes, class) == Join::Nothing
                        && upper_gives_a_head(classes, class)
                });
            if owed {
                return Settled::AwaitsUpper;
            }
            if self.fall_back_to_domains(classes, scopes, at, visit) || self.fall_back_to_never(classes, scopes, at, visit)
            {
                continue;
            }
            self.refuse_undecided(classes, visit);
            return Settled::Finished;
        }
    }

    /// Rungs 0 and 1, until neither changes anything.
    pub(super) fn settle_lower<F>(&mut self, classes: &mut Classes<K>, scopes: &Scopes, at: ItemId, visit: &F)
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        loop {
            let connected = self.connect_structure(classes, scopes);
            let decided = self.decide_by_join(classes, scopes, at, visit);
            if !connected && !decided {
                break;
            }
        }
    }

    /// Rung 0: connects what can be connected with no head decided, to a
    /// fixpoint, and says whether it connected anything.
    fn connect_structure(&mut self, classes: &mut Classes<K>, scopes: &Scopes) -> bool {
        let mut connected = false;
        loop {
            let merged = self.merge_cycles(classes, scopes);
            let carried = self.carry_parts(classes, scopes);
            let lent = self.connect_lends(classes, scopes);
            if !merged && !carried && !lent {
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

    /// Rung 3: each class that nothing enters and a literal has registered a
    /// domain in takes its row's fallback type, all at once.
    fn fall_back_to_domains<F>(&mut self, classes: &mut Classes<K>, scopes: &Scopes, at: ItemId, visit: &F) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let mut falling = Vec::new();
        for class in visit(classes) {
            if classes.is_decided(class) || join_lowers(classes, class) != Join::Nothing {
                continue;
            }
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

    /// Rung 4: each class that nothing enters and no literal has registered a
    /// domain in takes `!`, all at once.
    fn fall_back_to_never<F>(&mut self, classes: &mut Classes<K>, scopes: &Scopes, at: ItemId, visit: &F) -> bool
    where
        F: Fn(&Classes<K>) -> Vec<K::Id>,
    {
        let falling: Vec<K::Id> = visit(classes)
            .into_iter()
            .filter(|class| {
                !classes.is_decided(*class)
                    && join_lowers(classes, *class) == Join::Nothing
                    && matches!(
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
    use crate::primitive::ty::Scalar;
    use crate::primitive::{ClassId, Name};

    fn describe(world: &World, class: ClassId, depth: usize) -> String {
        if depth == 0 {
            return "…".to_owned();
        }
        let Some(shape) = shape_of(&world.classes, class) else {
            return "?".to_owned();
        };
        match shape {
            Shape::Option(payload) => format!("Option<{}>", describe(world, *payload, depth - 1)),
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

    /// The same outcome in every visit order: as written, reversed, and shuffled.
    fn in_every_order<F>(build: F, expected: &str, name: &str)
    where
        F: Fn(&mut World) -> ClassId,
    {
        for order in 0..24u64 {
            let mut world = World::new();
            let observed = build(&mut world);
            let count = world.open.len();
            match order {
                0 => {}
                1 => world.open.reverse(),
                seed => {
                    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                    for i in (1..count).rev() {
                        state = state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                        let j = usize::try_from(state >> 33).expect("a shifted word fits") % (i + 1);
                        world.open.swap(i, j);
                    }
                }
            }
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
    /// falls back to, which `z` then has to give `y`. All at once, `y` would be
    /// `!`, though what it holds has a type.
    #[test]
    fn never_waits_for_what_a_literal_falls_back_to() {
        let mut world = World::new();
        let y = world.undecided();
        let zero = literal(&mut world, &INTEGERS);
        let z = meet(&mut world, y, zero);

        assert_eq!(world.settle_fully(), Settled::AwaitsUpper);

        world.no_refusals();
        assert_eq!(world.shape(zero), Shape::Scalar(Scalar::I32));
        assert!(!world.classes.is_decided(y));
        assert!(!world.classes.is_decided(z));
    }

    /// Example 9: `let x = 0; take(x);`, with `take(u8)`. Falling back here would
    /// make the literal `i32`, and refuse the program.
    #[test]
    fn no_fallback_is_taken_while_an_upper_bound_has_a_head_to_give() {
        let mut world = World::new();
        let zero = literal(&mut world, &INTEGERS);
        let x = world.meets(zero);
        let boundary = world.meets(x);
        let parameter = world.external(U8);
        world.record(Bound::Change {
            lower: boundary,
            upper: parameter,
        });

        assert_eq!(world.settle_fully(), Settled::AwaitsUpper);

        world.no_refusals();
        assert!(!world.classes.is_decided(zero));
    }

    /// Example 23: `let a; if let Some(v) = a {} take(a);`. `v` says nothing of
    /// itself, and `!` would refuse it at the upper bound it gets later.
    #[test]
    fn no_class_falls_back_to_never_while_another_has_a_head_to_take() {
        let mut world = World::new();
        let a = world.undecided();
        let (v, _) = project(&mut world, a, PartKey::Payload(Name::new("Some")), Side::Upper);
        let boundary = world.meets(a);
        let declared_u8 = world.external(U8);
        let parameter = world.external(Shape::Option(declared_u8));
        world.record(Bound::Change {
            lower: boundary,
            upper: parameter,
        });

        assert_eq!(world.settle_fully(), Settled::AwaitsUpper);

        world.no_refusals();
        assert!(!world.classes.is_decided(v));
    }

    #[test]
    fn a_change_not_made_yet_stops_the_fallbacks() {
        let mut world = World::new();
        let other = world.undecided();
        let target = world.undecided();
        let argument = world.external(Shape::Ref(SHARED, target));
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let at = world.record(Bound::Change {
            lower: argument,
            upper: parameter,
        });

        assert_eq!(world.settle_fully(), Settled::AwaitsUpper);

        assert!(!world.classes.is_decided(other));
        assert!(!world.classes.is_decided(target));
        assert_eq!(
            world.outcome.bounds,
            BTreeSet::from([(
                Bound::Change {
                    lower: argument,
                    upper: parameter
                },
                at
            )])
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

        let settled = world.outcome.settle(&mut world.classes, &world.scopes, at);

        assert_eq!(settled, Settled::Finished);
        world.no_refusals();
        assert_eq!(world.shape(part), U8);
    }
}
