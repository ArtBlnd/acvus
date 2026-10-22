//! The cut of each lambda's captured environment (l.186;
//! `design/17-environment.md`): which binding group solves each class and each
//! constraint of the text, a lambda's environment moving them into the group
//! enclosing it. A use gives a constraint for each set of positions its callee
//! ties: a declaration by its signature, a function of another group by the
//! parts of its C, which an instance makes anew (`State::components`, l.215).
//! So the cut is made as the groups are solved: a lambda once the groups it
//! makes instances of are, before it is solved itself (l.213).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::program::{ENTRY, Item, Origin, Part, Position, Program};
use crate::solve::State;
use crate::ty::Name;

/// The classes and constraints the lambdas' captured environments hold, each
/// with the group that solves it, its enclosing one's: an item other than a
/// use, which gives one constraint, and the positions of each constraint a use
/// gives (l.186).
#[derive(Clone, Debug, Default)]
pub struct Cut {
    classes: BTreeMap<Name, Name>,
    items: BTreeMap<usize, Name>,
    positions: BTreeMap<(usize, Position), Name>,
}

/// A constraint an item gives (l.186): the positions it lies at, none for an
/// item other than a use, and the classes it ties.
struct Constraint<'a> {
    positions: Vec<Position>,
    classes: Vec<&'a Name>,
}

impl Cut {
    /// The cut of each lambda solved before `through` and of `through`
    /// itself, of every lambda where `through` is the entry, in the order the
    /// groups are solved, the lambdas inside a group before it (l.213).
    /// `solved` holds each group solved before `through`.
    pub fn of(program: &Program, solved: &BTreeMap<Name, Rc<State>>, through: &Name) -> Cut {
        let order = program.group_order();
        let upto = match through.as_str() {
            ENTRY => order.len(),
            _ => {
                let at = order
                    .iter()
                    .position(|group| group == through)
                    .unwrap_or_else(|| panic!("{through} is a binding group of the program"));
                at + 1
            }
        };
        let mut cut = Cut::default();
        for group in &order[..upto] {
            let parent = program
                .groups
                .iter()
                .find(|(name, _)| name == group)
                .and_then(|(_, parent)| parent.clone());
            if let Some(parent) = parent {
                cut.cut(program, solved, group, &parent);
            }
        }
        cut
    }

    /// The lambda `group`'s environment, moved to `parent`
    /// (`design/17-environment.md`, Definitions): the classes joined to a
    /// capture by the constraints of the lambda's text, those of the lambdas
    /// inside it moved into it among them, and the constraints whose classes
    /// those are (l.186).
    fn cut(
        &mut self,
        program: &Program,
        solved: &BTreeMap<Name, Rc<State>>,
        group: &Name,
        parent: &Name,
    ) {
        let owner = |cut: &Cut, class: &Name| -> Name {
            cut.classes
                .get(class)
                .cloned()
                .unwrap_or_else(|| program.class(class).group.clone())
        };
        // The text inside the group, which alone the cut has moved into it.
        let inside = |i: &usize| {
            let at = &program.items[*i].0;
            at == group || program.enclosing(at).contains(group)
        };
        let constraints: Vec<(usize, Constraint<'_>)> = (0..program.items.len())
            .filter(inside)
            .flat_map(|i| {
                constraints(program, i, solved)
                    .into_iter()
                    .map(move |constraint| (i, constraint))
            })
            .filter(|(i, constraint)| self.placed(program, *i, constraint) == group)
            .collect();
        let mut joined = Joined::default();
        for (i, constraint) in &constraints {
            let mut classes = constraint.classes.clone();
            // A binding is on the classes of the projections on its path as
            // well (l.186).
            if let Item::Bind { path, .. } = &program.items[*i].1 {
                for (j, other) in &constraints {
                    if let Item::Project { name, .. } = &program.items[*j].1
                        && path.contains(name)
                    {
                        classes.extend(other.classes.iter().copied());
                    }
                }
            }
            for pair in classes.windows(2) {
                joined.join(pair[0], pair[1]);
            }
            for name in classes {
                joined.add(name);
            }
        }
        for class in &program.classes {
            if owner(self, &class.name) != *group {
                continue;
            }
            joined.add(&class.name);
            if let Origin::Written(term) | Origin::Host(term) | Origin::Outward(term) =
                &class.origin
            {
                for (_, part) in term.parts() {
                    joined.join(&class.name, part);
                }
            }
        }
        let captures: BTreeSet<Name> = joined
            .names()
            .filter(|name| owner(self, name) != *group)
            .map(|name| joined.root(name))
            .collect();
        let held: Vec<Name> = joined
            .names()
            .filter(|name| captures.contains(&joined.root(name)))
            .cloned()
            .collect();
        for name in &held {
            if owner(self, name) == *group {
                self.classes.insert(name.clone(), parent.clone());
            }
        }
        for (i, constraint) in constraints {
            // A constraint's classes are joined, so they are the environment's
            // all together or none of them.
            let in_environment = constraint
                .classes
                .iter()
                .all(|name| captures.contains(&joined.root(name)));
            if !in_environment {
                continue;
            }
            if constraint.positions.is_empty() {
                self.items.insert(i, parent.clone());
            }
            for position in constraint.positions {
                self.positions.insert((i, position), parent.clone());
            }
        }
    }

    /// The group that records the constraint the item at `i` gives, as the
    /// cut has it so far.
    fn placed<'a>(
        &'a self,
        program: &'a Program,
        i: usize,
        constraint: &Constraint<'_>,
    ) -> &'a Name {
        let at = &program.items[i].0;
        match constraint.positions.first() {
            None => self.items.get(&i).unwrap_or(at),
            Some(&position) => self.positions.get(&(i, position)).unwrap_or(at),
        }
    }

    /// The group that solves the class: the one it lies in, or the group
    /// enclosing the lambda whose environment holds it.
    pub fn owner<'a>(&'a self, program: &'a Program, class: &Name) -> &'a Name {
        self.classes
            .get(class)
            .unwrap_or_else(|| &program.class(class).group)
    }

    /// What of the item at `i` in `items` the group records: an item other
    /// than a use, whole where the group records it, and the positions of a
    /// use's constraints it records (l.186); none where it records nothing of
    /// it.
    pub fn part(&self, program: &Program, i: usize, group: &Name) -> Option<Part> {
        let (at, item) = &program.items[i];
        match item {
            Item::Use { .. } => {
                let positions: BTreeSet<Position> = item
                    .positions()
                    .into_iter()
                    .filter(|&position| self.positions.get(&(i, position)).unwrap_or(at) == group)
                    .collect();
                (!positions.is_empty()).then_some(Part::At(positions))
            }
            _ => (self.items.get(&i).unwrap_or(at) == group).then_some(Part::Whole),
        }
    }

    /// Whether a lambda's environment holds the class.
    pub fn in_environment(&self, class: &Name) -> bool {
        self.classes.contains_key(class)
    }

    /// The classes the lambdas' environments hold.
    pub fn environment(&self) -> BTreeSet<Name> {
        self.classes.keys().cloned().collect()
    }

    /// Whether the class is universal: a function's parameter type that no
    /// environment holds, or a receiver's `T̄` (l.210, l.314).
    pub fn is_universal(&self, program: &Program, name: &Name) -> bool {
        let parameter = program
            .functions
            .iter()
            .any(|function| function.params.contains(name));
        program.universals.contains(name) || parameter && !self.in_environment(name)
    }

    /// Whether the class is an unknown of the group that solves it, as its
    /// text gives it (l.211): not written, and not universal.
    pub fn is_text_unknown(&self, program: &Program, name: &Name) -> bool {
        let class = program.class(name);
        matches!(class.origin, Origin::Var | Origin::Lit(_)) && !self.is_universal(program, name)
    }
}

/// The constraints the item at `i` gives (l.186), each with the classes it
/// ties. An item other than a use gives one, on all it relates. A use gives
/// one for each set of positions its callee ties. Of declarations: a position
/// where every candidate writes one closed type, alone; with one candidate,
/// the positions that write one of its variables, together; with several, the
/// rest, together, the choice of a candidate deciding them, and `e?`'s exit
/// with them. Of a function of its own or an enclosing group: each position
/// alone, tied to that function's parameter or result type, the body's own
/// types. Of another group: the positions whose types lie in one part of that
/// group's C, which an instance makes anew together, and each whose type lies
/// outside C alone (`State::components`, l.215).
fn constraints<'a>(
    program: &'a Program,
    i: usize,
    solved: &BTreeMap<Name, Rc<State>>,
) -> Vec<Constraint<'a>> {
    let (at, item) = &program.items[i];
    let Item::Use {
        callee, args, exit, ..
    } = item
    else {
        return vec![Constraint {
            positions: Vec::new(),
            classes: item.classes(),
        }];
    };
    let alone = |position: Position| Constraint {
        positions: vec![position],
        classes: vec![item.class_at(position)],
    };
    if let Some(function) = program.function(callee) {
        assert!(
            exit.is_none(),
            "{callee}: a call of a function has no exit (l.337)"
        );
        let params = (0..args.len())
            .map(|k| (Position::Arg(k), &function.params[k]))
            .chain([(Position::Result, &function.result)]);
        let own = function.group == *at || program.enclosing(at).contains(&function.group);
        if own {
            return params
                .map(|(position, param)| Constraint {
                    positions: vec![position],
                    classes: vec![item.class_at(position), param],
                })
                .collect();
        }
        let Some(from) = solved.get(&function.group) else {
            panic!("{} is solved before {at} (l.213)", function.group);
        };
        let components = from.components();
        let mut by_part: BTreeMap<usize, Constraint<'a>> = BTreeMap::new();
        let mut apart = Vec::new();
        for (position, param) in params {
            match components.of(from, from.text(param)) {
                Some(part) => {
                    let constraint = by_part.entry(part).or_insert(Constraint {
                        positions: Vec::new(),
                        classes: Vec::new(),
                    });
                    constraint.positions.push(position);
                    constraint.classes.push(item.class_at(position));
                }
                None => apart.push(alone(position)),
            }
        }
        return by_part.into_values().chain(apart).collect();
    }
    let sigs: Vec<&crate::sig::Signature> = program
        .decls
        .candidates(callee)
        .iter()
        .filter(|decl| decl.sig.params().len() == args.len())
        .map(|decl| &decl.sig)
        .collect();
    let positions = item.positions();
    if sigs.is_empty() {
        return vec![Constraint {
            classes: item.classes(),
            positions,
        }];
    }
    let together = |positions: Vec<Position>| Constraint {
        classes: positions
            .iter()
            .map(|&position| item.class_at(position))
            .collect(),
        positions,
    };
    if let [sig] = sigs.as_slice() {
        assert!(
            exit.is_none(),
            "`?`'s candidates are the language's two (l.337)"
        );
        let written = |position: Position| match position {
            Position::Arg(k) => &sig.params()[k],
            Position::Result => sig.result(),
            Position::Exit => unreachable!("a use of one candidate has no exit"),
        };
        let vars: Vec<crate::sig::SigVar> = (0..sig.type_params().len())
            .map(crate::sig::SigVar::Type)
            .chain(sig.names_identity().then_some(crate::sig::SigVar::Identity))
            .collect();
        // Each position's set, the sets of two positions writing one variable
        // merged.
        let mut set: Vec<usize> = (0..positions.len()).collect();
        for &var in &vars {
            let writing: Vec<usize> = (0..positions.len())
                .filter(|&p| written(positions[p]).mentions(var))
                .collect();
            let Some(&first) = writing.first() else {
                continue;
            };
            let merged: BTreeSet<usize> = writing.iter().map(|&p| set[p]).collect();
            let into = set[first];
            for label in &mut set {
                if merged.contains(label) {
                    *label = into;
                }
            }
        }
        let mut sets: BTreeMap<usize, Vec<Position>> = BTreeMap::new();
        for (p, &position) in positions.iter().enumerate() {
            sets.entry(set[p]).or_default().push(position);
        }
        return sets.into_values().map(together).collect();
    }
    let part = crate::node::common::CommonPart::of(&sigs);
    let (closed, chosen): (Vec<Position>, Vec<Position>) = positions
        .into_iter()
        .partition(|&position| part.is_closed_at(position));
    assert!(
        !chosen.is_empty(),
        "{callee}: candidates writing one closed type at every position are not encoded"
    );
    closed
        .into_iter()
        .map(alone)
        .chain([together(chosen)])
        .collect()
}

/// Classes joined by the constraints on them together
/// (`design/17-environment.md`).
#[derive(Default)]
struct Joined {
    parent: BTreeMap<Name, Name>,
}

impl Joined {
    fn add(&mut self, name: &Name) {
        self.parent
            .entry(name.clone())
            .or_insert_with(|| name.clone());
    }

    fn root(&self, name: &Name) -> Name {
        let mut at = name.clone();
        while let Some(next) = self.parent.get(&at).filter(|next| **next != at) {
            at = next.clone();
        }
        at
    }

    fn join(&mut self, a: &Name, b: &Name) {
        self.add(a);
        self.add(b);
        let (a, b) = (self.root(a), self.root(b));
        if a != b {
            self.parent.insert(a, b);
        }
    }

    fn names(&self) -> impl Iterator<Item = &Name> {
        self.parent.keys()
    }
}
