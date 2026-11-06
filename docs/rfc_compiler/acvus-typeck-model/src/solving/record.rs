//! What the text says about classes, and what has been decided from it.

use std::collections::{BTreeMap, BTreeSet};

use crate::decl::DeclId;
use crate::lit::LitVar;
use crate::program::Projection;
use crate::ty::{Member, Mutability, Name, PartKey, Term};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClassId(pub(crate) usize);

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ClassName {
    Text(Name),
    Part(Box<ClassName>, PartKey),
    Made(Name, usize),
}

#[derive(Clone, Default, Debug)]
pub struct Classes {
    names: Vec<ClassName>,
    parent: Vec<usize>,
    by_name: BTreeMap<ClassName, ClassId>,
}

impl Classes {
    pub fn intern(&mut self, name: ClassName) -> ClassId {
        if let Some(&id) = self.by_name.get(&name) {
            return id;
        }
        let id = ClassId(self.names.len());
        self.names.push(name.clone());
        self.parent.push(id.0);
        self.by_name.insert(name, id);
        id
    }

    pub fn part(&mut self, owner: ClassId, key: PartKey) -> ClassId {
        let owner = self.name(self.root(owner)).clone();
        self.intern(ClassName::Part(Box::new(owner), key))
    }

    pub fn fresh(&mut self, maker: &Name) -> ClassId {
        let n = self.names.len();
        self.intern(ClassName::Made(maker.clone(), n))
    }

    pub fn name(&self, id: ClassId) -> &ClassName {
        &self.names[id.0]
    }

    pub fn root(&self, id: ClassId) -> ClassId {
        let mut at = id.0;
        while self.parent[at] != at {
            at = self.parent[at];
        }
        ClassId(at)
    }

    pub fn equate(&mut self, a: ClassId, b: ClassId) -> bool {
        let (a, b) = (self.root(a), self.root(b));
        if a == b {
            return false;
        }
        let (keep, merge) = if a < b { (a, b) } else { (b, a) };
        self.parent[merge.0] = keep.0;
        true
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn roots(&self) -> impl Iterator<Item = ClassId> + '_ {
        (0..self.names.len())
            .map(ClassId)
            .filter(|&id| self.root(id) == id)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Requirement {
    pub projection: Projection,
    pub part: ClassId,
}

impl Requirement {
    pub fn member(&self) -> Option<Member> {
        match &self.projection {
            Projection::Field(name) => Some(Member::Field(name.clone())),
            Projection::Payload { variant, .. } => Some(Member::Variant(variant.clone())),
            Projection::Deref | Projection::Component(_) | Projection::Elem => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Relation {
    Enters {
        lower: ClassId,
        upper: ClassId,
        part: bool,
    },
    Into { value: ClassId, written: ClassId },
    Requires {
        class: ClassId,
        requirement: Requirement,
    },
    Literal { class: ClassId, lit: LitVar },
    Written {
        class: ClassId,
        structure: Term<ClassId>,
    },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Choice {
    pub name: Name,
    pub candidates: Vec<DeclId>,
    pub remaining: BTreeSet<usize>,
    pub args: Vec<ClassId>,
    pub result: ClassId,
    pub vars: Vec<ClassId>,
    pub result_var_indices: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum HintKind {
    Lend {
        mutability: Mutability,
        place: ClassId,
        result: ClassId,
    },
    Project {
        projection: Projection,
        operand: ClassId,
        result: ClassId,
    },
    Bind {
        path: Vec<Name>,
        position: ClassId,
        binding: ClassId,
    },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum HintDecision {
    Place,
    Through { mutability: Mutability },
    Applied,
    AppliedToTarget,
    Read,
}

impl Choice {
    pub fn result_vars(&self, record: &Record) -> Vec<ClassId> {
        let _ = record;
        self.result_var_indices.iter().map(|&k| self.vars[k]).collect()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Hint {
    pub name: Name,
    pub kind: HintKind,
    pub decision: Option<HintDecision>,
}

#[derive(Clone, Default, Debug)]
pub struct Record {
    pub classes: Classes,
    pub relations: Vec<Relation>,
    pub choices: Vec<Choice>,
    pub hints: Vec<Hint>,
    pub solved: BTreeMap<ClassId, Term<ClassId>>,
    pub refused: BTreeMap<ClassId, Refusal>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    NoJoin,
    Contexts,
    NoValue,
    Conflict,
    Infinite,
}

impl Record {
    pub fn root(&self, id: ClassId) -> ClassId {
        self.classes.root(id)
    }

    // A class is empty where no value enters it: nothing passes into it, by
    // `≤` or by `⇝`, and no literal is of it. Its occurrences, requirements
    // and the structures it is a part of bring it no value.
    pub fn is_empty(&self, relations: &Relations, id: ClassId) -> bool {
        let root = self.root(id);
        let entered = relations.all.iter().any(|relation| match relation {
            Relation::Enters { lower, upper, .. } => {
                self.root(*upper) == root && self.root(*lower) != root
            }
            Relation::Into { value, written } => {
                self.root(*written) == root && self.root(*value) != root
            }
            _ => false,
        });
        !entered && self.literals(root).is_empty() && !self.in_entered_written(root)
    }

    // A variable inside a written type a value passes into receives what the
    // coercion gives it, so it is not empty while that value is on its way.
    fn in_entered_written(&self, root: ClassId) -> bool {
        let entered: Vec<ClassId> = self
            .relations
            .iter()
            .filter_map(|relation| match relation {
                Relation::Into { written, .. } => Some(self.root(*written)),
                _ => None,
            })
            .collect();
        let mut seen = BTreeSet::new();
        let mut work = entered;
        while let Some(class) = work.pop() {
            if !seen.insert(class) {
                continue;
            }
            if class == root {
                return true;
            }
            if let Some(structure) = self.structure(class) {
                work.extend(structure.parts().into_iter().map(|(_, p)| self.root(*p)));
            }
        }
        false
    }

    pub fn may_enter(&self, relations: &Relations, id: ClassId) -> bool {
        let root = self.root(id);
        let by_choice = self.choices.iter().any(|choice| {
            choice.remaining.len() > 1
                && choice
                    .result_vars(self)
                    .iter()
                    .any(|&var| self.root(var) == root)
        });
        let by_hint = self.hints.iter().any(|hint| {
            hint.decision.is_none()
                && matches!(hint.kind, HintKind::Bind { binding, .. } if self.root(binding) == root)
        });
        let by_passing = relations.all.iter().any(|relation| match relation {
            Relation::Enters { lower, upper, .. } => {
                !self.is_decided(*lower)
                    && !self.is_refused(*lower)
                    && self
                        .structure(*upper)
                        .is_some_and(|above| above.parts().iter().any(|(_, p)| self.root(**p) == root))
            }
            _ => false,
        });
        let by_coercion = !self.is_decided(root)
            && self.relations.iter().any(|relation| match relation {
                Relation::Into { value, written } => {
                    !self.is_refused(*value)
                        && !self.value_closed(*value)
                        && self.contains_part(*written, root)
                }
                _ => false,
            });
        by_choice || by_hint || by_passing || by_coercion
    }

    fn contains_part(&self, owner: ClassId, root: ClassId) -> bool {
        let mut seen = BTreeSet::new();
        let mut work = vec![self.root(owner)];
        while let Some(class) = work.pop() {
            if !seen.insert(class) {
                continue;
            }
            if class == root {
                return true;
            }
            if let Some(structure) = self.structure(class) {
                work.extend(structure.parts().into_iter().map(|(_, p)| self.root(*p)));
            }
        }
        false
    }

    pub fn value_closed(&self, id: ClassId) -> bool {
        let mut seen = BTreeSet::new();
        let mut work = vec![self.root(id)];
        while let Some(class) = work.pop() {
            if !seen.insert(class) {
                continue;
            }
            let Some(structure) = self.structure(class) else {
                return false;
            };
            work.extend(structure.parts().into_iter().map(|(_, p)| self.root(*p)));
        }
        true
    }

    pub fn may_take(&self, relations: &Relations, id: ClassId) -> bool {
        let root = self.root(id);
        let by_hint = self.hints.iter().any(|hint| {
            hint.decision.is_none()
                && matches!(hint.kind, HintKind::Bind { position, .. } if self.root(position) == root)
        });
        let by_passing = relations.all.iter().any(|relation| match relation {
            Relation::Enters { lower, upper, .. } => {
                !self.is_decided(*upper)
                    && !self.is_refused(*upper)
                    && self
                        .structure(*lower)
                        .is_some_and(|below| below.parts().iter().any(|(_, p)| self.root(**p) == root))
            }
            _ => false,
        });
        by_hint || by_passing
    }

    pub fn may_require(&self, id: ClassId) -> bool {
        let root = self.root(id);
        self.hints.iter().any(|hint| {
            hint.decision.is_none()
                && matches!(hint.kind, HintKind::Project { operand, .. } if self.root(operand) == root)
        })
    }

    pub fn written(&self, id: ClassId) -> Vec<&Term<ClassId>> {
        let root = self.root(id);
        self.relations
            .iter()
            .filter_map(|relation| match relation {
                Relation::Written { class, structure } if self.root(*class) == root => {
                    Some(structure)
                }
                _ => None,
            })
            .collect()
    }

    pub fn structure(&self, id: ClassId) -> Option<&Term<ClassId>> {
        let root = self.root(id);
        if self.refused.contains_key(&root) {
            return None;
        }
        if let Some(solved) = self.solved.get(&root) {
            return Some(solved);
        }
        self.written(root).into_iter().next()
    }

    pub fn is_decided(&self, id: ClassId) -> bool {
        self.structure(id).is_some()
    }

    pub fn is_refused(&self, id: ClassId) -> bool {
        self.refused.contains_key(&self.root(id))
    }

    // Part relations are read from the decided structures each time instead of
    // being recorded: a recorded one would have to be revised when a structure
    // it was read from is joined with more lower bounds.
    // The relations of the record with those decomposition gives: for each
    // lower bound whose sides both hold a structure of one structural
    // constructor, a part lower bound between the parts at each key, to a
    // fixpoint. Computed at each reading of the record, never kept in it.
    fn relations_decomposed(&self) -> Vec<Relation> {
        let mut relations: Vec<Relation> = self.relations.clone();
        let mut seen: BTreeSet<(ClassId, ClassId)> = relations
            .iter()
            .filter_map(|relation| match relation {
                Relation::Enters { lower, upper, .. } => Some((self.root(*lower), self.root(*upper))),
                _ => None,
            })
            .collect();
        let mut at = 0;
        while at < relations.len() {
            let Relation::Enters { lower, upper, .. } = relations[at].clone() else {
                at += 1;
                continue;
            };
            at += 1;
            let (Some(below), Some(above)) = (self.structure(lower), self.structure(upper)) else {
                continue;
            };
            let constructor = below.constructor();
            if constructor != above.constructor() || !constructor.is_structural() {
                continue;
            }
            for (key, part) in below.parts() {
                let Some(other) = above.part(&key) else {
                    continue;
                };
                if seen.insert((self.root(*part), self.root(*other))) {
                    relations.push(Relation::Enters {
                        lower: *part,
                        upper: *other,
                        part: true,
                    });
                }
            }
        }
        relations
    }

    pub fn lower_bounds(&self, relations: &Relations, id: ClassId) -> Vec<(ClassId, bool)> {
        relations
            .lowers
            .get(&self.root(id))
            .cloned()
            .unwrap_or_default()
    }

    pub fn occurrences(&self, relations: &Relations, id: ClassId) -> Vec<ClassId> {
        relations
            .uppers
            .get(&self.root(id))
            .cloned()
            .unwrap_or_default()
    }

    pub fn requirements(&self, id: ClassId) -> Vec<&Requirement> {
        let root = self.root(id);
        self.relations
            .iter()
            .filter_map(|relation| match relation {
                Relation::Requires { class, requirement } if self.root(*class) == root => {
                    Some(requirement)
                }
                _ => None,
            })
            .collect()
    }

    pub fn literals(&self, id: ClassId) -> Vec<&LitVar> {
        let root = self.root(id);
        self.relations
            .iter()
            .filter_map(|relation| match relation {
                Relation::Literal { class, lit } if self.root(*class) == root => Some(lit),
                _ => None,
            })
            .collect()
    }
}

// The relations of one state of the record, with those decomposition gives,
// indexed by the class on each side, and the components of classes that
// wait for one another. Made once where the record is read, and dropped.
pub struct Relations {
    pub all: Vec<Relation>,
    lowers: BTreeMap<ClassId, Vec<(ClassId, bool)>>,
    uppers: BTreeMap<ClassId, Vec<ClassId>>,
    component: BTreeMap<ClassId, usize>,
}

impl Relations {
    pub fn of(record: &Record) -> Relations {
        let all = record.relations_decomposed();
        let mut lowers: BTreeMap<ClassId, Vec<(ClassId, bool)>> = BTreeMap::new();
        let mut uppers: BTreeMap<ClassId, Vec<ClassId>> = BTreeMap::new();
        for relation in &all {
            let (lower, upper, part) = match relation {
                Relation::Enters { lower, upper, part } => (*lower, *upper, Some(*part)),
                Relation::Into { value, written } => (*value, *written, None),
                _ => continue,
            };
            let (lower, upper) = (record.root(lower), record.root(upper));
            if lower == upper {
                continue;
            }
            if let Some(part) = part {
                lowers.entry(upper).or_default().push((lower, part));
            }
            uppers.entry(lower).or_default().push(upper);
        }
        let mut relations = Relations {
            all,
            lowers,
            uppers,
            component: BTreeMap::new(),
        };
        relations.component = components(record, &relations);
        relations
    }

    pub fn same_component(&self, a: ClassId, b: ClassId) -> bool {
        match (self.component.get(&a), self.component.get(&b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        }
    }
}

fn waits_for(record: &Record, relations: &Relations, root: ClassId) -> Vec<ClassId> {
    record
        .lower_bounds(relations, root)
        .into_iter()
        .map(|(lower, _)| lower)
        .chain(record.occurrences(relations, root))
        .filter(|&class| !record.is_decided(class) && !record.is_refused(class))
        .collect()
}

// Tarjan's strongly connected components over `waits_for`, for every
// undecided root.
fn components(record: &Record, relations: &Relations) -> BTreeMap<ClassId, usize> {
    struct Tarjan<'a> {
        record: &'a Record,
        relations: &'a Relations,
        index: BTreeMap<ClassId, usize>,
        low: BTreeMap<ClassId, usize>,
        on_stack: BTreeSet<ClassId>,
        stack: Vec<ClassId>,
        next: usize,
        found: BTreeMap<ClassId, usize>,
        count: usize,
    }
    impl Tarjan<'_> {
        fn visit(&mut self, at: ClassId) {
            self.index.insert(at, self.next);
            self.low.insert(at, self.next);
            self.next += 1;
            self.stack.push(at);
            self.on_stack.insert(at);
            for next in waits_for(self.record, self.relations, at) {
                if !self.index.contains_key(&next) {
                    self.visit(next);
                    let low = self.low[&next].min(self.low[&at]);
                    self.low.insert(at, low);
                } else if self.on_stack.contains(&next) {
                    let low = self.index[&next].min(self.low[&at]);
                    self.low.insert(at, low);
                }
            }
            if self.low[&at] == self.index[&at] {
                while let Some(member) = self.stack.pop() {
                    self.on_stack.remove(&member);
                    self.found.insert(member, self.count);
                    if member == at {
                        break;
                    }
                }
                self.count += 1;
            }
        }
    }
    let mut tarjan = Tarjan {
        record,
        relations,
        index: BTreeMap::new(),
        low: BTreeMap::new(),
        on_stack: BTreeSet::new(),
        stack: Vec::new(),
        next: 0,
        found: BTreeMap::new(),
        count: 0,
    };
    let roots: Vec<ClassId> = record
        .classes
        .roots()
        .filter(|&root| !record.is_decided(root) && !record.is_refused(root))
        .collect();
    for root in roots {
        if !tarjan.index.contains_key(&root) {
            tarjan.visit(root);
        }
    }
    tarjan.found
}
