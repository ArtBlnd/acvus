//! An equivalence class's value: the node of the cast tree and the members
//! its sources give, joined, the structure the class takes at that node with
//! its parts made anew, the closed type it holds once every part is decided,
//! and a class that reaches a part of its own, which no finite type holds.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Cause;
use crate::domain::Domain;
use crate::class::{ClassId, ClassName, Side, StructureConstraint, StructureSource};
use crate::decl::Node;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Member, Mutability, Name, PartKey, Term, Ty};

/// What a source gives at a join: a node of the cast tree, and the members of
/// the type at it. A join reads no deeper: the cast tree is a tree over nodes,
/// so the casts that raise a value are decided by its node alone, and what
/// lies below a node passes through them as type parameters, which the
/// decomposition relates.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Head {
    pub node: Node,
    pub members: BTreeSet<Member>,
}

impl Head {
    /// The constructor at the node: a reference's mutability, or the
    /// constructor itself.
    pub fn constructor(&self) -> Constructor {
        match &self.node {
            Node::Owned(constructor) => constructor.clone(),
            Node::Ref(mutability, _) => Constructor::Ref(*mutability),
        }
    }
}

/// The sources of an equivalence class's value, as the judgment reads them
/// from the record.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Sources {
    /// Each solved lower bound, with what it gives.
    pub lowers: Vec<(ClassId, Head)>,
    /// Each solved part lower bound, with what it gives, which reaches the
    /// value by `≤ₚ`.
    pub part_lowers: Vec<(ClassId, Head)>,
    /// What the structure constraints on the class give.
    pub structure_constraints: Vec<Head>,
}

impl Sources {
    pub fn is_empty(&self) -> bool {
        self.lowers.is_empty() && self.part_lowers.is_empty() && self.structure_constraints.is_empty()
    }
}

/// What an equivalence class took at its judgment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Took {
    /// The join of what its sources give, its parts made anew: the structures
    /// recorded, with what unification did as the class took them.
    Joined { head: Head, recorded: Recording },
    /// Nothing enters it: the meet of the classes it passes into, taken with
    /// its parts made anew.
    Met { head: Head, recorded: Recording },
    /// Its sources give it no node but a domain: it takes the domain,
    /// recorded, which bounds its value and which fallback reads.
    Bounded { domain: Domain, recorded: Recording },
    /// Its sources have no join: it has no value.
    NoJoin,
    /// Nothing enters it, and the classes it passes into have no meet: it
    /// has no value.
    NoMeet,
    /// It reaches a part of its own: no finite type holds it, so it has no
    /// value.
    Infinite,
}

impl State {
    /// The node and members of a solved class's structure, none where the
    /// class is a reference whose target has not taken a constructor.
    pub(crate) fn head_of(&self, id: ClassId) -> Option<Head> {
        let root = self.root(id);
        let (structure, _) = self.classes.held(root).structure.as_ref()?;
        let node = match structure {
            Term::Ref(mutability, target) => {
                let (target, _) = self.classes.held(self.root(*target)).structure.as_ref()?;
                Node::Ref(*mutability, target.constructor())
            }
            term => Node::Owned(term.constructor()),
        };
        Some(Head {
            node,
            members: structure.members(),
        })
    }

    /// The join of two sources' nodes and members: `!` below every one, two
    /// references of two mutabilities at the shared one, and two nodes raised
    /// by casts to their nearest common ancestor node, their members united.
    pub(crate) fn head_join(&self, a: &Head, b: &Head) -> Option<Head> {
        let (a, b) = match (&a.node, &b.node) {
            (Node::Owned(Constructor::Never), _) => return Some(b.clone()),
            (_, Node::Owned(Constructor::Never)) => return Some(a.clone()),
            (Node::Ref(m, c), Node::Ref(n, d)) if m != n => (
                Head {
                    node: Node::Ref(Mutability::Shared, c.clone()),
                    members: a.members.clone(),
                },
                Head {
                    node: Node::Ref(Mutability::Shared, d.clone()),
                    members: b.members.clone(),
                },
            ),
            (Node::Ref(..), Node::Owned(_)) | (Node::Owned(_), Node::Ref(..)) => return None,
            _ => (a.clone(), b.clone()),
        };
        let nca = self.decls.tree().nca(&a.node, &b.node)?;
        let (a, b) = (self.raise_head(&a, &nca)?, self.raise_head(&b, &nca)?);
        if a.node != b.node {
            return None;
        }
        let members = match a.node {
            Node::Owned(Constructor::Object | Constructor::Enum(_)) => {
                a.members.union(&b.members).cloned().collect()
            }
            _ if a.members == b.members => a.members,
            _ => return None,
        };
        Some(Head {
            node: a.node,
            members,
        })
    }

    /// The meet of two upper bounds' nodes and members: the greatest node
    /// that reaches both. `!` is no candidate, so a bound of `!` has no meet;
    /// an exclusive reference reaches a shared one by a reborrow, so two
    /// references of two mutabilities meet at the exclusive one; and of two
    /// nodes the one below the other, along the casts, reaches both, so it is
    /// the meet, and two nodes neither of which is below the other have none.
    /// A type that reaches two objects or two enums has the members of both.
    pub(crate) fn head_meet(&self, a: &Head, b: &Head) -> Option<Head> {
        if a.node == Node::Owned(Constructor::Never) || b.node == Node::Owned(Constructor::Never) {
            return None;
        }
        let (a, b) = match (&a.node, &b.node) {
            (Node::Ref(m, c), Node::Ref(n, d)) if m != n => (
                Head {
                    node: Node::Ref(Mutability::Exclusive, c.clone()),
                    members: a.members.clone(),
                },
                Head {
                    node: Node::Ref(Mutability::Exclusive, d.clone()),
                    members: b.members.clone(),
                },
            ),
            (Node::Ref(..), Node::Owned(_)) | (Node::Owned(_), Node::Ref(..)) => return None,
            _ => (a.clone(), b.clone()),
        };
        let tree = self.decls.tree();
        let node = if tree.ancestors(&a.node).contains(&b.node) {
            a.node.clone()
        } else if tree.ancestors(&b.node).contains(&a.node) {
            b.node.clone()
        } else {
            return None;
        };
        let members = match node {
            Node::Owned(Constructor::Object | Constructor::Enum(_)) => {
                a.members.union(&b.members).cloned().collect()
            }
            _ if a.members == b.members => a.members,
            _ => return None,
        };
        Some(Head { node, members })
    }

    /// The join raised where the sources carry two different identities, which
    /// no class holds: to the nearest node above that carries none,
    /// each value cast up to it; none where no such node is, or where a part
    /// lower bound carries one, which reaches with no cast.
    fn without_two_identities(&self, sources: &Sources, joined: Head) -> Option<Head> {
        let identities = |given: &[(ClassId, Head)]| -> BTreeSet<Ty> {
            given
                .iter()
                .filter_map(|(class, _)| self.value(*class))
                .filter_map(|value| crate::join::identity(&value).cloned())
                .collect()
        };
        let (lowers, part_lowers) = (
            identities(&sources.lowers),
            identities(&sources.part_lowers),
        );
        if lowers.union(&part_lowers).count() < 2 {
            return Some(joined);
        }
        if !part_lowers.is_empty() {
            return None;
        }
        let above = crate::join::without_identity(&self.decls, &joined.node)?;
        self.raise_head(&joined, &above)
    }

    /// What structure constraints give: where they are the only sources
    /// (`alone`), the constructor of each member they name, visible or not;
    /// otherwise those of the members visible to this body, which alone a
    /// structure constraint adds to a type another source gives.
    pub(crate) fn constraint_heads(
        &self,
        constraints: &[StructureConstraint],
        alone: bool,
    ) -> Vec<Head> {
        if alone {
            return constraints
                .iter()
                .filter_map(named_member)
                .map(|(constructor, members)| Head {
                    node: Node::Owned(constructor),
                    members,
                })
                .collect();
        }
        constraints
            .iter()
            .filter_map(visible_member)
            .map(|(constructor, member)| Head {
                node: Node::Owned(constructor),
                members: [member].into(),
            })
            .collect()
    }

    /// The node and members that the casts up from `head`'s node reach at
    /// `node`: those of the last cast's result.
    fn raise_head(&self, head: &Head, node: &Node) -> Option<Head> {
        let path = self.decls.tree().path(&head.node, node)?;
        let Some(last) = path.last() else {
            return Some(head.clone());
        };
        let result = self.decls.decl(last).sig.result().clone();
        let crate::sig::SigTy::Term(term) = result else {
            return None;
        };
        let node = match &*term {
            Term::Ref(mutability, crate::sig::SigTy::Term(target)) => {
                Node::Ref(*mutability, target.constructor())
            }
            Term::Ref(_, crate::sig::SigTy::Var(_)) => return None,
            term => Node::Owned(term.constructor()),
        };
        Some(Head {
            node,
            members: term.members(),
        })
    }

    /// The structure at `head`'s node, each part a class made anew and named
    /// after `owner`, recorded as solved. A reference's target is a part,
    /// made anew with the structure of the node's target constructor where
    /// `target_structure`, and holding nothing otherwise: the targets of the
    /// references entering the class are then equated with it by the
    /// decomposition. Every structure's constructor agrees with its
    /// declaration: the text's are checked when they are recorded, and a
    /// signature's when it is declared. Returns the relations recorded.
    pub(crate) fn take_skeleton(
        &mut self,
        root: ClassId,
        head: &Head,
        owner: &ClassName,
        target_structure: bool,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        let (constructor, target) = match &head.node {
            Node::Owned(constructor) => (constructor.clone(), None),
            Node::Ref(mutability, target) => {
                let owner = Rc::new(owner.clone());
                let class = self.fresh(ClassName::Part(owner, PartKey::Target));
                let structure = Term::Ref(*mutability, class);
                let relation = Relation::Structure(root, structure, StructureSource::Solved);
                recorded.push(self.record(relation, recorder.clone()));
                if !target_structure {
                    return recorded;
                }
                (target.clone(), Some(class))
            }
        };
        let (class, owner, members) = match target {
            Some(class) => (class, self.class_name(class), BTreeSet::new()),
            None => (root, owner.clone(), head.members.clone()),
        };
        let structure = self.skeleton(&constructor, &members, &owner);
        let relation = Relation::Structure(class, structure, StructureSource::Solved);
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// The structure of `constructor` with `members`, each part a class made
    /// anew and named after `owner`.
    pub(crate) fn skeleton(
        &mut self,
        constructor: &Constructor,
        members: &BTreeSet<Member>,
        owner: &ClassName,
    ) -> Term<ClassId> {
        let owner = Rc::new(owner.clone());
        let decls = Rc::clone(&self.decls);
        let mut part = |key: PartKey| self.fresh(ClassName::Part(owner.clone(), key));
        let named = |members: &BTreeSet<Member>| -> Vec<Name> {
            members
                .iter()
                .map(|member| match member {
                    Member::Field(name) | Member::Variant(name) => name.clone(),
                })
                .collect()
        };
        match constructor {
            Constructor::Ref(mutability) => Term::Ref(*mutability, part(PartKey::Target)),
            Constructor::Tuple(n) => {
                Term::Tuple((0..*n).map(|i| part(PartKey::Component(i))).collect())
            }
            Constructor::Array(n) => Term::Array(*n, part(PartKey::Element)),
            Constructor::Slice => Term::Slice(part(PartKey::Element)),
            Constructor::Option => Term::Option(part(PartKey::Payload(Name::new(crate::ty::SOME)))),
            Constructor::Result => Term::Result {
                ok: part(PartKey::Payload(Name::new(crate::ty::OK))),
                err: part(PartKey::Payload(Name::new(crate::ty::ERR))),
            },
            Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
            Constructor::Object => Term::Object(
                named(members)
                    .into_iter()
                    .map(|field| (field.clone(), part(PartKey::Field(field))))
                    .collect(),
            ),
            Constructor::Enum(name) => Term::Enum(
                name.clone(),
                named(members)
                    .into_iter()
                    .map(|variant| (variant.clone(), part(PartKey::Payload(variant))))
                    .collect(),
            ),
            Constructor::ErrorOf => Term::ErrorOf(part(PartKey::Wrapped)),
            Constructor::Struct(module) => {
                let fields = decls
                    .types()
                    .struct_decl(module)
                    .expect("a structure's struct is declared")
                    .fields
                    .len();
                Term::Struct(
                    module.clone(),
                    (0..fields).map(|i| part(PartKey::Arg(i))).collect(),
                )
            }
            Constructor::Extern(name) => {
                let decl = decls
                    .types()
                    .extern_decl(name)
                    .expect("a structure's extern-defined type is declared");
                Term::Extern {
                    name: name.clone(),
                    args: (0..decl.arity).map(|i| part(PartKey::Arg(i))).collect(),
                    identity: decl.identity.then(|| part(PartKey::Identity)),
                }
            }
            Constructor::Scalar(scalar) => Term::Scalar(*scalar),
            Constructor::String => Term::String,
            Constructor::Str => Term::Str,
            Constructor::Unit => Term::Unit,
            Constructor::Never => Term::Never,
            Constructor::Identity(name) => Term::Identity(name.clone()),
        }
    }

    /// The equivalence class takes the join of what its sources give, its
    /// parts made anew, each part lower bound's node reaching it by `≤ₚ`,
    /// with no cast; where they have no join it is refused.
    pub(crate) fn take_join(&mut self, root: ClassId, sources: &Sources) -> Took {
        let root = self.root(root);
        let cycle = self.own_part_cycle(root);
        if !cycle.is_empty() {
            for class in cycle {
                self.classes.refuse(class, Cause::Infinite);
            }
            return Took::Infinite;
        }
        let given: Vec<Head> = sources
            .lowers
            .iter()
            .chain(&sources.part_lowers)
            .map(|(_, given)| given.clone())
            .chain(sources.structure_constraints.iter().cloned())
            .collect();
        let Some(first) = given.first().cloned() else {
            unreachable!("a class is joined where a source gives something");
        };
        let joined = given
            .iter()
            .skip(1)
            .try_fold(first, |joined, source| self.head_join(&joined, source))
            .filter(|joined| {
                sources.part_lowers.iter().all(|(_, given)| {
                    given.node == Node::Owned(Constructor::Never) || given.node == joined.node
                })
            });
        let joined = joined.and_then(|joined| self.without_two_identities(sources, joined));
        let Some(joined) = joined else {
            self.classes.refuse(root, Cause::NoJoin);
            return Took::NoJoin;
        };
        // A reference's target is made with its node's structure only where
        // a source is raised by a cast to reach the join; otherwise the
        // targets entering are equated with it, a reference's target being
        // invariant.
        let raised = given.iter().any(|source| {
            source.node != joined.node && source.node != Node::Owned(Constructor::Never)
        });
        let owner = self.class_name(root);
        let recorded = self.take_skeleton(root, &joined, &owner, raised, &Recorder::Join(root));
        Took::Joined {
            head: joined,
            recorded,
        }
    }

    /// The equivalence class, which nothing enters, takes the meet of the
    /// classes it passes into, `uppers`, at its node with its parts made
    /// anew; where they have no meet it is refused.
    pub(crate) fn take_meet(&mut self, root: ClassId, uppers: &[Head]) -> Took {
        let root = self.root(root);
        let Some(met) = self.meet_of(uppers) else {
            self.classes.refuse(root, Cause::NoMeet);
            return Took::NoMeet;
        };
        let owner = self.class_name(root);
        let recorded = self.take_skeleton(root, &met, &owner, true, &Recorder::Join(root));
        Took::Met { head: met, recorded }
    }

    /// The meet of the upper bounds' heads, none where they are none or have
    /// no meet.
    pub(crate) fn meet_of(&self, uppers: &[Head]) -> Option<Head> {
        let first = uppers.first()?.clone();
        uppers
            .iter()
            .skip(1)
            .try_fold(first, |met, head| self.head_meet(&met, head))
    }

    /// The equivalence class, whose sources give it no node, takes the domain
    /// they give, recorded by its judgment.
    pub(crate) fn take_domain(&mut self, root: ClassId, domain: Domain) -> Took {
        let root = self.root(root);
        let mut recorded = Recording::default();
        let relation = Relation::Domain(root, domain.clone());
        recorded.push(self.record(relation, Recorder::Join(root)));
        Took::Bounded { domain, recorded }
    }

    /// The lower bounds and part lower bounds of the equivalence class, other
    /// than itself.
    pub(crate) fn lower_bounds(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.classes
            .occurrences(root)
            .iter()
            .filter(|occurrence| occurrence.side() == Some(Side::Upper))
            .filter_map(|occurrence| self.classes.items()[occurrence.at].relation())
            .filter_map(|relation| match relation {
                Relation::LowerBound { lower, .. } | Relation::PartLowerBound { lower, .. }
                    if self.root(*lower) != root =>
                {
                    Some(*lower)
                }
                _ => None,
            })
            .collect()
    }

    /// The classes on a path by which the equivalence class reaches a part
    /// of its own along recorded lower bounds, part lower bounds and parts,
    /// the class itself among them; empty where it reaches none. Every class
    /// on such a path reaches a part of its own, and no finite type holds
    /// any of them.
    fn own_part_cycle(&self, root: ClassId) -> Vec<ClassId> {
        let root = self.root(root);
        let mut seen: BTreeSet<(ClassId, bool)> = BTreeSet::new();
        let mut came_from: BTreeMap<(ClassId, bool), (ClassId, bool)> = BTreeMap::new();
        let mut work = vec![(root, false)];
        while let Some((at, through_part)) = work.pop() {
            let mut next: Vec<(ClassId, bool)> = self
                .lower_bounds(at)
                .into_iter()
                .map(|lower| (self.root(lower), through_part))
                .collect();
            if let Some((structure, _)) = &self.classes.held(at).structure {
                next.extend(
                    structure
                        .parts()
                        .into_iter()
                        .map(|(_, part)| (self.root(*part), true)),
                );
            }
            for step in next {
                if step.0 == root && step.1 {
                    let mut path = vec![root];
                    let mut back = (at, through_part);
                    while back.0 != root || back.1 {
                        path.push(back.0);
                        let Some(previous) = came_from.get(&back) else {
                            break;
                        };
                        back = *previous;
                    }
                    return path;
                }
                if seen.insert(step) {
                    came_from.insert(step, (at, through_part));
                    work.push(step);
                }
            }
        }
        Vec::new()
    }

    /// The closed type the equivalence class holds, where it has one.
    pub fn value(&self, id: ClassId) -> Option<Ty> {
        self.value_in(id, &mut BTreeSet::new())
    }

    fn value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Ty> {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return None;
        }
        let (structure, _) = self.classes.held(root).structure.clone()?;
        let term = structure
            .try_map(|_, part| self.value_in(part, visiting).ok_or(()))
            .ok()?;
        visiting.remove(&root);
        Ty::new(term).ok()
    }
}

/// The constructor of the member the structure constraint names, with the
/// members that constructor holds of it, whether visible or not: an object's
/// field, an enum's variant the text names, and the language's `Some`,
/// `None`, `Ok` and `Err`, whose constructor holds both its members.
fn named_member(structure_constraint: &StructureConstraint) -> Option<(Constructor, BTreeSet<Member>)> {
    if let Some((constructor, member)) = visible_member(structure_constraint) {
        return Some((constructor, [member].into()));
    }
    let StructureConstraint::Payload {
        enum_name: None,
        variant,
        ..
    } = structure_constraint
    else {
        return None;
    };
    let both = |a: &str, b: &str| -> BTreeSet<Member> {
        [a, b]
            .into_iter()
            .map(|name| Member::Variant(Name::new(name)))
            .collect()
    };
    match variant.as_str() {
        crate::ty::SOME | crate::ty::NONE => Some((Constructor::Option, both(crate::ty::SOME, crate::ty::NONE))),
        crate::ty::OK | crate::ty::ERR => Some((Constructor::Result, both(crate::ty::OK, crate::ty::ERR))),
        _ => None,
    }
}

/// The member the structure constraint names where the body's text writes
/// it, which is visible to this body: a field, or a variant of an enum the
/// text names.
fn visible_member(structure_constraint: &StructureConstraint) -> Option<(Constructor, Member)> {
    match structure_constraint {
        StructureConstraint::Field(field, _) => {
            Some((Constructor::Object, Member::Field(field.clone())))
        }
        StructureConstraint::Payload {
            enum_name: Some(enum_name),
            variant,
            ..
        } => Some((
            Constructor::Enum(enum_name.clone()),
            Member::Variant(variant.clone()),
        )),
        _ => None,
    }
}
