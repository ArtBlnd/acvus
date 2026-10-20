//! Invariant points and the invariant rule (`design/18-invariant-points.md`):
//! a class takes its type from its own relations first, and where they give
//! it none, from the one closed type its invariant points give that fits it.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{Arena, ClassId, ClassName, Classes, KindSource};
use crate::dep::Vertex;
use crate::lit::LitVar;
use crate::node::{Decision, NodeKind};
use crate::read::ReadValue;
use crate::relation::{Recorder, Relation};
use crate::sig::{DeclaredTypes, SigTy, Signature};
use crate::solve::{Refusal, State};
use crate::ty::{Mutability, Name, Term, Ty, Variance};

/// A use not decided as one candidate, whose candidates each give the
/// invariant points they would record: all of an undecided use's, and the
/// remaining ones of a use left unreached (note 08, reading 13).
pub(crate) struct Pending<'a> {
    args: &'a [ClassId],
    sigs: Vec<&'a Signature>,
}

/// The closed types each class's invariant points give it, by its root.
pub(crate) type Points = BTreeMap<ClassId, BTreeSet<Ty>>;

/// Which step of the invariant rule applies to a class (row 3).
enum Taken {
    /// Exactly one of its candidates fits it.
    One(Ty),
    /// Several fit, its LitVar's default among them.
    Default(Ty),
    /// Several fit otherwise.
    Several,
}

impl State {
    /// The uses not decided as one candidate.
    pub(crate) fn pending_uses(&self) -> Vec<Pending<'_>> {
        self.nodes
            .values()
            .filter_map(|node| {
                let NodeKind::Use { callee, args, .. } = &node.kind else {
                    return None;
                };
                let sigs = match &node.decision {
                    None => self
                        .candidates(callee, args.len())
                        .into_iter()
                        .map(|decl| &decl.sig)
                        .collect(),
                    Some(Decision::Unreached(remaining)) => remaining
                        .iter()
                        .map(|id| &self.decls.decl(id).sig)
                        .collect(),
                    Some(_) => return None,
                };
                Some(Pending { args, sigs })
            })
            .collect()
    }

    /// The types the invariant points of each class not yet solved give it:
    /// those recorded, a LitVar's boundaries into a closed type among them,
    /// and those each candidate of each use of `pending` would record, read
    /// in this state's classes.
    pub(crate) fn invariant_points(&self, pending: &[Pending<'_>]) -> Points {
        let mut points = Points::new();
        for recorded in &self.log {
            let (class, closed) = match recorded.relation {
                Relation::Invariant { class, closed } => (class, closed),
                Relation::Boundary { from, to }
                    if self.is_lit(from) && self.is_invariant_point_target(to) =>
                {
                    (from, to)
                }
                _ => continue,
            };
            let root = self.root(class);
            if self.is_solved(root) {
                continue;
            }
            if let Some(ty) = self.value(closed) {
                points.entry(root).or_default().insert(ty);
            }
        }
        for use_ in pending {
            for sig in &use_.sigs {
                for (arg, param) in use_.args.iter().zip(sig.params()) {
                    self.would_give(*arg, param, false, &mut points);
                }
            }
        }
        points
    }

    /// The invariant points a candidate's parameter type would give the
    /// classes in the argument's kind, as `split_parts` and l.291 record them
    /// once both sides are solved: where an invariant part, or a LitVar's
    /// class, meets a closed type.
    fn would_give(&self, class: ClassId, ty: &SigTy, equated: bool, points: &mut Points) {
        let SigTy::Term(term) = ty else {
            return;
        };
        let root = self.root(class);
        if !self.is_solved(root) {
            if (equated || self.is_lit(root))
                && let Some(closed) = ReadValue::of_sig(ty).to_ty()
            {
                points.entry(root).or_default().insert(closed);
            }
            return;
        }
        if equated && !ty.is_closed() {
            return;
        }
        let Some((kind, _)) = &self.classes.held(root).kind else {
            return;
        };
        if kind.constructor() == term.constructor() {
            let invariant = equated || kind.constructor().variance() == Variance::Invariant;
            for (key, written) in term.parts() {
                if let Some(&part) = kind.part(&key) {
                    self.would_give(part, written, invariant, points);
                }
            }
        } else if let (
            false,
            Term::Ref(Mutability::Exclusive, target),
            Term::Ref(Mutability::Shared, written),
        ) = (equated, kind, &**term)
        {
            self.would_give(*target, written, true, points);
        }
    }

    /// Whether the class is one the invariant rule may still type: not
    /// solved, refused or of an enclosing group, with an invariant point
    /// whose type fits it.
    pub(crate) fn is_open(&self, class: ClassId, points: &Points) -> bool {
        let root = self.root(class);
        !self.is_solved(root)
            && !self.has_no_value(root)
            && !self.is_foreign(root)
            && points
                .get(&root)
                .is_some_and(|types| types.iter().any(|ty| self.fits(root, ty)))
    }

    /// Whether the class may be unified with the closed type: by its kind,
    /// its LitVar's family and values (l.169), and what it holds besides,
    /// tried on a copy of the classes.
    fn fits(&self, root: ClassId, ty: &Ty) -> bool {
        let mut arena = self.arena.clone();
        let mut classes = self.classes.clone();
        let types = self.decls.types();
        let name = ClassName::InvariantRule(Rc::new(self.class_name(root)));
        trial_class(&mut arena, &mut classes, name, ty, types)
            .is_some_and(|made| classes.unify(root, made, types).is_ok())
    }

    /// The invariant rule at a binding group's fixpoint (row 3): of the
    /// classes not solved, with invariant points, and with no unbroken
    /// dependency once each node depends on the classes it waits for, those
    /// the first step that applies to some of them takes, all at once, from
    /// this state. Returns whether it took any.
    pub fn apply_invariant_rule(&mut self) -> bool {
        if !self.keeps_invariant_points() {
            return false;
        }
        let taken = self.taken_by_rule();
        let unique: Vec<(ClassId, Ty)> = taken
            .iter()
            .filter_map(|(root, taken)| match taken {
                Taken::One(ty) => Some((*root, ty.clone())),
                Taken::Default(_) | Taken::Several => None,
            })
            .collect();
        let defaults: Vec<(ClassId, Ty)> = taken
            .iter()
            .filter_map(|(root, taken)| match taken {
                Taken::Default(ty) => Some((*root, ty.clone())),
                Taken::One(_) | Taken::Several => None,
            })
            .collect();
        let typed = if unique.is_empty() { defaults } else { unique };
        if !typed.is_empty() {
            for (root, ty) in typed {
                self.type_by_rule(root, &ty);
            }
            self.settle();
            return true;
        }
        let ambiguous: Vec<ClassId> = taken.into_keys().collect();
        for &root in &ambiguous {
            let name = self.class_name(root);
            let members = self.classes.held(root).members.clone();
            self.no_value.extend(members);
            self.refusals.insert(Refusal::Ambiguous(name));
        }
        !ambiguous.is_empty()
    }

    /// The classes the rule may take now, each with the step that applies
    /// to it. What a class's own relations give it so far, its invariant
    /// points, and which types fit it, are read in the value read leaving
    /// out nothing, which solves each class by its sources whatever its
    /// dependencies, as the nodes read them. A class no candidate fits is
    /// not taken, and is left to fallback.
    fn taken_by_rule(&self) -> BTreeMap<ClassId, Taken> {
        let whole = self.read_state(&BTreeSet::new());
        let points = whole.invariant_points(&self.pending_uses());
        let waits: Vec<(Name, BTreeSet<ClassId>)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter_map(|(name, _)| {
                let waits = self.decision_or_waits(name, &whole, &points).err()?;
                Some((name.clone(), waits))
            })
            .collect();
        let graph = self.graph_waiting(&waits);
        let settled = self.settled_in(&graph);
        let roots: Vec<ClassId> = self.classes.roots().collect();
        let mut taken = BTreeMap::new();
        for root in roots {
            if self.is_solved(root)
                || self.has_no_value(root)
                || self.is_foreign(root)
                || !self.is_unblocked_in(&graph, &Vertex::Class(root), &settled)
            {
                continue;
            }
            let at = whole.root(root);
            if whole.is_solved(at) || whole.has_no_value(at) {
                continue;
            }
            let Some(types) = points.get(&at) else {
                continue;
            };
            let fitting: Vec<&Ty> = types.iter().filter(|ty| whole.fits(at, ty)).collect();
            let lit: Option<&LitVar> = whole.classes.held(at).lit.as_ref();
            let default = lit.map(|lit| Ty::new(Term::Scalar(lit.default_scalar())));
            let step = match fitting.as_slice() {
                [] => continue,
                [one] => Taken::One((*one).clone()),
                _ => match default {
                    Some(Ok(default)) if fitting.contains(&&default) => Taken::Default(default),
                    _ => Taken::Several,
                },
            };
            taken.insert(root, step);
        }
        taken
    }

    /// Unifies the class with a class made anew holding the closed type, and
    /// keeps that the rule typed it, so that only that type and `!` reach it
    /// (row 5).
    fn type_by_rule(&mut self, root: ClassId, ty: &Ty) {
        let name = ClassName::InvariantRule(Rc::new(self.class_name(root)));
        let made = self.closed(name, ty, &Recorder::Invariant);
        self.ruled.insert(root);
        self.record(Relation::Equation(root, made), Recorder::Invariant);
        self.classes.mark_solved(root);
    }
}

/// A class of `arena` made anew that holds the closed type `ty`, its parts
/// named after it, where `classes` gives every part its kind.
fn trial_class(
    arena: &mut Arena,
    classes: &mut Classes,
    name: ClassName,
    ty: &Ty,
    types: &DeclaredTypes,
) -> Option<ClassId> {
    let shared = Rc::new(name.clone());
    let term = ty
        .term()
        .clone()
        .try_map(|key, part| {
            trial_class(
                arena,
                classes,
                ClassName::Part(shared.clone(), key),
                &part,
                types,
            )
            .ok_or(())
        })
        .ok()?;
    let class = arena.intern(name);
    classes.grow(arena);
    classes
        .give_kind(class, term, KindSource::Written, types)
        .ok()?;
    Some(class)
}
