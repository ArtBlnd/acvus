//! Uses and lends: the positions each lies at, what each reads, and what each
//! records once decided (l.303–345).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Fact};
use crate::coerce::coerce_read;
use crate::decl::DeclId;
use crate::read::ReadValue;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{HasTerm, Mutability, Name, PartKey, Term};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NodeKind {
    /// A use that calls the declarations `callee` names (l.331).
    Use {
        callee: Name,
        args: Vec<ClassId>,
        result: ClassId,
    },
    /// `&ₘ p` (l.307).
    Lend {
        mutability: Mutability,
        place: ClassId,
        result: ClassId,
    },
}

/// What a use or a lend became (l.267).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decision {
    /// The use is decided as this candidate (l.344).
    Use(DeclId),
    /// The use leaves no candidate, or several, and is refused (l.344).
    Unreached(Vec<DeclId>),
    /// The lend's place is not a reference in the value it reads, so its
    /// target is the place (l.307).
    LendPlace,
    /// The lend's place is a reference of this mutability in the value it
    /// reads, so its target is that reference's (l.307).
    LendThrough(Mutability),
}

impl Decision {
    /// Whether the node is reached by what it reads, and records.
    pub fn is_reached(&self) -> bool {
        !matches!(self, Decision::Unreached(_))
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NodeState {
    pub kind: NodeKind,
    pub decision: Option<Decision>,
}

/// Which class a node made anew is (l.331).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum NodeAt {
    /// A variable of the signature of the candidate the use is decided as.
    Var(crate::sig::SigVar),
    /// The type of the parameter at this place.
    Param(usize),
    Result,
    /// The reference a lend gives.
    Lent,
    /// The target of the reference a lend's place is.
    Target,
}

impl State {
    /// The classes at a node's positions, and those within the depth its
    /// candidates reach (l.244 (b)).
    pub(crate) fn positions(&self, kind: &NodeKind) -> BTreeSet<ClassId> {
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
            } => {
                let mut positions: BTreeSet<ClassId> =
                    args.iter().chain([result]).copied().collect();
                for candidate in self.decls.candidates(callee) {
                    let sig = &candidate.sig;
                    for (arg, param) in args.iter().zip(sig.params()) {
                        positions.extend(self.within_depth(*arg, param));
                    }
                    positions.extend(self.within_depth(*result, sig.result()));
                }
                positions
            }
            NodeKind::Lend { place, result, .. } => [*place, *result].into(),
        }
    }

    /// The classes at the positions inside `class`'s kind where `ty` writes a
    /// term, down to where it writes a variable.
    fn within_depth(&self, class: ClassId, ty: &SigTy) -> BTreeSet<ClassId> {
        let mut found = BTreeSet::new();
        let SigTy::Term(term) = ty else {
            return found;
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return found;
        };
        for (key, written) in term.parts() {
            if let (SigTy::Term(_), Some(&part)) = (written, kind.part(&key)) {
                found.insert(self.root(part));
                found.extend(self.within_depth(part, written));
            }
        }
        found
    }

    /// The positions where a use's candidates differ, which it reads (l.267):
    /// each argument and the result, by their index, the result last.
    fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        (0..=args)
            .filter(|&i| {
                let at = |sig: &Signature| {
                    if i < args {
                        sig.params().get(i).cloned()
                    } else {
                        Some(sig.result().clone())
                    }
                };
                let first = at(candidates[0]);
                candidates.iter().any(|sig| at(sig) != first)
            })
            .collect()
    }

    /// The classes at the positions a node reads: for a use, those where its
    /// candidates differ; for a lend, its place.
    pub(crate) fn read_positions(&self, kind: &NodeKind) -> Vec<ClassId> {
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
            } => {
                let candidates: Vec<&Signature> = self
                    .decls
                    .candidates(callee)
                    .iter()
                    .map(|decl| &decl.sig)
                    .collect();
                if candidates.len() < 2 {
                    return Vec::new();
                }
                Self::differing(&candidates, args.len())
                    .into_iter()
                    .map(|i| if i < args.len() { args[i] } else { *result })
                    .collect()
            }
            NodeKind::Lend { place, .. } => vec![*place],
        }
    }

    /// Decides the node by the value it reads, and records what it then
    /// records.
    pub(crate) fn decide(&mut self, name: &Name) {
        let kind = self.nodes[name].kind.clone();
        let excluded = self.dependents(name);
        let decision = match &kind {
            NodeKind::Use {
                callee,
                args,
                result,
            } => self.narrow(callee, args, *result, &excluded),
            NodeKind::Lend { place, .. } => {
                let read = self.read_state(&excluded);
                match read.read_value(*place).decided() {
                    Some(Term::Ref(mutability, _)) => Decision::LendThrough(*mutability),
                    _ => Decision::LendPlace,
                }
            }
        };
        self.nodes
            .get_mut(name)
            .expect("the node is the state's")
            .decision = Some(decision.clone());
        let recorder = Recorder::Node(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        match (&kind, &decision) {
            (NodeKind::Use { args, result, .. }, Decision::Use(id)) => {
                let sig = self.decls.decl(id).sig.clone();
                let mut vars = BTreeMap::new();
                let var_name = |var| at(NodeAt::Var(var));
                for (i, (arg, param)) in args.iter().zip(sig.params()).enumerate() {
                    let param = self.instantiate(
                        param,
                        &mut vars,
                        at(NodeAt::Param(i)),
                        &var_name,
                        &recorder,
                    );
                    let relation = if self.is_unknown(param) {
                        Relation::LowerBound {
                            lower: *arg,
                            upper: param,
                        }
                    } else {
                        Relation::Boundary {
                            from: *arg,
                            to: param,
                        }
                    };
                    self.record(relation, recorder.clone());
                }
                let returned = self.instantiate(
                    sig.result(),
                    &mut vars,
                    at(NodeAt::Result),
                    &var_name,
                    &recorder,
                );
                self.record(
                    Relation::LowerBound {
                        lower: returned,
                        upper: *result,
                    },
                    recorder,
                );
            }
            (
                NodeKind::Lend {
                    mutability,
                    place,
                    result,
                },
                Decision::LendPlace,
            ) => {
                let lent =
                    self.written(at(NodeAt::Lent), Term::Ref(*mutability, *place), &recorder);
                self.record(Relation::Equation(*result, lent), recorder);
            }
            (
                NodeKind::Lend {
                    mutability,
                    place,
                    result,
                },
                Decision::LendThrough(through),
            ) => {
                let target = self.arena.intern(at(NodeAt::Target));
                self.classes.grow(&self.arena);
                self.record(Relation::Fact(*place, Fact::Ref(target)), recorder.clone());
                let lent = self.written(
                    at(NodeAt::Lent),
                    Term::Ref((*mutability).min(*through), target),
                    &recorder,
                );
                self.record(Relation::Equation(*result, lent), recorder);
            }
            _ => {}
        }
    }

    /// Narrows the candidates by the value the use reads at each position
    /// where they differ, then by the result (l.337–344).
    fn narrow(
        &self,
        callee: &Name,
        args: &[ClassId],
        result: ClassId,
        excluded: &BTreeSet<Name>,
    ) -> Decision {
        let candidates = self.decls.candidates(callee);
        let ids: Vec<DeclId> = candidates.iter().map(|decl| decl.id.clone()).collect();
        if candidates.len() == 1 {
            return Decision::Use(ids[0].clone());
        }
        let sigs: Vec<&Signature> = candidates.iter().map(|decl| &decl.sig).collect();
        let differing = Self::differing(&sigs, args.len());
        let read = self.read_state(excluded);
        let values: Vec<(usize, ReadValue)> = differing
            .iter()
            .map(|&i| {
                let class = if i < args.len() { args[i] } else { result };
                (i, read.read_value(class))
            })
            .collect();
        let narrowed = |values: &[(usize, ReadValue)]| {
            let mut remaining: Vec<usize> = (0..sigs.len()).collect();
            for (i, value) in values {
                let ranks: Vec<(usize, Option<crate::coerce::Rank>)> = remaining
                    .iter()
                    .map(|&c| {
                        let rank = if *i < args.len() {
                            let to = ReadValue::of_sig(&sigs[c].params()[*i]);
                            coerce_read(&self.decls, value, &to)
                        } else {
                            let from = ReadValue::of_sig(sigs[c].result());
                            coerce_read(&self.decls, &from, value)
                        };
                        (c, rank.ok().and_then(|coercion| coercion.rank()))
                    })
                    .collect();
                let lowest = ranks.iter().filter_map(|(_, rank)| rank.clone()).min();
                remaining = ranks
                    .into_iter()
                    .filter(|(_, rank)| rank.is_some() && *rank == lowest)
                    .map(|(c, _)| c)
                    .collect();
            }
            remaining
        };
        let mut remaining = narrowed(&values);
        if remaining.len() > 1 && values.iter().any(|(_, value)| value.has_lit()) {
            let defaulted: Vec<(usize, ReadValue)> = values
                .iter()
                .map(|(i, value)| (*i, value.with_defaults()))
                .collect();
            remaining = narrowed(&defaulted);
        }
        match remaining.as_slice() {
            [one] => Decision::Use(ids[*one].clone()),
            _ => Decision::Unreached(remaining.into_iter().map(|c| ids[c].clone()).collect()),
        }
    }

    /// A class made anew that holds `term` as a declaration writes it.
    fn written(&mut self, name: ClassName, term: Term<ClassId>, recorder: &Recorder) -> ClassId {
        let class = self.arena.intern(name);
        self.classes.grow(&self.arena);
        self.decided.insert(class);
        self.record(Relation::Kind(class, term), recorder.clone());
        class
    }

    /// The classes of a type a declaration writes: the type named `name`, its
    /// parts named after it, and its variables those of `vars`, each made
    /// where first met and named by `var_name`. A variable is an unknown, and
    /// a written term is decided (l.218).
    pub(crate) fn instantiate<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) -> ClassId
    where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        match ty {
            SigTy::Var(var) => {
                if let Some(&class) = vars.get(var) {
                    return class;
                }
                let class = self.arena.intern(var_name(*var));
                self.classes.grow(&self.arena);
                vars.insert(*var, class);
                class
            }
            SigTy::Term(term) => {
                let shared = Rc::new(name.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    let part_name = ClassName::Part(shared.clone(), key);
                    self.instantiate(&part, vars, part_name, var_name, recorder)
                });
                self.written(name, term, recorder)
            }
        }
    }
}
