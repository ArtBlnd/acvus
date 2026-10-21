//! What all candidates of a use not yet decided share, recorded before it is
//! decided (l.252, l.335; `design/08-node.md`, readings 13 and 14).

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName};
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{Name, Term, Variance};

use super::{NodeAt, NodeKind};

/// Whether the type a signature writes is a reference, none where it is a
/// variable.
pub(super) fn is_reference(ty: &SigTy) -> Option<bool> {
    match ty {
        SigTy::Var(_) => None,
        SigTy::Term(term) => Some(matches!(**term, Term::Ref(..))),
    }
}

impl State {
    /// Records, for each undecided use with several candidates, what all its
    /// candidates share: each position their signatures write alike with no
    /// variable (l.335), and whether each class they would record a lower
    /// bound or an equation into is a reference, where they agree (l.252).
    pub(crate) fn record_at_once(&mut self) {
        let uses: Vec<Name> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter(|(_, node)| matches!(node.kind, NodeKind::Use { .. }))
            .map(|(name, _)| name.clone())
            .collect();
        for name in uses {
            self.record_shared(&name);
            self.record_references(&name);
        }
    }

    pub(super) fn record_shared(&mut self, name: &Name) {
        let node = self.nodes[name].clone();
        let NodeKind::Use {
            callee,
            args,
            result,
        } = &node.kind
        else {
            return;
        };
        let sigs: Vec<Signature> = self
            .candidates(callee, args.len())
            .into_iter()
            .map(|decl| decl.sig.clone())
            .collect();
        let Some(first) = sigs.first() else {
            return;
        };
        let differing = Self::differing(&sigs.iter().collect::<Vec<_>>(), args.len());
        let recorder = Recorder::Node(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        let var_name = |var| at(NodeAt::Var(var));
        for i in 0..=args.len() {
            let ty = Self::at_position(first, i).clone();
            if differing.contains(&i) || node.shared.contains(&i) || !ty.is_closed() {
                continue;
            }
            let mut vars = BTreeMap::new();
            match args.get(i) {
                Some(arg) => self.record_argument(
                    *arg,
                    &ty,
                    &mut vars,
                    at(NodeAt::Param(i)),
                    &var_name,
                    &recorder,
                ),
                None => self.record_result(
                    *result,
                    &ty,
                    &mut vars,
                    at(NodeAt::Result),
                    &var_name,
                    &recorder,
                ),
            }
            self.nodes
                .get_mut(name)
                .expect("the node is the state's")
                .shared
                .insert(i);
        }
    }

    pub(super) fn record_references(&mut self, name: &Name) {
        let per_candidate = self.would_record_per_candidate(name);
        let Some((first, rest)) = per_candidate.split_first() else {
            return;
        };
        let agreed: Vec<(ClassId, bool)> = first
            .iter()
            .filter_map(|(class, reference)| Some((*class, (*reference)?)))
            .filter(|(class, reference)| {
                rest.iter()
                    .all(|other| other.get(class) == Some(&Some(*reference)))
            })
            .filter(|(class, _)| !self.recorded_reference(name, *class))
            .collect();
        for (class, reference) in agreed {
            self.nodes
                .get_mut(name)
                .expect("the node is the state's")
                .references
                .insert(class);
            self.record(
                Relation::Reference(class, reference),
                Recorder::Node(name.clone()),
            );
        }
    }

    /// For each candidate of a use, the classes it would record a lower bound
    /// or an equation into, each with whether that type is a reference, none
    /// where it is an unknown: its result, and the classes inside its
    /// arguments' kinds that its parameters reach.
    pub(super) fn would_record_per_candidate(
        &self,
        name: &Name,
    ) -> Vec<BTreeMap<ClassId, Option<bool>>> {
        let NodeKind::Use {
            callee,
            args,
            result,
        } = &self.nodes[name].kind
        else {
            return Vec::new();
        };
        self.candidates(callee, args.len())
            .into_iter()
            .map(|decl| self.would_record_by(args, *result, &decl.sig))
            .collect()
    }

    /// The classes a candidate of signature `sig` would record a lower bound
    /// or an equation into, for a use of these arguments and result, as
    /// `would_record_per_candidate` gives them.
    pub(crate) fn would_record_by(
        &self,
        args: &[ClassId],
        result: ClassId,
        sig: &Signature,
    ) -> BTreeMap<ClassId, Option<bool>> {
        let mut found = BTreeMap::new();
        for (arg, param) in args.iter().zip(sig.params()) {
            self.would_record(*arg, param, false, &mut found);
        }
        found.insert(self.root(result), is_reference(sig.result()));
        found
    }

    /// The classes inside an argument's kind that a candidate's parameter
    /// would record a lower bound or an equation into, by the table of l.286:
    /// the parts an invariant part equates, and every part beneath them. A
    /// part the parameter writes as a variable is a lower bound of that
    /// unknown, recording into it only, unless an invariant part equates it.
    pub(super) fn would_record(
        &self,
        class: ClassId,
        ty: &SigTy,
        equated: bool,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        if equated {
            found.insert(self.root(class), is_reference(ty));
        }
        let SigTy::Term(term) = ty else {
            return;
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return;
        };
        if kind.constructor() != term.constructor() {
            return;
        }
        let invariant = kind.constructor().variance() == Variance::Invariant;
        for (key, written) in term.parts() {
            if let Some(&part) = kind.part(&key) {
                self.would_record(part, written, equated || invariant, found);
            }
        }
    }
}
