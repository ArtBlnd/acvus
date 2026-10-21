//! What all candidates of a use not yet decided share, recorded before it is
//! decided (l.248, l.326; `design/08-node.md`, readings 13 and 14).

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName};
use crate::decl::Node;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{Constructor, Mutability, Name, Term};

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
    /// variable (l.326), and whether each class they would record a lower
    /// bound or an equation into is a reference, where they agree (l.248).
    /// Returns the relations recorded.
    pub(crate) fn record_at_once(&mut self) -> Recording {
        let mut recorded = Recording::default();
        let uses: Vec<Name> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter(|(_, node)| matches!(node.kind, NodeKind::Use { .. }))
            .map(|(name, _)| name.clone())
            .collect();
        for name in uses {
            recorded.append(self.record_shared(&name));
            recorded.append(self.record_references(&name));
        }
        recorded
    }

    pub(super) fn record_shared(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let node = self.nodes[name].clone();
        let NodeKind::Use {
            callee,
            args,
            result,
        } = &node.kind
        else {
            return recorded;
        };
        let sigs: Vec<Signature> = self
            .candidates(callee, args.len())
            .into_iter()
            .map(|decl| decl.sig.clone())
            .collect();
        let Some(first) = sigs.first() else {
            return recorded;
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
            recorded.append(match args.get(i) {
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
            });
            self.nodes
                .get_mut(name)
                .expect("the node is the state's")
                .shared
                .insert(i);
        }
        recorded
    }

    pub(super) fn record_references(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let per_candidate = self.would_record_per_candidate(name);
        let Some((first, rest)) = per_candidate.split_first() else {
            return recorded;
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
            let relation = Relation::Reference(class, reference);
            recorded.push(self.record(relation, Recorder::Node(name.clone())));
        }
        recorded
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
            self.would_record(*arg, param, &mut found);
        }
        found.insert(self.root(result), is_reference(sig.result()));
        found
    }

    /// The classes inside an argument's kind that a candidate's parameter
    /// would record a lower bound or an equation into, by the splits of
    /// l.280. The argument passes into the type the parameter writes, which
    /// records into that type alone, and each of its parts passes into that
    /// type's part: a meeting where the parameter writes a variable, an
    /// unknown (l.221), recording an equation into the part, unless it is a
    /// structural type's, whose part lower bound records into the variable
    /// alone; and where the parameter writes a type, the part passes into it
    /// in turn. Two constructors are linked as a split links them: the targets
    /// of a reborrow, and the parts of the first declared cast's parameter,
    /// into which the argument passes; the later casts' parameters and the
    /// parameter's type relate the casts' results, which are not the
    /// argument's.
    pub(super) fn would_record(
        &self,
        class: ClassId,
        ty: &SigTy,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        let SigTy::Term(term) = ty else {
            return;
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return;
        };
        let (from, to) = (kind.constructor(), term.constructor());
        if from == to {
            for (key, written) in term.parts() {
                if let Some(&part) = kind.part(&key) {
                    self.would_record_part(&to, part, written, found);
                }
            }
            return;
        }
        if let (Term::Ref(Mutability::Exclusive, target), Term::Ref(Mutability::Shared, written)) =
            (kind, &**term)
        {
            self.would_record_part(&to, *target, written, found);
            return;
        }
        let path = self.decls.tree().path(&Node::Owned(from), &Node::Owned(to));
        if let Some(cast) = path.as_deref().and_then(<[_]>::first) {
            let param = &self.decls.decl(cast).sig.params()[0];
            self.would_record(class, param, found);
        }
    }

    /// As `would_record`, at the part `part` of an argument passing into the
    /// part `written` of a type of constructor `of`.
    fn would_record_part(
        &self,
        of: &Constructor,
        part: ClassId,
        written: &SigTy,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        match written {
            SigTy::Var(_) if !of.is_structural() => {
                found.insert(self.root(part), is_reference(written));
            }
            SigTy::Var(_) => {}
            SigTy::Term(_) => self.would_record(part, written, found),
        }
    }
}
