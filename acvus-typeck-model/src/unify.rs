//! Unification (l.149–170): the equivalence classes hold what a relation
//! records — an equation unifies two into one, whose kind is the conjunction
//! of theirs and whose parts of one name are unified in turn (l.165), and a
//! kind, a kind known in part, a LitVar (l.169–170), whether a class is a
//! reference, or a lambda its Values name is held by the class it is given to
//! — and a kind "indexes `τ`" an extern-defined container holds is linked to
//! the slice's element along the signature of its view (l.161).

use std::collections::BTreeMap;
use std::ops::Range;
use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, ClassName, Conflict, Fact, LinkAt};
use crate::decl::Node;
use crate::relation::{Named, Recorder, Relation};
use crate::sig::SigTy;
use crate::solve::State;
use crate::ty::{Mutability, Term};

/// A kind "indexes `τ`" an extern-defined container's class holds, linked to
/// the slice's element along the signature of its view (l.161): the relations
/// recorded, and the refusal where the container has no view.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Linked {
    pub container: ClassId,
    pub element: ClassId,
    pub recorded: Range<usize>,
    pub refused: Option<Refusal>,
}

impl State {
    /// The equivalence classes take what `relation` records, as it is recorded
    /// (l.230, l.165–170). What no type holds is refused at the relation, and
    /// a type the text writes that disagrees with its declaration is refused.
    /// Returns the refusal made.
    pub(crate) fn hold(&mut self, relation: &Relation) -> Option<Refusal> {
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        let given = match relation {
            Relation::Equation(a, b) => self.classes.unify(*a, *b, types),
            Relation::Fact(class, fact) => self.classes.give_fact(*class, fact.clone(), types),
            Relation::Kind(class, term, source) => match types.check(term) {
                Ok(()) => {
                    let given = self.classes.give_kind(*class, term.clone(), *source, types);
                    self.classes.mark_solved(*class);
                    given
                }
                Err(error) => {
                    let refusal = Refusal::Undeclared(format!("{error:?}"));
                    self.refusals.insert(refusal.clone());
                    return Some(refusal);
                }
            },
            Relation::Lit(class, lit) => self.classes.give_lit(*class, lit.clone(), types),
            Relation::Reference(class, reference) => {
                self.classes.give_reference(*class, *reference, types)
            }
            Relation::Values(class, lambda) => {
                self.classes.give_value(*class, lambda.clone());
                Ok(())
            }
            Relation::LowerBound { .. } | Relation::Boundary { .. } => Ok(()),
        };
        let Err(conflict) = given else {
            return None;
        };
        let at = Named {
            relation,
            arena: &self.arena,
        }
        .to_string();
        Some(self.refuse_conflict(at, conflict))
    }

    /// Links each kind "indexes `τ`" an extern-defined container's class holds
    /// to the slice's element along the signature of its view (l.161). A
    /// container with no view cannot have the kind. Returns each link made.
    pub(crate) fn link_views(&mut self) -> Vec<Linked> {
        let pending: Vec<(ClassId, ClassId)> = self
            .classes
            .roots()
            .flat_map(|root| {
                let held = self.classes.held(root);
                let container = matches!(held.kind, Some((Term::Extern { .. }, _)));
                held.facts.iter().filter_map(move |fact| match fact {
                    Fact::Indexes(element) if container => Some((root, *element)),
                    _ => None,
                })
            })
            .filter(|(_, element)| !self.viewed.contains(element))
            .collect();
        let mut linked = Vec::new();
        for (root, element) in pending {
            self.viewed.insert(element);
            linked.push(self.link_view(root, element));
        }
        linked
    }

    pub(crate) fn link_view(&mut self, root: ClassId, element: ClassId) -> Linked {
        let start = self.log.len();
        let linked = |state: &State, refused| Linked {
            container: root,
            element,
            recorded: start..state.log.len(),
            refused,
        };
        let recorder = Recorder::Split(root);
        let Some((kind, _)) = self.classes.held(root).kind.clone() else {
            return linked(self, None);
        };
        let constructor = kind.constructor();
        let view = self
            .decls
            .tree()
            .up(&Node::Ref(Mutability::Shared, constructor.clone()))
            .cloned()
            .filter(|cast| {
                matches!(self.decls.decl(cast).sig.result(), SigTy::Term(term)
                    if matches!(&**term, Term::Ref(_, SigTy::Term(target))
                        if matches!(&**target, Term::Slice(_))))
            });
        let Some(view) = view else {
            let at = format!(
                "{}: indexes {}",
                self.class_name(root),
                self.class_name(element)
            );
            let refused =
                self.refuse_conflict(at, Conflict::Fact(Fact::Indexes(element), constructor));
            return linked(self, Some(refused));
        };
        let sig = self.decls.decl(&view).sig.clone();
        let (from, to) = (
            Rc::new(self.class_name(root)),
            Rc::new(self.class_name(element)),
        );
        let link = |at: LinkAt| ClassName::Link {
            from: from.clone(),
            to: to.clone(),
            step: 0,
            at,
        };
        let mut vars = BTreeMap::new();
        let var_name = |var| link(LinkAt::Var(var));
        let param = self.instantiate(
            &sig.params()[0],
            &mut vars,
            link(LinkAt::Param),
            &var_name,
            &recorder,
        );
        let result = self.instantiate(
            sig.result(),
            &mut vars,
            link(LinkAt::Result),
            &var_name,
            &recorder,
        );
        let target = |state: &State, id: ClassId| match &state.classes.held(id).kind {
            Some((Term::Ref(_, target), _)) => Some(*target),
            _ => None,
        };
        let (Some(container), Some(slice)) = (target(self, param), target(self, result)) else {
            return linked(self, None);
        };
        let (Some((container_kind, _)), Some((Term::Slice(slice_element), _))) = (
            self.classes.held(container).kind.clone(),
            self.classes.held(slice).kind.clone(),
        ) else {
            return linked(self, None);
        };
        for ((_, part), (_, param_part)) in kind.parts().into_iter().zip(container_kind.parts()) {
            self.record(Relation::Equation(*part, *param_part), recorder.clone());
        }
        self.record(Relation::Equation(element, slice_element), recorder);
        linked(self, None)
    }
}
