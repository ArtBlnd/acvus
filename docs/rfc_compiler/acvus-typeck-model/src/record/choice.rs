//! A use of declarations and its choice of a candidate: one record per use,
//! kept among the record's items, and what the use records once its choice
//! admits one candidate, or once an argument of it is `!`.
//!
//! The choice is no decision: the record keeps the candidates that remain
//! (`Choice::remaining`), which only shrink as the types the use's
//! variables will take are read (`solve::narrow`), and what the use became
//! is read off it (`State::chosen`).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, StructureSource};
use crate::decl::DeclId;
use crate::disjunction::DisjunctionAt;
use crate::disjunction::generalization::Generalization;
use crate::program::Position;
use crate::record::{Entry, Making, Owner, Recording};
use crate::relation::{Item, Recorder, Relation, Role};
use crate::sig::{SigTy, SigVar, Signature};
use crate::solve::State;
use crate::ty::{HasTerm, Name, PartKey, Term};
use crate::unify::Unification;

/// A use of declarations: the name it calls, and the class at each of its
/// positions this group records, none at a position another group records.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Use {
    pub callee: Name,
    pub args: Vec<Option<ClassId>>,
    pub result: Option<ClassId>,
    /// For `e?`, the class its exit's value passes into.
    pub exit: Option<ClassId>,
}

impl Use {
    /// The classes at the use's positions, each with its position's role.
    pub fn occurrences(&self) -> Vec<(ClassId, Role)> {
        self.args
            .iter()
            .enumerate()
            .filter_map(|(k, arg)| Some(((*arg)?, Role::Position(Position::Arg(k)))))
            .chain(
                self.result
                    .iter()
                    .map(|&result| (result, Role::Position(Position::Result))),
            )
            .chain(
                self.exit
                    .iter()
                    .map(|&exit| (exit, Role::Position(Position::Exit))),
            )
            .collect()
    }

    /// The use with each class at its positions put through `f`.
    pub fn map<F>(self, mut f: F) -> Use
    where
        F: FnMut(ClassId) -> ClassId,
    {
        Use {
            callee: self.callee,
            args: self.args.into_iter().map(|arg| arg.map(&mut f)).collect(),
            result: self.result.map(&mut f),
            exit: self.exit.map(&mut f),
        }
    }
}

/// A use's choice of a candidate: the declarations its name reaches with as
/// many parameters as it has arguments, in the order declared, and the
/// indices of those that remain. `remaining` only shrinks.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Choice {
    pub candidates: Vec<DeclId>,
    pub remaining: BTreeSet<usize>,
}

impl Choice {
    /// A choice admitting every one of `candidates`.
    pub fn of(candidates: Vec<DeclId>) -> Choice {
        let remaining = (0..candidates.len()).collect();
        Choice {
            candidates,
            remaining,
        }
    }

    /// The one candidate the choice admits, where it admits one.
    pub fn one(&self) -> Option<usize> {
        match self.remaining.iter().collect::<Vec<_>>().as_slice() {
            [one] => Some(**one),
            _ => None,
        }
    }
}

/// What a use became, read off its choice and its arguments; never stored.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Chosen {
    /// The one candidate its choice admits.
    Candidate(DeclId),
    /// Its choice admits several candidates and an argument's value is `!`:
    /// the use is never carried out, and its result is `!` (the never rule).
    Never,
}

impl State {
    /// Records the use `name`, `of`, with its choice `choice`. Recording it
    /// relates no class: what it records is recorded apart, by its
    /// generalization (`record_generalization`) and by its one candidate
    /// (`record_candidate`).
    pub(crate) fn add_choice(
        &mut self,
        name: &Name,
        of: Use,
        choice: Choice,
        recorder: Recorder,
    ) -> Entry {
        let item = Item::Choice {
            name: name.clone(),
            of,
            choice,
        };
        let at = self.classes.record(item, recorder);
        Entry {
            at,
            unification: Unification::default(),
        }
    }

    /// The place in the record of the use of this name.
    pub(crate) fn choice_at(&self, name: &Name) -> usize {
        self.classes
            .choice_at(name)
            .unwrap_or_else(|| panic!("the record has the use {name}"))
    }

    /// The use of this name, and its choice.
    pub(crate) fn choice(&self, name: &Name) -> (&Use, &Choice) {
        let at = self.choice_at(name);
        let (_, of, choice) = self.classes.items()[at]
            .choice()
            .expect("the place of a use holds its choice");
        (of, choice)
    }

    /// The use of this name as the maker of classes.
    pub(crate) fn use_owner(&self, name: &Name) -> Owner {
        Owner {
            item: self.choice_at(name),
            name: Rc::new(name.clone()),
        }
    }

    /// The signatures of a choice's candidates, in their order.
    pub(crate) fn signatures(&self, choice: &Choice) -> Vec<Signature> {
        choice
            .candidates
            .iter()
            .map(|id| self.decls.decl(id).sig.clone())
            .collect()
    }

    /// The never rule's test: whether an argument of the use `name`, `of`,
    /// reads as `!`. Each argument is read as narrowing reads a position
    /// (`State::read`), without what the use itself records and what the
    /// disjunctive constraints and uses in `left_out` record, so that an
    /// argument whose bounds decide `!` reads `!` before it holds a
    /// structure. A read that already leaves out `left_out` passes it on, so
    /// that the reads a use's test makes from inside another reading leave
    /// out one more use each, and end. It records nothing and decides
    /// nothing.
    pub(crate) fn never_argument(&self, name: &Name, of: &Use, left_out: &BTreeSet<Name>) -> bool {
        let mut left_out = left_out.clone();
        left_out.insert(name.clone());
        of.args
            .iter()
            .flatten()
            .any(|&arg| matches!(self.read(arg, &left_out).decided(), Some(Term::Never)))
    }

    /// What the use `name`, `of`, became, read off `choice` and, through the
    /// never rule's test (`never_argument`), off its arguments read without
    /// what `left_out` records: its one candidate where the choice admits
    /// one; `Never` where it admits several and an argument reads as `!`;
    /// none otherwise. It records nothing.
    pub(crate) fn chosen_of(
        &self,
        name: &Name,
        of: &Use,
        choice: &Choice,
        left_out: &BTreeSet<Name>,
    ) -> Option<Chosen> {
        if let Some(one) = choice.one() {
            return Some(Chosen::Candidate(choice.candidates[one].clone()));
        }
        (choice.remaining.len() > 1 && self.never_argument(name, of, left_out))
            .then_some(Chosen::Never)
    }

    /// What the use of this name became (`chosen_of`), reading every
    /// relation.
    pub(crate) fn chosen(&self, name: &Name) -> Option<Chosen> {
        let (of, choice) = self.choice(name);
        self.chosen_of(name, of, choice, &BTreeSet::new())
    }

    /// Every use's name and what it became (`chosen_of`), reading every
    /// relation.
    pub(crate) fn choices(&self) -> BTreeMap<Name, Option<Chosen>> {
        self.classes
            .choices()
            .map(|(_, name, of, choice)| {
                let chosen = self.chosen_of(name, of, choice, &BTreeSet::new());
                (name.clone(), chosen)
            })
            .collect()
    }

    /// The roots of the variables of the use's generalization, in the
    /// variables' order; none for a use of fewer than two candidates.
    pub(crate) fn generalized(&self, name: &Name) -> Vec<ClassId> {
        let at = self.choice_at(name);
        let mut found: Vec<(usize, ClassId)> = self
            .classes
            .roles_of(at)
            .into_iter()
            .filter_map(|(class, role)| match role {
                Role::Made(DisjunctionAt::Generalized(k)) => Some((k, self.root(class))),
                _ => None,
            })
            .collect();
        found.sort();
        found.into_iter().map(|(_, class)| class).collect()
    }

    /// The never rule's relation, recorded once where the choice of the use
    /// `name` admits several candidates and an argument reads as `!`
    /// (`never_argument`, the test `chosen_of` takes too, which the caller
    /// takes before it): the use's result is `!`, a written `!` entering it.
    /// Records nothing where the use holds no result here or the relation is
    /// recorded already.
    pub(crate) fn record_never(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let at = self.choice_at(name);
        let (of, _) = self.choice(name);
        let Some(result) = of.result else {
            return recorded;
        };
        let made = Role::Made(DisjunctionAt::Result);
        if self.classes.in_role(at, &made).is_some() {
            return recorded;
        }
        let recorder = Recorder::Choice(name.clone());
        let owner = self.use_owner(name);
        let never = self.written(
            owner.made(DisjunctionAt::Result),
            Term::Never,
            &recorder,
            &mut recorded,
        );
        let relation = Relation::LowerBound {
            lower: never,
            upper: result,
        };
        recorded.push(self.record(relation, recorder));
        recorded
    }

    /// Records the signature of the one candidate the choice of the use
    /// `name` admits, its variables classes made anew (`Var`). Of several
    /// candidates, at the variables of their generalization, which holds the
    /// rest at every position: the candidate's type at each variable
    /// (`Chosen(k)`), into which the variable's value passes where a
    /// parameter writes it, as an argument into its parameter, and which
    /// enters the variable where the result writes it. Of one candidate, at
    /// its positions: each argument passing whole into its parameter, a
    /// coercion being of a whole value, and its result type entering the
    /// use's result. Then the identity it makes anew, where it makes one, and
    /// `e?`'s exit. Called once, where the choice comes to admit one
    /// candidate, or as a use of one candidate is recorded.
    pub(crate) fn record_candidate(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let (of, choice) = self.choice(name);
        let (of, choice) = (of.clone(), choice.clone());
        let one = choice
            .one()
            .expect("a candidate is recorded where its choice admits it alone");
        let sigs = self.signatures(&choice);
        let sig = sigs[one].clone();
        let recorder = Recorder::Choice(name.clone());
        let owner = self.use_owner(name);
        let var_name = |var| owner.made(DisjunctionAt::Var(var));
        let mut vars = BTreeMap::new();
        if sigs.len() >= 2 {
            let part = Generalization::of(&sigs.iter().collect::<Vec<_>>());
            for (k, variable) in self.generalized(name).into_iter().enumerate() {
                let written = &part.vars[k][one];
                let chosen = self.instantiate(
                    written,
                    &mut vars,
                    owner.made(DisjunctionAt::Chosen(k)),
                    &var_name,
                    &recorder,
                    &mut recorded,
                );
                if part.writes_in_params(k) {
                    let relation = if self.is_unknown(chosen) {
                        Relation::LowerBound {
                            lower: variable,
                            upper: chosen,
                        }
                    } else {
                        Relation::CoercionConstraint {
                            from: variable,
                            to: chosen,
                        }
                    };
                    recorded.push(self.record(relation, recorder.clone()));
                }
                if part.writes_in_result(k) {
                    let relation = Relation::LowerBound {
                        lower: chosen,
                        upper: variable,
                    };
                    recorded.push(self.record(relation, recorder.clone()));
                }
            }
        } else {
            for (i, (arg, param)) in of.args.iter().zip(sig.params()).enumerate() {
                let Some(arg) = arg else {
                    continue;
                };
                recorded.append(self.record_argument(
                    *arg,
                    param,
                    &mut vars,
                    owner.made(DisjunctionAt::Param(i)),
                    &var_name,
                    &recorder,
                ));
            }
            if let Some(result) = of.result {
                recorded.append(self.record_result(
                    result,
                    sig.result(),
                    &mut vars,
                    owner.made(DisjunctionAt::Result),
                    &var_name,
                    &recorder,
                ));
            }
        }
        if sig.existential_identity()
            && let Some(&identity) = vars.get(&SigVar::Identity)
        {
            // The candidate's signature writes the identity it makes anew,
            // as it writes the rest of its result: no join of lower bounds
            // gives it, so no union drops it.
            let made = Term::Identity(Name::new(&format!("new {name}")));
            let relation = Relation::Structure(identity, made, StructureSource::Written);
            recorded.push(self.record(relation, recorder.clone()));
        }
        if let Some(exit) = of.exit {
            recorded.append(self.record_exit(
                &sig,
                &vars,
                owner.made(DisjunctionAt::Exit),
                exit,
                &recorder,
            ));
        }
        recorded
    }

    /// `e?`'s exit, its use's choice admitting the candidate of `sig` alone,
    /// whose variables are the classes of `vars`: as `return None`, a value
    /// `Option⟨K⟩`, or as `return Err(x)`, a value `Result⟨O, X⟩`, `K` and `O`
    /// made anew and the candidate's `E` wrapped into `X`. The value passes
    /// into `exit`, the class the text gives it, a lower bound where that is
    /// an unknown.
    fn record_exit(
        &mut self,
        sig: &Signature,
        vars: &BTreeMap<SigVar, ClassId>,
        making: Making,
        exit: ClassId,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        let named = Rc::new(making.label.clone());
        let part = |key: PartKey| ClassName::Part(named.clone(), key);
        let some = PartKey::Payload(Name::new(crate::ty::SOME));
        let ok = PartKey::Payload(Name::new(crate::ty::OK));
        let err = PartKey::Payload(Name::new(crate::ty::ERR));
        let term = match sig.params() {
            [SigTy::Term(param)] if matches!(**param, Term::Option(_)) => {
                Term::Option(self.fresh(part(some)))
            }
            [SigTy::Term(param)] if matches!(**param, Term::Result { .. }) => {
                let error = *vars
                    .get(&SigVar::Type(1))
                    .expect("the argument made the candidate's `E`");
                let wrapped = self.fresh(part(err));
                let relation = Relation::Wrap {
                    from: error,
                    to: wrapped,
                };
                recorded.push(self.record(relation, recorder.clone()));
                Term::Result {
                    ok: self.fresh(part(ok)),
                    err: wrapped,
                }
            }
            _ => unreachable!("`?`'s candidates are the language's two"),
        };
        let value = self.written(making, term, recorder, &mut recorded);
        let relation = if self.is_unknown(exit) {
            Relation::LowerBound {
                lower: value,
                upper: exit,
            }
        } else {
            Relation::CoercionConstraint {
                from: value,
                to: exit,
            }
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }
}
