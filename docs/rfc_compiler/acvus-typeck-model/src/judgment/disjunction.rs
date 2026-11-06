//! Disjunction: the alternative a disjunctive constraint takes from what the
//! classes it reads read.
//!
//! A disjunctive constraint stands where the AST gives a hint whose meaning
//! depends on the constructor of a class it reads: `&r` lends `r`'s target
//! where `r` is a reference, since `&&T` does not exist (D1), and `r` itself
//! otherwise; a projection `x.f` reads as `(*x).f` where `x` is a reference
//! (auto-deref); a pattern binding reached through a reference lends its
//! position, as a lend does; and a call of a function value calls the
//! lambdas the value names. Each reads that constructor, and waits where it
//! is not known: a class whose constructor is not known is never read as one
//! that is no reference.
//!
//! The free functions (`lend`, `project`, `passes_through`, `passed`,
//! `bind`, `head_value`) are the layer: they take values and return values, and import no `State`
//! and no other layer. The `impl State` functions below them are the
//! orchestrator's reading of a disjunctive constraint's positions in the
//! record (`reads`, `records_into`, `reading`, `read`), which hand those
//! values to the layer.

use std::collections::BTreeSet;

use crate::class::{ClassId, Side};
use crate::disjunction::{Decision, Disjunction, LendTarget};
use crate::sig::DeclaredTypes;
#[cfg(test)]
use crate::domain::Domain;
use crate::program::Projection;
use crate::read::ReadValue;
use crate::relation::{Relation, Role};
use crate::solve::State;
use crate::ty::{Constructor, HasTerm, Mutability, Name, Term, lesser};

use super::Head;
use super::requirements::domain_of;
use super::view::left_out_by;

/// What a disjunctive constraint reads and becomes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Reading {
    /// The alternative the record gives it; `None` where it cannot yet decide.
    pub(crate) decision: Option<Decision>,
}

/// The alternative a lend `&ₘ p` takes (D1, the AST's hint: `&&T` does not
/// exist, so `&r` with `r: &T` is `&T`), from the reading of its place,
/// `place`, and its own mutability, `mutability`: `Lend { Through, n' }`
/// where the place reads as `Ref(n, _)`, `n'` the lesser of `mutability` and
/// `n` (`&mut` with `&` is `&`); `Lend { Place, mutability }` where it reads
/// as any other constructor; and `None` (it waits) where the reading gives no
/// constructor, being undecided or a domain alone. It never reads a place
/// whose constructor is not known as one that is no reference, and reads no
/// class itself.
pub fn lend(place: &ReadValue, mutability: Mutability) -> Option<Decision> {
    match place.decided()? {
        Term::Ref(of, _) => Some(Decision::Lend {
            target: LendTarget::Through,
            mutability: lesser(Some(mutability), Some(*of))?,
        }),
        _ => Some(Decision::Lend {
            target: LendTarget::Place,
            mutability,
        }),
    }
}

/// The alternative a projection other than a Deref takes (auto-deref, the
/// AST's hint), from the reading of its operand, `operand`:
/// `AppliedToTarget` where the operand reads as a reference, `Applied` where
/// it reads as any other constructor, and `None` (it waits) where the
/// reading gives no constructor, being undecided or a domain alone. It never
/// reads an operand whose constructor is not known as one that is no
/// reference.
pub fn project(operand: &ReadValue) -> Option<Decision> {
    match operand.decided()? {
        Term::Ref(..) => Some(Decision::AppliedToTarget),
        _ => Some(Decision::Applied),
    }
}

/// What a step of a binding's path that applies to its operand's target
/// passes the binding through, from the reading of that operand, `operand`:
/// `Some(Some(m))` where it reads as `Ref(m, _)`, a reference of `m`;
/// `Some(None)` where it reads as any other constructor, no reference; and
/// `None` (the binding waits) where the reading gives no constructor.
pub fn passes_through(operand: &ReadValue) -> Option<Option<Mutability>> {
    match operand.decided()? {
        Term::Ref(mutability, _) => Some(Some(*mutability)),
        _ => Some(None),
    }
}

/// What a binding's path passes it through, from what each step of the path
/// after its last Deref passes, `passes`, as `passes_through` gives it:
/// `Some(None)` where no step passes a reference, `Some(Some(m))` where one
/// does, `m` the lesser of the mutabilities passed (`&mut` with `&` is `&`),
/// and `None` (the binding waits) where a step waits. It reads no class.
pub fn passed<I>(passes: I) -> Option<Option<Mutability>>
where
    I: IntoIterator<Item = Option<Option<Mutability>>>,
{
    passes
        .into_iter()
        .try_fold(None, |passed: Option<Mutability>, pass| {
            Some(match pass? {
                Some(mutability) => lesser(passed.or(Some(mutability)), Some(mutability)),
                None => passed,
            })
        })
}

/// The alternative a pattern binding takes from what its path passes it
/// through, `passed`, as `passed` gives it: `Read` where the path passes no
/// reference, and otherwise the lend of its position at the mutability
/// passed (D1), as `lend` gives it from the reading `position` returns,
/// `BindLend` of that lend's target and mutability; `None` (it waits) where
/// that reading gives no constructor. `position` is called only where the
/// path passes a reference; the function reads no class itself.
pub fn bind<F>(passed: Option<Mutability>, position: F) -> Option<Decision>
where
    F: FnOnce() -> ReadValue,
{
    let Some(mutability) = passed else {
        return Some(Decision::Read);
    };
    match lend(&position(), mutability)? {
        Decision::Lend { target, mutability } => Some(Decision::BindLend { target, mutability }),
        other => unreachable!("a lend decides a lend, not {other:?}"),
    }
}

/// Converts a `Head` into a shallow `ReadValue`: its top-level constructor
/// and members, with every part `ReadValue::Undecided`, a struct's one part
/// per field and an extern-defined type's one per argument and its identity,
/// as `types` declares them, and a reference's target among those parts. A
/// constructor the head gives is read as that constructor, never as a
/// domain, so that a reading of it decides what it decides. It reads no
/// class.
pub fn head_value(head: Head, types: &DeclaredTypes) -> ReadValue {
    let named = |members: &BTreeSet<crate::ty::Member>| -> Vec<Name> {
        members
            .iter()
            .map(|member| match member {
                crate::ty::Member::Field(name) | crate::ty::Member::Variant(name) => name.clone(),
            })
            .collect()
    };
    let part = || ReadValue::Undecided;
    let term: Term<ReadValue> = match &head.constructor {
        Constructor::Ref(mutability) => Term::Ref(*mutability, part()),
        Constructor::Tuple(n) => Term::Tuple((0..*n).map(|_| part()).collect()),
        Constructor::Array(n) => Term::Array(*n, part()),
        Constructor::Slice => Term::Slice(part()),
        Constructor::Option => Term::Option(part()),
        Constructor::Result => Term::Result {
            ok: part(),
            err: part(),
        },
        Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
        Constructor::Object => Term::Object(
            named(&head.members)
                .into_iter()
                .map(|field| (field, part()))
                .collect(),
        ),
        Constructor::Enum(name) => Term::Enum(
            name.clone(),
            named(&head.members)
                .into_iter()
                .map(|variant| (variant, part()))
                .collect(),
        ),
        Constructor::ErrorOf => Term::ErrorOf(part()),
        Constructor::Struct(module) => {
            let fields = types
                .struct_decl(module)
                .expect("a head's struct is declared")
                .fields
                .len();
            Term::Struct(module.clone(), (0..fields).map(|_| part()).collect())
        }
        Constructor::Extern(name) => {
            let decl = types
                .extern_decl(name)
                .expect("a head's extern-defined type is declared");
            Term::Extern {
                name: name.clone(),
                args: (0..decl.arity).map(|_| part()).collect(),
                identity: decl.identity.then(part),
            }
        }
        Constructor::Scalar(scalar) => Term::Scalar(*scalar),
        Constructor::String => Term::String,
        Constructor::Str => Term::Str,
        Constructor::Unit => Term::Unit,
        Constructor::Never => Term::Never,
        Constructor::Identity(name) => Term::Identity(name.clone()),
    };
    ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
}

impl State {
    /// What the disjunctive constraint `name` decides reading the record
    /// without the relations the disjunctive constraints `left_out` recorded;
    /// `None` where that decides nothing.
    pub(crate) fn decision(&self, name: &Name, left_out: &BTreeSet<Name>) -> Option<Decision> {
        self.reading(name, left_out).decision
    }

    /// The roots of the classes the disjunctive constraint `name` reads: a
    /// lend's place; a projection's operand, none for a Deref, which reads
    /// nothing; a binding's position and the operands of the projections of
    /// its path after its last Deref, whose constructors tell whether the
    /// path passes a reference; and a call's callee and the class whose
    /// function value it calls.
    pub(crate) fn reads(&self, name: &Name) -> BTreeSet<ClassId> {
        let (kind, _) = self.disjunction(name);
        let classes: Vec<ClassId> = match kind {
            Disjunction::Lend { place, .. } => vec![*place],
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => Vec::new(),
            Disjunction::Project { operand, .. } => vec![*operand],
            Disjunction::Bind { path, position, .. } => std::iter::once(*position)
                .chain(self.passed_steps(path).iter().filter_map(|step| {
                    match self.disjunction(step).0 {
                        Disjunction::Project { operand, .. } => Some(*operand),
                        _ => None,
                    }
                }))
                .collect(),
            Disjunction::CallValue { callee, .. } => vec![*callee, self.called(*callee)],
        };
        classes.into_iter().map(|class| self.root(class)).collect()
    }

    /// Whether a class the disjunctive constraint `name` reads (`reads`) has
    /// no value once solving ends (D5): the constraint then has nothing to
    /// read, and is refused with that class, whose refusal names it among
    /// its items. Read once solving ends, by the outcome and by I\*; during
    /// solving, a class with no value yet may still take one. It records
    /// nothing.
    pub(crate) fn refused_with(&self, name: &Name) -> bool {
        self.reads(name)
            .into_iter()
            .any(|class| self.value(class).is_none())
    }

    /// The roots of the classes a decision still open to the undecided
    /// disjunctive constraint `name` records into: its upper sides and parts,
    /// a binding's position, and a lend's place or a projection's operand,
    /// on which a decision may record a requirement; a call records nothing
    /// into its callee. It lists what a decision records, for I\*'s
    /// dependencies at the end, and is no waiting fact: the waiting facts
    /// are `State::held_at`'s. It records nothing.
    pub(crate) fn records_into(&self, name: &Name) -> BTreeSet<ClassId> {
        let (kind, _) = self.disjunction(name);
        self.classes
            .roles_of(self.disjunction_at(name))
            .into_iter()
            .filter(|(_, role)| {
                matches!(role.side(), Some(Side::Upper | Side::Own | Side::Part))
                    || matches!(role, Role::Bound)
            })
            .filter(|(class, _)| {
                !matches!(kind, Disjunction::CallValue { callee, .. }
                    if self.root(*callee) == self.root(*class))
            })
            .map(|(class, _)| self.root(class))
            .collect()
    }

    /// The class whose function value a call of `callee` calls: the target of
    /// the reference `callee` holds, or `callee` itself otherwise.
    pub(crate) fn called(&self, callee: ClassId) -> ClassId {
        let root = self.root(callee);
        match &self.classes.held(root).structure {
            Some((Term::Ref(_, target), _)) => self.root(*target),
            _ => root,
        }
    }

    /// Evaluates what the disjunctive constraint `name` reads and becomes,
    /// leaving out relations recorded by `left_out` and by `name` itself.
    ///
    /// - A lend takes what `lend` gives the reading of its place
    ///   (`read_without_projections`) (D1).
    /// - A Deref has one alternative, `Applied`.
    /// - Another projection takes what `project` gives the reading of its
    ///   operand (`read_without_projections`).
    /// - A binding takes what `bind` gives what `passed` gives the steps of
    ///   its path after its last Deref, and the reading of its position: a
    ///   step not yet decided is read as it would decide, and a step
    ///   applying to its operand's target passes what `passes_through` gives
    ///   that operand's reading.
    /// - A call of a function value calls the lambdas its callee's value
    ///   names, once it names one.
    ///
    /// `None` where it waits. It records nothing.
    pub(crate) fn reading(&self, name: &Name, left_out: &BTreeSet<Name>) -> Reading {
        let (kind, _) = self.disjunction(name);
        let mut left_out = left_out.clone();
        left_out.insert(name.clone());
        let decision = match kind {
            Disjunction::Lend {
                mutability, place, ..
            } => lend(
                &self.read_without_projections(*place, &left_out),
                *mutability,
            ),
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => Some(Decision::Applied),
            Disjunction::Project { operand, .. } => {
                project(&self.read_without_projections(*operand, &left_out))
            }
            Disjunction::Bind { path, position, .. } => {
                passed(self.passed_steps(path).iter().map(|step| {
                    let (kind, decided) = self.disjunction(step);
                    let decided = decided
                        .cloned()
                        .or_else(|| self.reading(step, &left_out).decision)?;
                    let Decision::AppliedToTarget = decided else {
                        return Some(None);
                    };
                    let Disjunction::Project { operand, .. } = kind else {
                        unreachable!("a binding's path is of projections");
                    };
                    passes_through(&self.read_without_projections(*operand, &left_out))
                }))
                .and_then(|passed| {
                    bind(passed, || {
                        self.read_without_projections(*position, &left_out)
                    })
                })
            }
            Disjunction::CallValue { callee, .. } => {
                let lambdas = self.values_of(self.called(*callee));
                (!lambdas.is_empty()).then_some(Decision::Calls(lambdas))
            }
        };
        Reading { decision }
    }

    /// The steps of a binding's path after its last Deref: those whose
    /// operand may pass the binding through a reference. A step before an
    /// explicit Deref passes none, the Deref having taken the reference.
    fn passed_steps<'a>(&self, path: &'a [Name]) -> &'a [Name] {
        let after_deref = path
            .iter()
            .rposition(|step| {
                matches!(
                    self.disjunction(step).0,
                    Disjunction::Project {
                        projection: Projection::Deref,
                        ..
                    }
                )
            })
            .map_or(0, |deref| deref + 1);
        &path[after_deref..]
    }

    /// The disjunctive constraints `left_out` and every projection other
    /// than a Deref whose operand is `class`: what a reading of the
    /// constructor of `class` leaves out. Such a projection records only
    /// structure constraints on its operand, and no requirement gives a class
    /// a head.
    fn projections_left_out(&self, class: ClassId, left_out: &BTreeSet<Name>) -> BTreeSet<Name> {
        let root = self.root(class);
        let projections = self
            .classes
            .disjunctions()
            .filter_map(|(_, name, kind, _)| match kind {
                Disjunction::Project {
                    projection,
                    operand,
                    ..
                } if !matches!(projection, Projection::Deref) && self.root(*operand) == root => {
                    Some(name.clone())
                }
                _ => None,
            });
        left_out.iter().cloned().chain(projections).collect()
    }

    /// The reading of a class whose constructor a disjunctive constraint
    /// reads (a lend's place, a projection's operand, a binding's position or
    /// the operand of a step of its path), `class`, without what
    /// `projections_left_out` leaves out. It records nothing.
    fn read_without_projections(&self, class: ClassId, left_out: &BTreeSet<Name>) -> ReadValue {
        self.read(class, &self.projections_left_out(class, left_out))
    }

    /// Reads the `ReadValue` of class `id` without relations recorded by `left_out`.
    pub(crate) fn read(&self, id: ClassId, left_out: &BTreeSet<Name>) -> ReadValue {
        self.read_in(id, left_out, &mut BTreeSet::new())
    }

    fn read_in(
        &self,
        id: ClassId,
        left_out: &BTreeSet<Name>,
        visiting: &mut BTreeSet<ClassId>,
    ) -> ReadValue {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return ReadValue::Undecided;
        }
        let kept = |at: usize| {
            let recorded = &self.classes.items()[at];
            !recorded.dropped && !left_out_by(&recorded.recorder, left_out)
        };
        let occurrences = self.classes.occurrences(root);
        let structure = self
            .classes
            .held(root)
            .structure
            .as_ref()
            .filter(|_| {
                occurrences.iter().any(|occurrence| {
                    occurrence.side() == Some(Side::Own)
                        && kept(occurrence.at)
                        && matches!(
                            self.classes.items()[occurrence.at].relation(),
                            Some(Relation::Structure(..))
                        )
                })
            })
            .map(|(structure, _)| structure.clone());
        let value = match structure {
            Some(structure) => {
                let term = structure.map(|_, part| self.read_in(part, left_out, visiting));
                ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
            }
            None => {
                let bounds = self.bounds(root, left_out);
                match self.value_head(&bounds) {
                    Some(head) => head_value(head, self.decls.types()),
                    None => domain_of(bounds.domains)
                        .map_or(ReadValue::Undecided, ReadValue::Domain),
                }
            }
        };
        visiting.remove(&root);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(mutability: Mutability) -> ReadValue {
        ReadValue::from_term(Term::Ref(mutability, ReadValue::Undecided)).expect("valid term")
    }

    #[test]
    fn a_projection_waits_for_its_operands_constructor() {
        let unit = ReadValue::from_term(Term::Unit).expect("valid term");
        assert_eq!(project(&unit), Some(Decision::Applied));
        assert_eq!(
            project(&reference(Mutability::Shared)),
            Some(Decision::AppliedToTarget)
        );
        assert_eq!(project(&ReadValue::Undecided), None);
        assert_eq!(
            project(&ReadValue::Domain(Domain::Except(BTreeSet::new()))),
            None
        );
    }

    #[test]
    fn a_lend_reborrows_through_a_reference_and_waits_for_the_places_constructor() {
        let unit = ReadValue::from_term(Term::Unit).expect("valid term");
        assert_eq!(
            lend(&unit, Mutability::Exclusive),
            Some(Decision::Lend {
                target: LendTarget::Place,
                mutability: Mutability::Exclusive,
            })
        );
        assert_eq!(
            lend(&reference(Mutability::Shared), Mutability::Exclusive),
            Some(Decision::Lend {
                target: LendTarget::Through,
                mutability: Mutability::Shared,
            })
        );
        assert_eq!(lend(&ReadValue::Undecided, Mutability::Shared), None);
        assert_eq!(
            lend(&ReadValue::Domain(Domain::references()), Mutability::Shared),
            None
        );
    }

    #[test]
    fn a_binding_lends_at_the_lesser_mutability_its_path_passes() {
        let unit = || ReadValue::from_term(Term::Unit).expect("valid term");
        assert_eq!(
            passed([Some(None), Some(Some(Mutability::Exclusive))]),
            Some(Some(Mutability::Exclusive))
        );
        assert_eq!(
            passed([
                Some(Some(Mutability::Exclusive)),
                Some(Some(Mutability::Shared))
            ]),
            Some(Some(Mutability::Shared))
        );
        assert_eq!(passed([Some(None)]), Some(None));
        assert_eq!(passed([Some(Some(Mutability::Shared)), None]), None);
        assert_eq!(bind(None, || ReadValue::Undecided), Some(Decision::Read));
        assert_eq!(
            bind(Some(Mutability::Exclusive), unit),
            Some(Decision::BindLend {
                target: LendTarget::Place,
                mutability: Mutability::Exclusive
            })
        );
        assert_eq!(
            bind(Some(Mutability::Exclusive), || reference(
                Mutability::Shared
            )),
            Some(Decision::BindLend {
                target: LendTarget::Through,
                mutability: Mutability::Shared
            })
        );
        assert_eq!(
            bind(Some(Mutability::Shared), || ReadValue::Undecided),
            None
        );
        assert_eq!(
            passes_through(&reference(Mutability::Exclusive)),
            Some(Some(Mutability::Exclusive))
        );
        assert_eq!(passes_through(&ReadValue::Undecided), None);
    }
}
