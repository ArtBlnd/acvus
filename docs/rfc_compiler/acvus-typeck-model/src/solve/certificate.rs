//! The certificate an accepted order of solving leaves, and its checker.
//!
//! The checker reads the program and the certificate alone, on closed types:
//! every relation the text and the decisions give is regenerated from the
//! program's items, never read from a state's record.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::choice::Chosen;
use crate::class::{ClassId, ClassName};
use crate::coerce::{Coercion, Structural, along_le, coerce};
use crate::cut::Cut;
use crate::decl::{Decls, Node};
use crate::disjunction::generalization::Generalization;
use crate::disjunction::{Decision, DisjunctionAt, LendTarget};
use crate::relation::Role;
use crate::explore::Final;
use crate::instance::Made;
use crate::program::{ENTRY, Function, Item, Origin, Part, Place, Position, Program, Projection};
use crate::relation::{Recorder, copy_name};
use crate::sig::{SigArgs, SigTy, SigVar, Signature};
use crate::instance::LambdaCopy;
use crate::solve::State;
use crate::ty::{Constructor, HasTerm, Mutability, Name, PartKey, Term, Ty, TyError, check_term};

/// A closed value as the certificate gives it: a type each of whose function
/// values carries the lambdas its Values names, as a function value carries
/// its facts on its type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Value {
    term: Rc<Term<Value>>,
    /// At a function value, the lambdas its Values names; none elsewhere. No
    /// declaration, text or context writes a function value, so every
    /// function value in a value is a class's, and names its Values.
    lambdas: Option<BTreeSet<Name>>,
}

impl HasTerm for Value {
    fn decided(&self) -> Option<&Term<Value>> {
        Some(&self.term)
    }

    fn from_term(term: Term<Value>) -> Result<Value, TyError> {
        check_term(&term)?;
        Ok(Value {
            term: Rc::new(term),
            lambdas: None,
        })
    }
}

impl Value {
    fn term(&self) -> &Term<Value> {
        &self.term
    }

    /// The type, its function values' Values left out.
    fn ty(&self) -> Ty {
        let term = (*self.term).clone().map(|_, part| part.ty());
        Ty::new(term).expect("a certified value is a type")
    }

    fn part(&self, key: &crate::ty::PartKey) -> Option<&Value> {
        self.term.part(key)
    }

    /// A value a test plants, as a case writes it: a function value naming
    /// the lambdas of its Values, `lambdas`, which no other term names.
    #[cfg(test)]
    pub(crate) fn planted(
        term: Term<Value>,
        lambdas: Option<BTreeSet<Name>>,
    ) -> Result<Value, String> {
        assert_eq!(
            matches!(term, Term::FunctionValue { .. }),
            lambdas.is_some(),
            "a function value, and no other term, names its Values"
        );
        check_term(&term).map_err(|error| format!("{error:?}"))?;
        Ok(Value {
            term: Rc::new(term),
            lambdas,
        })
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.term, f)?;
        if let Some(lambdas) = &self.lambdas {
            let names: Vec<&str> = lambdas.iter().map(Name::as_str).collect();
            write!(f, "{{{}}}", names.join(", "))?;
        }
        Ok(())
    }
}

/// The value the state gives the class at `root`, each function value in it
/// carrying the lambdas its Values names; none where the class
/// has no closed value.
fn certified(state: &State, root: ClassId) -> Option<Value> {
    let ty = state.value(root)?;
    let (structure, _) = state.classes.held(root).structure.clone()?;
    let term = structure
        .try_map(|_, part| certified(state, state.root(part)).ok_or(()))
        .ok()?;
    let lambdas = matches!(term, Term::FunctionValue { .. }).then(|| state.values_of(root));
    let value = Value {
        term: Rc::new(term),
        lambdas,
    };
    assert_eq!(value.ty(), ty, "a certified value is the class's value");
    Some(value)
}

/// What one state of a final state decided, as the checker reads it.
pub struct Sheet {
    group: Name,
    /// Each class by its label, as the outcome and the case language name
    /// it: what a case's change names a class by, and what a violation
    /// names it by. A lookup by the record's own structure is by a text
    /// class's name alone, the program's interface, or by an item and a
    /// role.
    labels: BTreeMap<ClassName, ClassId>,
    /// The class each disjunctive constraint and each use made anew in each
    /// role, read from the record's items.
    made: BTreeMap<(Name, DisjunctionAt), ClassId>,
    roots: BTreeMap<ClassId, ClassId>,
    /// Each root's value.
    values: BTreeMap<ClassId, Option<Value>>,
    /// Each root's parts, by their roots, where it holds a structure: where the
    /// decomposition linked a part's two sides.
    parts: BTreeMap<ClassId, Vec<(PartKey, ClassId)>>,
    /// The roots that are unknowns of the state's group, which a value
    /// passing into one enters.
    unknowns: BTreeSet<ClassId>,
    decisions: BTreeMap<Name, Option<Decision>>,
    /// What each use became, read off its choice.
    choices: BTreeMap<Name, Option<Chosen>>,
    instances: Vec<Made>,
    /// The lambdas an instance made anew here, with their environments.
    lambdas: BTreeMap<Name, LambdaCopy>,
}

impl Sheet {
    fn of(state: &State) -> Sheet {
        let labels: BTreeMap<ClassName, ClassId> = state
            .arena
            .classes()
            .map(|(id, name)| (name.clone(), id))
            .collect();
        let made: BTreeMap<(Name, DisjunctionAt), ClassId> = state
            .classes
            .disjunctions()
            .map(|(at, name, _, _)| (at, name))
            .chain(state.classes.choices().map(|(at, name, _, _)| (at, name)))
            .flat_map(|(at, name)| {
                state
                    .classes
                    .roles_of(at)
                    .into_iter()
                    .filter_map(move |(class, role)| match role {
                        Role::Made(made) => Some(((name.clone(), made), class)),
                        _ => None,
                    })
            })
            .collect();
        let roots: BTreeMap<ClassId, ClassId> =
            labels.values().map(|&id| (id, state.root(id))).collect();
        let distinct: BTreeSet<ClassId> = roots.values().copied().collect();
        let parts = distinct
            .iter()
            .map(|&root| {
                let parts = match &state.classes.held(root).structure {
                    Some((structure, _)) => structure
                        .parts()
                        .into_iter()
                        .map(|(key, &part)| (key, state.root(part)))
                        .collect(),
                    None => Vec::new(),
                };
                (root, parts)
            })
            .collect();
        let unknowns = distinct
            .iter()
            .copied()
            .filter(|&root| state.is_unknown(root))
            .collect();
        let values = distinct
            .into_iter()
            .map(|root| (root, certified(state, root)))
            .collect();
        let decisions = state.decisions();
        Sheet {
            group: state.group.clone(),
            labels,
            made,
            roots,
            values,
            parts,
            unknowns,
            decisions,
            choices: state.choices(),
            instances: state.instances(),
            lambdas: state.lambda_copies(),
        }
    }

    fn is_entry(&self) -> bool {
        self.group.as_str() == ENTRY
    }

    /// The class of the text class of this name.
    fn id(&self, name: &ClassName) -> Option<ClassId> {
        assert!(
            matches!(name, ClassName::Text(_)),
            "a lookup by name is of a text class, not {name}"
        );
        self.labels.get(name).copied()
    }

    /// The class the disjunctive constraint `owner` made anew in the role
    /// `made`, where it made one.
    fn made(&self, owner: &Name, made: &DisjunctionAt) -> Option<ClassId> {
        self.made.get(&(owner.clone(), made.clone())).copied()
    }

    fn root(&self, id: ClassId) -> ClassId {
        self.roots[&id]
    }

    /// A name of a member of the root.
    fn name_of(&self, root: ClassId) -> &ClassName {
        self.labels
            .iter()
            .find(|(_, id)| self.root(**id) == root)
            .map(|(name, _)| name)
            .expect("a root has a member")
    }

    fn instance(&self, at: &Name) -> Option<&Made> {
        self.instances.iter().find(|made| made.at == *at)
    }
}

/// The certificate of a final state: a sheet for the entry and for each
/// group solved before it.
pub struct Certificate {
    sheets: BTreeMap<Name, Sheet>,
    /// Which group solves each class and constraint, as the groups the order
    /// solved cut the lambdas' environments.
    cut: Cut,
    accepted: bool,
}

impl Certificate {
    pub fn of(program: &Program, end: &Final<'_>) -> Certificate {
        let sheets = end
            .groups
            .iter()
            .map(|(name, state)| (name.clone(), Sheet::of(state)))
            .chain([(Name::new(ENTRY), Sheet::of(end.entry))])
            .collect();
        Certificate {
            sheets,
            cut: Cut::of(program, end.groups, &Name::new(ENTRY)),
            accepted: end.entry.outcome_with(end.groups).accepted,
        }
    }
}

/// A certificate as a solver that erred would leave it, so that a test sees
/// the checker find the error.
#[cfg(test)]
impl Certificate {
    /// Gives the class `name` of `group`'s state, with every class equated to
    /// it, the value `value`.
    /// The class is named as the outcome writes it, so that a class an
    /// instance or a disjunctive constraint made is named as well as the text's.
    pub(crate) fn set_value(&mut self, group: &str, name: &str, value: Value) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let id = sheet
            .labels
            .iter()
            .find(|(class, _)| class.to_string() == name)
            .map(|(_, &id)| id)
            .unwrap_or_else(|| panic!("{group} has the class {name}"));
        let root = sheet.root(id);
        sheet.values.insert(root, Some(value));
    }

    /// Gives the use `used` of `group`'s state what it became, `chosen`.
    pub(crate) fn set_choice(&mut self, group: &str, used: &str, chosen: Chosen) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let became = sheet
            .choices
            .get_mut(&Name::new(used))
            .unwrap_or_else(|| panic!("{group} has the use {used}"));
        *became = Some(chosen);
    }

    /// Gives the disjunctive constraint `disjunction` of `group`'s state the decision `decision`.
    pub(crate) fn set_decision(&mut self, group: &str, disjunction: &str, decision: Decision) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let decided = sheet
            .decisions
            .get_mut(&Name::new(disjunction))
            .unwrap_or_else(|| panic!("{group} has the disjunctive constraint {disjunction}"));
        *decided = Some(decision);
    }
}

/// A relation the text or a decision gives that the certificate's values do
/// not satisfy.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Violation {
    /// The binding group whose text gives it, and the instances through which
    /// the context reaches the state its values are read in.
    pub context: String,
    /// The item, disjunctive constraint or class that gives it.
    pub at: String,
    pub what: String,
}

/// Every violation of an accepted final state; none for a refused one.
pub fn check(program: &Program, certificate: &Certificate) -> Vec<Violation> {
    if !certificate.accepted {
        return Vec::new();
    }
    let text = Text::of(program, &certificate.cut);
    let mut checker = Checker {
        program,
        cut: &certificate.cut,
        decls: &program.decls,
        text: &text,
        certificate,
        violations: Vec::new(),
    };
    checker.acceptance();
    for context in contexts(certificate) {
        checker.context(&context);
    }
    let mut violations = checker.violations;
    violations.sort();
    violations.dedup();
    violations
}

/// One instance on a context's way: the instance, and the sheet of the state
/// that made it.
struct Step<'a> {
    made: &'a Made,
    sheet: &'a Sheet,
}

/// Where a binding group's text is checked: its own state, or an instance of
/// its C, through the instances that copied it on, to the state its values
/// are read in. Level 0 is the group's
/// own sheet, and level `k` the sheet of `steps[k - 1]`.
struct Context<'a> {
    group: Name,
    own: &'a Sheet,
    steps: Vec<Step<'a>>,
}

/// What a context gives a class.
enum Lookup {
    Value(Value),
    /// An original left in C, or a class of a state other than the entry's
    /// with no value there: nothing is checked of it here.
    Absent,
    /// A class of the entry's state with no value.
    Missing,
}

/// A disjunctive constraint's decision, the level it is made at, and the disjunctive constraint's name there.
struct DecidedAt {
    decision: Decision,
    level: usize,
    owner: Name,
}

/// What a context gives a disjunctive constraint of the group.
enum Decided {
    At(DecidedAt),
    /// Left undecided in C at the last level, which is not the entry's.
    InC,
}

/// What a use became, the level it became it at, and the use's name there.
struct ChosenAt {
    chosen: Chosen,
    level: usize,
    owner: Name,
}

/// Where a function value passes into a function type: the level of
/// the sheet that decomposed it, the roots there of the two sides, and whether the
/// type's result is an unknown the lambda's result enters.
struct IntoFunctionType {
    level: usize,
    from: ClassId,
    to: ClassId,
    ret_enters: bool,
}

/// Where a value is read: a class at a level of a context (`Context::locate`).
#[derive(Clone, Copy, Debug)]
struct At {
    level: usize,
    class: ClassId,
}

/// A value and the class it is read at, none where the checker reads it at no
/// class, as a type a signature writes with its variables bound.
#[derive(Clone)]
struct Read {
    value: Value,
    at: Option<At>,
}

/// A parameter's type, and whether an argument enters it or passes into it by a
/// coercion constraint.
struct Param {
    ty: Read,
    enters: bool,
}

/// A path of parts from the top of a value.
type Path = Vec<PartKey>;

impl<'a> Context<'a> {
    /// Where the part at `path` of the class read at `at` is read, none where a
    /// class on the way holds no such part.
    fn part_at(&self, at: At, path: &[PartKey]) -> Option<At> {
        let (mut level, mut root) = self.locate(at.level, at.class);
        for key in path {
            let parts = self.sheet(level).parts.get(&root)?;
            let &(_, part) = parts.iter().find(|(held, _)| held == key)?;
            (level, root) = self.locate(level, part);
        }
        Some(At { level, class: root })
    }

    fn sheet(&self, level: usize) -> &'a Sheet {
        match level {
            0 => self.own,
            _ => self.steps[level - 1].sheet,
        }
    }

    fn last(&self) -> usize {
        self.steps.len()
    }

    fn describe(&self) -> String {
        let through: Vec<String> = self
            .steps
            .iter()
            .map(|step| format!("{} in {}", step.made.at, step.sheet.group))
            .collect();
        if through.is_empty() {
            format!("{}", self.group)
        } else {
            format!("{} through {}", self.group, through.join(", "))
        }
    }

    /// The level a class of level `level` is read at, and its root there.
    fn locate(&self, level: usize, id: ClassId) -> (usize, ClassId) {
        let (mut level, mut id) = (level, id);
        loop {
            let root = self.sheet(level).root(id);
            let Some(step) = self.steps.get(level) else {
                return (level, root);
            };
            match step.made.made.get(&root) {
                Some(&next) => {
                    level += 1;
                    id = next;
                }
                None => return (level, root),
            }
        }
    }

    fn lookup(&self, level: usize, id: ClassId) -> Lookup {
        let (at, root) = self.locate(level, id);
        let sheet = self.sheet(at);
        match &sheet.values[&root] {
            Some(ty) => Lookup::Value(ty.clone()),
            None if sheet.is_entry() => Lookup::Missing,
            None => Lookup::Absent,
        }
    }

    /// The decision of the group's disjunctive constraint `name`: its own, or its copy's in an
    /// instance on the way.
    fn decision(&self, name: &Name) -> Result<Decided, String> {
        let mut name = name.clone();
        for level in 0..=self.last() {
            let sheet = self.sheet(level);
            let Some(decision) = sheet.decisions.get(&name) else {
                return Err(format!(
                    "{} has no disjunctive constraint {name}",
                    sheet.group
                ));
            };
            if let Some(decision) = decision {
                return Ok(Decided::At(DecidedAt {
                    decision: decision.clone(),
                    level,
                    owner: name,
                }));
            }
            if level < self.last() {
                name = copy_name(&self.steps[level].made.at, &name);
            }
        }
        if self.sheet(self.last()).is_entry() {
            Err(format!("{name} is not decided"))
        } else {
            Ok(Decided::InC)
        }
    }
}

impl Context<'_> {
    /// What the group's use `name` became: at its own level, or its copy's in
    /// an instance on the way. None where it became nothing up to a last
    /// level that is not the entry's, left in C.
    fn chosen(&self, name: &Name) -> Result<Option<ChosenAt>, String> {
        let mut name = name.clone();
        for level in 0..=self.last() {
            let sheet = self.sheet(level);
            let Some(became) = sheet.choices.get(&name) else {
                return Err(format!("{} has no use {name}", sheet.group));
            };
            if let Some(chosen) = became {
                return Ok(Some(ChosenAt {
                    chosen: chosen.clone(),
                    level,
                    owner: name,
                }));
            }
            if level < self.last() {
                name = copy_name(&self.steps[level].made.at, &name);
            }
        }
        if self.sheet(self.last()).is_entry() {
            Err(format!("{name} became nothing"))
        } else {
            Ok(None)
        }
    }
}

/// Every context of every group: its own sheet, and each way of instances
/// from it to a sheet that made one.
fn contexts(certificate: &Certificate) -> Vec<Context<'_>> {
    let mut contexts: Vec<Context<'_>> = certificate
        .sheets
        .values()
        .map(|sheet| Context {
            group: sheet.group.clone(),
            own: sheet,
            steps: Vec::new(),
        })
        .collect();
    let mut i = 0;
    while i < contexts.len() {
        let last = contexts[i].sheet(contexts[i].last()).group.clone();
        let mut extended = Vec::new();
        for sheet in certificate.sheets.values() {
            for made in sheet.instances.iter().filter(|made| made.group == last) {
                let mut steps: Vec<Step<'_>> = contexts[i]
                    .steps
                    .iter()
                    .map(|step| Step {
                        made: step.made,
                        sheet: step.sheet,
                    })
                    .collect();
                steps.push(Step { made, sheet });
                extended.push(Context {
                    group: contexts[i].group.clone(),
                    own: contexts[i].own,
                    steps,
                });
            }
        }
        contexts.extend(extended);
        i += 1;
    }
    contexts
}

/// What the program's text says of its classes, without solving: which its
/// equations equate, and which its text decides or makes universal.
struct Text {
    parent: BTreeMap<Name, Name>,
    /// The roots of the classes that hold a type the text writes, gives from
    /// the host, or makes a lambda's value.
    decided: BTreeSet<Name>,
    /// The roots of the classes that are universal.
    universal: BTreeSet<Name>,
}

impl Text {
    fn of(program: &Program, cut: &Cut) -> Text {
        let mut text = Text {
            parent: BTreeMap::new(),
            decided: BTreeSet::new(),
            universal: BTreeSet::new(),
        };
        for (_, item) in &program.items {
            if let Item::Equation(a, b) = item {
                let (a, b) = (text.root(a), text.root(b));
                if a != b {
                    text.parent.insert(a, b);
                }
            }
        }
        for class in &program.classes {
            if matches!(
                class.origin,
                Origin::Written(_) | Origin::Host(_) | Origin::Outward(_) | Origin::Lambda(_)
            ) {
                let root = text.root(&class.name);
                text.decided.insert(root);
            }
        }
        let universals = program
            .functions
            .iter()
            .flat_map(|function| function.params.iter())
            .chain(&program.universals)
            .filter(|name| cut.is_universal(program, name));
        for name in universals {
            let root = text.root(name);
            text.universal.insert(root);
        }
        text
    }

    fn root(&self, name: &Name) -> Name {
        let mut at = name.clone();
        while let Some(next) = self.parent.get(&at) {
            at = next.clone();
        }
        at
    }

    /// Whether the class is an unknown of its own group by its text: not
    /// decided, and not universal.
    fn unknown(&self, name: &Name) -> bool {
        let root = self.root(name);
        !self.decided.contains(&root) && !self.universal.contains(&root)
    }

    /// Whether a value passing into `into` at `place` enters it.
    fn enters(
        &self,
        program: &Program,
        cut: &Cut,
        group: &Name,
        into: &Name,
        place: Place,
    ) -> bool {
        match place {
            Place::ThroughRef => false,
            Place::ByName => self.unknown(into),
            Place::Slot => cut.owner(program, into) == group && self.unknown(into),
        }
    }

    /// Whether an argument enters the copy of a parameter's type in an
    /// instance: the copy is an unknown of the group the instance is recorded
    /// in unless the text decides the type.
    fn enters_copy(&self, param: &Name) -> bool {
        !self.decided.contains(&self.root(param))
    }
}

struct Checker<'a> {
    program: &'a Program,
    cut: &'a Cut,
    decls: &'a Decls,
    text: &'a Text,
    certificate: &'a Certificate,
    violations: Vec<Violation>,
}

impl Checker<'_> {
    fn violation(&mut self, context: &Context<'_>, at: &str, what: String) {
        self.violations.push(Violation {
            context: context.describe(),
            at: at.to_string(),
            what,
        });
    }

    fn require(&mut self, context: &Context<'_>, at: &str, holds: bool, what: String) {
        if !holds {
            self.violation(context, at, what);
        }
    }

    /// The value of a lookup, recording a violation where it is missing;
    /// none where it is absent or missing.
    fn get(&mut self, context: &Context<'_>, at: &str, of: &str, lookup: Lookup) -> Option<Value> {
        match lookup {
            Lookup::Value(ty) => Some(ty),
            Lookup::Absent => None,
            Lookup::Missing => {
                self.violation(context, at, format!("{of} has no value"));
                None
            }
        }
    }

    /// The value of the group's text class `name` in the context.
    fn text(&mut self, context: &Context<'_>, at: &str, name: &Name) -> Option<Value> {
        let id = context
            .own
            .id(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("{} names the class {name}", context.group));
        let lookup = context.lookup(0, id);
        self.get(context, at, name.as_str(), lookup)
    }

    /// The decided disjunctive constraint's decision, none where it is left in C, recording a
    /// violation where the certificate does not give one it must.
    fn decided(&mut self, context: &Context<'_>, at: &str, name: &Name) -> Option<DecidedAt> {
        match context.decision(name) {
            Ok(Decided::At(decided)) => Some(decided),
            Ok(Decided::InC) => None,
            Err(what) => {
                self.violation(context, at, what);
                None
            }
        }
    }

    /// What the use became, none where it is left in C, recording a
    /// violation where the certificate does not give it something it must.
    fn chosen(&mut self, context: &Context<'_>, at: &str, name: &Name) -> Option<ChosenAt> {
        match context.chosen(name) {
            Ok(chosen) => chosen,
            Err(what) => {
                self.violation(context, at, what);
                None
            }
        }
    }

    /// The value of the group's text class `name` in the context, with the
    /// class it is read at.
    fn text_read(&mut self, context: &Context<'_>, at: &str, name: &Name) -> Option<Read> {
        let id = context
            .own
            .id(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("{} names the class {name}", context.group));
        let value = self.text(context, at, name)?;
        Some(Read {
            value,
            at: Some(At {
                level: 0,
                class: id,
            }),
        })
    }

    /// Where the class a disjunctive constraint `owner` made at `made` is read at `level`, none
    /// where it made none there.
    fn disjunction_at(
        context: &Context<'_>,
        level: usize,
        owner: &Name,
        made: DisjunctionAt,
    ) -> Option<At> {
        let class = context.sheet(level).made(owner, &made)?;
        Some(At { level, class })
    }

    /// A value passing into a type: `from ≤ to` where it enters
    /// it, `from ⇝ to` by a coercion otherwise, each function value in `from`
    /// carried along the coercion as the decomposition carries it (`reach`).
    /// Where it reaches a function value, its lambdas are among that one's;
    /// where it reaches a function type, an instance of each of
    /// its lambdas was made where the decomposition found the two, at the classes they
    /// are read at, as `function_type` checks. Returns whether the
    /// coercion and the Values hold; what an instance breaks is recorded there.
    fn flows(
        &mut self,
        context: &Context<'_>,
        at: &str,
        from: &Read,
        to: &Read,
        enters: bool,
    ) -> bool {
        let (from_ty, to_ty) = (from.value.ty(), to.value.ty());
        let coercion = if enters {
            along_le(self.decls, &from_ty, &to_ty)
        } else {
            coerce(self.decls, &from_ty, &to_ty)
        };
        let Ok(coercion) = coercion else {
            return false;
        };
        let reached = match reach(self.decls, &from.value, &to.value, &coercion) {
            Ok(reached) => reached,
            Err(what) => {
                self.violation(context, at, what);
                return false;
            }
        };
        let mut holds = true;
        for (from_path, to_path) in reached {
            let value =
                value_at(&from.value, &from_path).expect("a function value lies at its path");
            let Some(target) = value_at(&to.value, &to_path) else {
                holds = false;
                continue;
            };
            match target.term() {
                Term::FunctionValue { .. } => {
                    let (Some(lambdas), Some(named)) = (&value.lambdas, &target.lambdas) else {
                        unreachable!(
                            "{value} and {target} are function values of classes, which name \
                             their Values"
                        );
                    };
                    holds &= lambdas.is_subset(named);
                }
                Term::Function { .. } => {
                    let into = Self::into_function_type(context, from, &from_path, to, &to_path);
                    self.function_type(context, at, &into, value, target);
                }
                _ => holds = false,
            }
        }
        holds
    }

    /// Where the function value at `from_path` of `from` reaches the function
    /// type at `to_path` of `to`, by the classes the two are read at, which the
    /// decomposition related there.
    fn into_function_type(
        context: &Context<'_>,
        from: &Read,
        from_path: &[PartKey],
        to: &Read,
        to_path: &[PartKey],
    ) -> IntoFunctionType {
        let (Some(from_at), Some(to_at)) = (from.at, to.at) else {
            panic!(
                "{} reaches the function type {} where the checker reads no class \
                 ",
                from.value, to.value
            );
        };
        let (Some(value), Some(ty)) = (
            context.part_at(from_at, from_path),
            context.part_at(to_at, to_path),
        ) else {
            panic!(
                "the classes of {} and {} hold no parts at {from_path:?} and {to_path:?}",
                from.value, to.value
            );
        };
        if value.level != ty.level {
            panic!(
                "{} and {} are read at two levels, which no rule links a function value to a \
                 function type at yet",
                from.value, to.value
            );
        }
        let ret_enters = context
            .part_at(ty, &[PartKey::Return])
            .is_some_and(|ret| context.sheet(ret.level).unknowns.contains(&ret.class));
        IntoFunctionType {
            level: value.level,
            from: value.class,
            to: ty.class,
            ret_enters,
        }
    }

    /// Every class of the entry has a value, every disjunctive constraint of the entry
    /// is decided, every use of the entry became a candidate or `never`, and every
    /// decided disjunctive constraint is reached.
    fn acceptance(&mut self) {
        for (group, sheet) in &self.certificate.sheets {
            if sheet.is_entry() {
                for (&root, value) in &sheet.values {
                    if value.is_none() {
                        self.violations.push(Violation {
                            context: group.to_string(),
                            at: sheet.name_of(root).to_string(),
                            what: "no value".to_string(),
                        });
                    }
                }
            }
            for (name, decision) in &sheet.decisions {
                if decision.is_none() && sheet.is_entry() {
                    self.violations.push(Violation {
                        context: group.to_string(),
                        at: name.to_string(),
                        what: "undecided".to_string(),
                    });
                }
            }
            for (name, became) in &sheet.choices {
                if became.is_none() && sheet.is_entry() {
                    self.violations.push(Violation {
                        context: group.to_string(),
                        at: name.to_string(),
                        what: "became nothing".to_string(),
                    });
                }
            }
        }
    }

    fn context(&mut self, context: &Context<'_>) {
        let program = self.program;
        for class in program
            .classes
            .iter()
            .filter(|class| *self.cut.owner(program, &class.name) == context.group)
        {
            self.origin(context, &class.name, &class.origin);
        }
        for (i, (_, item)) in program.items.iter().enumerate() {
            if let Some(part) = self.cut.part(program, i, &context.group) {
                self.item(context, i, item, &part);
            }
        }
    }

    fn origin(&mut self, context: &Context<'_>, name: &Name, origin: &Origin) -> Option<()> {
        let at = format!("the class {name}");
        let value = self.text(context, &at, name)?;
        match origin {
            Origin::Var => {}
            Origin::Written(term) | Origin::Host(term) | Origin::Outward(term) => {
                let mut parts = BTreeMap::new();
                for (_, part) in term.parts() {
                    let ty = self.text(context, &at, part)?;
                    parts.insert(part.clone(), ty);
                }
                let written = term.clone().map(|_, part| parts[&part].clone());
                let holds = Value::from_term(written.clone()).is_ok_and(|written| written == value);
                self.require(
                    context,
                    &at,
                    holds,
                    format!("{value} is not the written {written:?}"),
                );
            }
            Origin::Lit(lit) => {
                let holds = matches!(value.term(), Term::Scalar(scalar) if lit.admits(*scalar));
                self.require(context, &at, holds, format!("{value} does not admit {lit}"));
            }
            Origin::Lambda(lambda) => {
                let arity = self
                    .program
                    .function(lambda)
                    .expect("a lambda is a function")
                    .params
                    .len();
                let holds = *value.term() == Term::FunctionValue { arity };
                self.require(context, &at, holds, format!("{value} is not Fn/{arity}"));
                // Each instance on the way that made the lambda anew names
                // its copy.
                let mut named_as = lambda.clone();
                for step in &context.steps {
                    let copy = copy_name(&step.made.at, &named_as);
                    if step.sheet.lambdas.contains_key(&copy) {
                        named_as = copy;
                    }
                }
                let named = value
                    .lambdas
                    .as_ref()
                    .is_some_and(|lambdas| lambdas.contains(&named_as));
                self.require(
                    context,
                    &at,
                    named,
                    format!("{value}'s Values do not name {named_as}"),
                );
            }
        }
        Some(())
    }

    /// The item's part the context's group records.
    fn item(&mut self, context: &Context<'_>, i: usize, item: &Item, part: &Part) -> Option<()> {
        match item {
            Item::Pass { value, into, place } => {
                let at = format!("the pass of {value} into {into}");
                let from = self.text_read(context, &at, value)?;
                let to = self.text_read(context, &at, into)?;
                let enters = self
                    .text
                    .enters(self.program, self.cut, &context.group, into, *place);
                let holds = self.flows(context, &at, &from, &to, enters);
                let way = if enters { "<=" } else { "~>" };
                let (from, to) = (&from.value, &to.value);
                self.require(context, &at, holds, format!("{from} {way} {to} fails"));
            }
            Item::Equation(a, b) => {
                let at = format!("the equation {a} = {b}");
                let (a, b) = (self.text(context, &at, a)?, self.text(context, &at, b)?);
                self.require(context, &at, a == b, format!("{a} != {b}"));
            }
            Item::Use {
                name,
                callee,
                args,
                result,
                exit,
            } => match self.program.function(callee) {
                Some(function) => {
                    let function = function.clone();
                    self.call(context, i, &function, args, result, part)?;
                }
                None => self.use_(context, name, callee, args, result, exit.as_ref(), part)?,
            },
            Item::Lend {
                name,
                mutability,
                place,
                result,
            } => self.lend(context, name, *mutability, place, result)?,
            Item::Project {
                name,
                projection,
                operand,
                result,
            } => self.project(context, name, projection, operand, result)?,
            Item::Bind {
                name,
                position,
                binding,
                ..
            } => self.bind(context, name, position, binding)?,
            Item::CallValue {
                name,
                callee,
                args,
                result,
            } => self.call_value(context, name, callee, args, result)?,
        }
        Some(())
    }

    /// The value, at `level`, of the copy in the instance `made` of the class
    /// `name` of the group `function` lies in, with the class it is read at.
    fn copy(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        made: &Made,
        function: &Function,
        name: &Name,
    ) -> Option<Read> {
        let from = &self.certificate.sheets[&function.group];
        let id = from
            .id(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("{} names the class {name}", function.group));
        let Some(&copy) = made.made.get(&from.root(id)) else {
            self.violation(context, at, format!("{} copies no {name}", made.at));
            return None;
        };
        let lookup = context.lookup(level, copy);
        let value = self.get(context, at, &copy_name(&made.at, name).to_string(), lookup)?;
        Some(Read {
            value,
            at: Some(At { level, class: copy }),
        })
    }

    /// The lambda of the program a lambda is, or an instance copied.
    fn lambda_function(&self, lambda: &Name, copied: Option<&LambdaCopy>) -> Function {
        let of = copied.map_or(lambda, |copied| &copied.of);
        self.program
            .function(of)
            .expect("a lambda is a function")
            .clone()
    }

    /// The value of the parameter or result type at `i` in a lambda's
    /// environment: the text's class, or, for a lambda an instance copied,
    /// that instance's copy, with the class it is read at.
    fn environment(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        copied: Option<&LambdaCopy>,
        i: usize,
        name: &Name,
    ) -> Option<Read> {
        let Some(copied) = copied else {
            return self.text_read(context, at, name);
        };
        let class =
            copied.environment[i].expect("a copy's environment holds what the lambda's does");
        let lookup = context.lookup(level, class);
        let value = self.get(
            context,
            at,
            &copy_name(&copied.at, name).to_string(),
            lookup,
        )?;
        Some(Read {
            value,
            at: Some(At { level, class }),
        })
    }

    /// Each argument into its parameter's type.
    fn passes(&mut self, context: &Context<'_>, at: &str, args: &[Read], params: &[Param]) {
        for (k, (arg, param)) in args.iter().zip(params).enumerate() {
            self.pass(context, at, k, arg, param);
        }
    }

    /// The `k`th argument into its parameter's type.
    fn pass(&mut self, context: &Context<'_>, at: &str, k: usize, arg: &Read, param: &Param) {
        let holds = self.flows(context, at, arg, &param.ty, param.enters);
        let (arg, ty) = (&arg.value, &param.ty.value);
        self.require(
            context,
            at,
            holds,
            format!("argument {k}: {arg} into {ty} fails"),
        );
    }

    /// A function value passing into a function type: each lambda
    /// its Values names has an instance made where that coercion constraint was decomposed,
    /// as though the function type called it, each parameter of the type
    /// passing into the lambda's parameter and the lambda's result into the
    /// type's result.
    fn function_type(
        &mut self,
        context: &Context<'_>,
        at: &str,
        into: &IntoFunctionType,
        value: &Value,
        ty: &Value,
    ) {
        let Term::Function { params, ret } = ty.term() else {
            unreachable!("{ty} is a function type");
        };
        let Some(lambdas) = &value.lambdas else {
            unreachable!("{value} is a function value of a class, which names its Values");
        };
        let sheet = context.sheet(into.level);
        for lambda in lambdas {
            let copied = sheet.lambdas.get(lambda).cloned();
            let function = self.lambda_function(lambda, copied.as_ref());
            let made = sheet.instances.iter().find(|made| {
                made.group == function.group
                    && matches!(made.by, Recorder::FunctionType { from, to }
                        if sheet.root(from) == into.from && sheet.root(to) == into.to)
            });
            let Some(made) = made else {
                self.violation(
                    context,
                    at,
                    format!("no instance of {lambda} is made where {value} passes into {ty}"),
                );
                continue;
            };
            let Some(lambda_params) = function
                .params
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    if self.cut.in_environment(param) {
                        let ty =
                            self.environment(context, at, into.level, copied.as_ref(), i, param)?;
                        let enters = self.text.unknown(param);
                        return Some(Param { ty, enters });
                    }
                    let ty = self.copy(context, at, into.level, made, &function, param)?;
                    let enters = self.text.enters_copy(param);
                    Some(Param { ty, enters })
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let returned = if self.cut.in_environment(&function.result) {
                let i = function.params.len();
                self.environment(
                    context,
                    at,
                    into.level,
                    copied.as_ref(),
                    i,
                    &function.result,
                )
            } else {
                self.copy(context, at, into.level, made, &function, &function.result)
            };
            let Some(returned) = returned else {
                continue;
            };
            // The type's parameters and result, at the parts of the class the
            // type is read at.
            let ty_at = At {
                level: into.level,
                class: into.to,
            };
            let part = |key: PartKey, value: &Value| Read {
                value: value.clone(),
                at: context.part_at(ty_at, &[key]),
            };
            let type_params: Vec<Read> = params
                .iter()
                .enumerate()
                .map(|(i, param)| part(PartKey::Param(i), param))
                .collect();
            self.passes(context, at, &type_params, &lambda_params);
            let ret = part(PartKey::Return, ret);
            let holds = self.flows(context, at, &returned, &ret, into.ret_enters);
            let (returned, ret) = (&returned.value, &ret.value);
            self.require(
                context,
                at,
                holds,
                format!("the result of {lambda}: {returned} into {ret} fails"),
            );
        }
    }

    /// The result type into the result, as a lower bound.
    fn returns(&mut self, context: &Context<'_>, at: &str, returned: &Read, result: &Read) {
        let holds = self.flows(context, at, returned, result, true);
        let (returned, result) = (&returned.value, &result.value);
        self.require(context, at, holds, format!("{returned} <= {result} fails"));
    }

    /// A use of a function, at the positions of its constraints
    /// the context's group records: in its own group or an enclosing
    /// one, each argument passes into its universal parameter's type by a
    /// coercion constraint; from another group, into its copy in the instance
    /// made here.
    fn call(
        &mut self,
        context: &Context<'_>,
        i: usize,
        function: &Function,
        args: &[Name],
        result: &Name,
        part: &Part,
    ) -> Option<()> {
        let at = format!("the use {}@{i}", function.name);
        let held: Vec<(usize, &Name)> = args
            .iter()
            .enumerate()
            .filter(|(k, _)| part.holds(Position::Arg(*k)))
            .collect();
        let own = function.group == context.group
            || self
                .program
                .enclosing(&context.group)
                .contains(&function.group);
        let made = if own {
            None
        } else {
            let made_at = Name::new(&format!("{}@{i}", function.name));
            let Some(made) = context.own.instance(&made_at) else {
                self.violation(context, &at, "no instance is made".to_string());
                return None;
            };
            Some(made)
        };
        let param = |checker: &mut Self, name: &Name| -> Option<Param> {
            match made {
                None => {
                    let ty = checker.text_read(context, &at, name)?;
                    Some(Param { ty, enters: false })
                }
                Some(made) => {
                    let ty = checker.copy(context, &at, 0, made, function, name)?;
                    let enters = checker.text.enters_copy(name);
                    Some(Param { ty, enters })
                }
            }
        };
        for (k, arg) in held {
            let arg = self.text_read(context, &at, arg)?;
            let param = param(self, &function.params[k])?;
            self.pass(context, &at, k, &arg, &param);
        }
        if part.holds(Position::Result) {
            let result = self.text_read(context, &at, result)?;
            let returned = param(self, &function.result)?.ty;
            self.returns(context, &at, &returned, &result);
        }
        Some(())
    }

    /// A use of declarations, at the positions of its constraints the
    /// context's group records: of several candidates, their generalization
    /// at every position, which holds whatever candidate the use becomes
    /// (`generalized`); and what the use became: the candidate, by its
    /// signature, each argument passing whole into its parameter and its
    /// result entering the use's result, or `never`, by an argument that is
    /// `!`.
    #[allow(clippy::too_many_arguments)]
    fn use_(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        callee: &Name,
        args: &[Name],
        result: &Name,
        exit: Option<&Name>,
        part: &Part,
    ) -> Option<()> {
        let at = format!("the use {name}");
        let decls: Vec<&crate::decl::Decl> = self
            .decls
            .candidates(callee)
            .iter()
            .filter(|decl| decl.sig.params().len() == args.len())
            .collect();
        let candidates: Vec<&Signature> = decls.iter().map(|decl| &decl.sig).collect();
        let mut arg_reads: Vec<Option<Read>> = Vec::new();
        for (k, arg) in args.iter().enumerate() {
            let read = match part.holds(Position::Arg(k)) {
                true => Some(self.text_read(context, &at, arg)?),
                false => None,
            };
            arg_reads.push(read);
        }
        let result_read = match part.holds(Position::Result) {
            true => Some(self.text_read(context, &at, result)?),
            false => None,
        };
        let exit = exit.filter(|_| part.holds(Position::Exit));
        let generalization = (candidates.len() >= 2).then(|| Generalization::of(&candidates));
        if let Some(generalization) = &generalization {
            self.generalized(
                context,
                &at,
                name,
                generalization,
                &arg_reads,
                result_read.as_ref(),
            )?;
        }
        let ChosenAt {
            chosen,
            level,
            owner,
        } = self.chosen(context, &at, name)?;
        match chosen {
            Chosen::Candidate(id) => {
                let Some(one) = decls.iter().position(|decl| decl.id == id) else {
                    self.violation(context, &at, format!("{id:?} is no candidate"));
                    return None;
                };
                let sig = decls[one].sig.clone();
                match &generalization {
                    Some(generalization) => {
                        self.at_variables(context, &at, level, &owner, &sig, generalization, one)?;
                    }
                    None => {
                        self.at_positions(
                            context,
                            &at,
                            level,
                            &owner,
                            &sig,
                            &arg_reads,
                            result_read.as_ref(),
                        )?;
                    }
                }
                if let Some(exit) = exit {
                    self.exit(context, &at, level, &owner, &sig, exit)?;
                }
            }
            Chosen::Never => {
                let never = arg_reads
                    .iter()
                    .flatten()
                    .any(|read| *read.value.term() == Term::Never);
                self.require(
                    context,
                    &at,
                    never,
                    "no argument it reads is `!`".to_string(),
                );
            }
        }
        Some(())
    }

    /// The one candidate `sig` of a use, at the positions its group records:
    /// each argument passing whole into its parameter, entering it where it is
    /// a variable, and its result type entering the use's result, its
    /// variables the values of the classes it made for them.
    #[allow(clippy::too_many_arguments)]
    fn at_positions(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
        args: &[Option<Read>],
        result: Option<&Read>,
    ) -> Option<()> {
        let written: Vec<&SigTy> = sig
            .params()
            .iter()
            .zip(args)
            .filter(|(_, read)| read.is_some())
            .map(|(param, _)| param)
            .chain(result.map(|_| sig.result()))
            .collect();
        let bound = self.bound(context, at, level, owner, sig, &written)?;
        for (k, param) in sig.params().iter().enumerate() {
            let Some(arg) = &args[k] else {
                continue;
            };
            let Ok(ty) = param.substitute(&bound) else {
                self.violation(context, at, format!("parameter {k} has no type"));
                return None;
            };
            let ty = Read {
                value: ty,
                at: Self::disjunction_at(context, level, owner, DisjunctionAt::Param(k)),
            };
            let enters = matches!(param, SigTy::Var(_));
            self.pass(context, at, k, arg, &Param { ty, enters });
        }
        if let Some(result) = result {
            let Ok(returned) = sig.result().substitute(&bound) else {
                self.violation(context, at, "the result has no type".to_string());
                return None;
            };
            let returned = Read {
                value: returned,
                at: Self::disjunction_at(context, level, owner, DisjunctionAt::Result),
            };
            self.returns(context, at, &returned, result);
        }
        Some(())
    }

    /// The one candidate `sig`, the `one`th, of a use of several, at the
    /// variables of their generalization: the value of each variable passing
    /// into the candidate's type there, entering it where that is a variable
    /// of its own, where a parameter writes the variable; and the
    /// candidate's type entering the variable where the result writes it.
    /// The candidate's variables are the values of the classes it made for
    /// them.
    #[allow(clippy::too_many_arguments)]
    fn at_variables(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
        generalization: &Generalization,
        one: usize,
    ) -> Option<()> {
        let written: Vec<&SigTy> = generalization
            .vars
            .iter()
            .map(|types| &types[one])
            .collect();
        let bound = self.bound(context, at, level, owner, sig, &written)?;
        for (k, ty) in written.iter().enumerate() {
            let made = DisjunctionAt::Generalized(k);
            let Some(class) = context.sheet(level).made(owner, &made) else {
                self.violation(context, at, format!("the use made no {owner}.{made:?}"));
                return None;
            };
            let lookup = context.lookup(level, class);
            let value = self.get(context, at, &format!("{owner}.{made:?}"), lookup)?;
            let variable = Read {
                value,
                at: Some(At { level, class }),
            };
            let Ok(chosen) = ty.substitute(&bound) else {
                self.violation(
                    context,
                    at,
                    format!("the candidate has no type at variable {k}"),
                );
                return None;
            };
            let chosen = Read {
                value: chosen,
                at: match ty {
                    SigTy::Var(var) => {
                        Self::disjunction_at(context, level, owner, DisjunctionAt::Var(*var))
                    }
                    SigTy::Term(_) => {
                        Self::disjunction_at(context, level, owner, DisjunctionAt::Chosen(k))
                    }
                },
            };
            if generalization.writes_in_params(k) {
                let enters = matches!(ty, SigTy::Var(_));
                let holds = self.flows(context, at, &variable, &chosen, enters);
                let (from, to) = (&variable.value, &chosen.value);
                self.require(
                    context,
                    at,
                    holds,
                    format!("variable {k}: {from} into {to} fails"),
                );
            }
            if generalization.writes_in_result(k) {
                self.returns(context, at, &chosen, &variable);
            }
        }
        Some(())
    }

    /// A use's constraints at every position the context's group records, by
    /// its candidates' generalization, which hold whatever candidate it
    /// becomes: each argument passing into the generalization's parameter
    /// type, entering it where that is a variable, and the generalization's
    /// result type entering the result, each variable the value of the class
    /// the generalization made for it.
    fn generalized(
        &mut self,
        context: &Context<'_>,
        at: &str,
        name: &Name,
        generalization: &Generalization,
        args: &[Option<Read>],
        result: Option<&Read>,
    ) -> Option<()> {
        let mut bound: SigArgs<Value> = SigArgs {
            types: vec![None; generalization.vars.len()],
            identity: None,
        };
        for (k, ty) in bound.types.iter_mut().enumerate() {
            let made = DisjunctionAt::Generalized(k);
            let Some(class) = context.sheet(0).made(name, &made) else {
                self.violation(context, at, format!("the use made no {name}.{made:?}"));
                return None;
            };
            let lookup = context.lookup(0, class);
            *ty = Some(self.get(context, at, &format!("{name}.{made:?}"), lookup)?);
        }
        // Where the generalization writes a variable, the class it is read at
        // is the variable's.
        let class_of = |ty: &SigTy, made: DisjunctionAt| match ty {
            SigTy::Var(SigVar::Type(k)) => {
                Self::disjunction_at(context, 0, name, DisjunctionAt::Generalized(*k))
            }
            _ => Self::disjunction_at(context, 0, name, made),
        };
        for (k, (arg, param)) in args.iter().zip(&generalization.params).enumerate() {
            let Some(arg) = arg else {
                continue;
            };
            let Ok(value) = param.substitute(&bound) else {
                self.violation(
                    context,
                    at,
                    format!("the generalization's parameter {k} has no type"),
                );
                return None;
            };
            let ty = Read {
                value,
                at: class_of(param, DisjunctionAt::GeneralizedParam(k)),
            };
            let enters = matches!(param, SigTy::Var(_));
            self.pass(context, at, k, arg, &Param { ty, enters });
        }
        if let Some(result) = result {
            let Ok(value) = generalization.result.substitute(&bound) else {
                self.violation(
                    context,
                    at,
                    "the generalization's result has no type".to_string(),
                );
                return None;
            };
            let returned = Read {
                value,
                at: class_of(&generalization.result, DisjunctionAt::GeneralizedResult),
            };
            self.returns(context, at, &returned, result);
        }
        Some(())
    }

    /// `e?`'s exit: its value is `Option⟨K⟩` where the use became
    /// `?(Option⟨T⟩)`, and `Result⟨O, X⟩` where it became
    /// `?(Result⟨T, E⟩)`, `E`'s value wrapped into `X`: passing into it as it
    /// is where it is an `Error` or `!`, into the payload of an `Error` whose
    /// one member is its name where it has one, and into that of an
    /// `Error⟨..⟩` otherwise; and that value passes into the class the text
    /// gives the exit, a lower bound where that is an unknown.
    fn exit(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
        exit: &Name,
    ) -> Option<()> {
        let class = |checker: &mut Self, made: DisjunctionAt| -> Option<Value> {
            let Some(id) = context.sheet(level).made(owner, &made) else {
                checker.violation(context, at, format!("the use made no {owner}.{made:?}"));
                return None;
            };
            let lookup = context.lookup(level, id);
            checker.get(context, at, &format!("{owner}.{made:?}"), lookup)
        };
        let value = class(self, DisjunctionAt::Exit)?;
        let exit_read = self.text_read(context, at, exit)?;
        let value_read = Read {
            value: value.clone(),
            at: Self::disjunction_at(context, level, owner, DisjunctionAt::Exit),
        };
        let enters = self.text.unknown(exit);
        let holds = self.flows(context, at, &value_read, &exit_read, enters);
        let exit_value = &exit_read.value;
        self.require(
            context,
            at,
            holds,
            format!("the exit {value} into {exit_value} fails"),
        );
        match (sig.params(), value.term()) {
            ([SigTy::Term(param)], Term::Option(_)) if matches!(**param, Term::Option(_)) => {}
            ([SigTy::Term(param)], Term::Result { err, .. })
                if matches!(**param, Term::Result { .. }) =>
            {
                let error = class(self, DisjunctionAt::Var(SigVar::Type(1)))?;
                let into = match error.term() {
                    term if crate::join::is_error(term) || matches!(term, Term::Never) => Some(err),
                    Term::Enum(name, _) | Term::Struct(name, _) | Term::Extern { name, .. } => {
                        match err.term() {
                            Term::Enum(error_name, variants)
                                if error_name.as_str() == crate::join::ERROR
                                    && variants.len() == 1 =>
                            {
                                variants.get(name)
                            }
                            _ => None,
                        }
                    }
                    _ => match err.term() {
                        Term::ErrorOf(payload) => Some(payload),
                        _ => None,
                    },
                };
                // The wrapping, at no class the checker reads: an error's
                // payload holds no function type.
                let holds = into.is_some_and(|into| {
                    let error = Read {
                        value: error.clone(),
                        at: None,
                    };
                    let into = Read {
                        value: into.clone(),
                        at: None,
                    };
                    self.flows(context, at, &error, &into, true)
                });
                self.require(
                    context,
                    at,
                    holds,
                    format!("{error} is not wrapped into {err}"),
                );
            }
            _ => self.violation(
                context,
                at,
                format!("the exit {value} is not the chosen candidate's"),
            ),
        }
        Some(())
    }

    /// The values of the variables the signature writes at the positions the
    /// use holds, `written`, those of the classes its candidate made for them.
    fn bound(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
        written: &[&SigTy],
    ) -> Option<SigArgs<Value>> {
        let mut bound: SigArgs<Value> = sig.unbound();
        let vars = (0..sig.type_params().len())
            .map(SigVar::Type)
            .chain(sig.names_identity().then_some(SigVar::Identity));
        for var in vars {
            let written = written.iter().any(|ty| ty.mentions(var));
            if !written {
                continue;
            }
            let made = DisjunctionAt::Var(var);
            let Some(id) = context.sheet(level).made(owner, &made) else {
                self.violation(context, at, format!("the use made no {owner}.{made:?}"));
                return None;
            };
            let lookup = context.lookup(level, id);
            let ty = self.get(context, at, &format!("{owner}.{made:?}"), lookup)?;
            match var {
                SigVar::Type(k) => bound.types[k] = Some(ty),
                SigVar::Identity => bound.identity = Some(ty),
            }
        }
        Some(bound)
    }

    /// `&ₘ p` decided as `Lend { target, mutability: n }` (D1): the decision
    /// reaches, `Place` with `n` the lend's own mutability, or `Through`
    /// with the place's value a reference `&ₖ t` and `n` the lesser of the
    /// lend's mutability and `k`; and the result's value is `&ₙ p`, or `&ₙ t`
    /// through the place, no reference to a reference.
    fn lend(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        mutability: Mutability,
        place: &Name,
        result: &Name,
    ) -> Option<()> {
        let at = format!("the lend {name}");
        let DecidedAt { decision, .. } = self.decided(context, &at, name)?;
        let Decision::Lend {
            target,
            mutability: lent,
        } = decision
        else {
            self.violation(context, &at, format!("not reached: {decision:?}"));
            return None;
        };
        let place = self.text(context, &at, place)?;
        let result = self.text(context, &at, result)?;
        let (expected, target_ty) = match (target, place.term()) {
            (LendTarget::Place, _) => (mutability, place.clone()),
            (LendTarget::Through, Term::Ref(of, inner)) => (mutability.min(*of), inner.clone()),
            (LendTarget::Through, _) => {
                self.violation(context, &at, format!("{place} is no reference"));
                return None;
            }
        };
        self.require(
            context,
            &at,
            lent == expected,
            format!("{lent:?} is not {expected:?}"),
        );
        let holds = Value::from_term(Term::Ref(lent, target_ty)).is_ok_and(|lent| lent == result);
        self.require(
            context,
            &at,
            holds,
            format!("{result} is not the reference lent"),
        );
        Some(())
    }

    /// A projection.
    fn project(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        projection: &Projection,
        operand: &Name,
        result: &Name,
    ) -> Option<()> {
        let at = format!("the projection {name}");
        let DecidedAt { decision, .. } = self.decided(context, &at, name)?;
        let operand = self.text(context, &at, operand)?;
        let result = self.text(context, &at, result)?;
        let holds = match decision {
            Decision::Applied => has(self.decls, &operand, projection, &result),
            Decision::AppliedToTarget => match operand.term() {
                Term::Ref(_, target) => has(self.decls, target, projection, &result),
                _ => false,
            },
            other => {
                self.violation(context, &at, format!("not reached: {other:?}"));
                return None;
            }
        };
        self.require(
            context,
            &at,
            holds,
            format!("{operand} does not have {projection:?} of {result}"),
        );
        Some(())
    }

    /// A binding.
    fn bind(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        position: &Name,
        binding: &Name,
    ) -> Option<()> {
        let at = format!("the binding {name}");
        let DecidedAt {
            decision,
            level,
            owner,
        } = self.decided(context, &at, name)?;
        let position_read = self.text_read(context, &at, position)?;
        let binding_read = self.text_read(context, &at, binding)?;
        let position = position_read.value.clone();
        let binding = binding_read.value.clone();
        let read = match decision {
            Decision::Read => position_read,
            Decision::BindLend { target, mutability } => {
                let target = match (target, position.term()) {
                    (LendTarget::Place, _) => position.clone(),
                    (LendTarget::Through, Term::Ref(_, inner)) => inner.clone(),
                    (LendTarget::Through, _) => {
                        self.violation(context, &at, format!("{position} is no reference"));
                        return None;
                    }
                };
                let Ok(lent) = Value::from_term(Term::Ref(mutability, target)) else {
                    self.violation(context, &at, "no reference is lent".to_string());
                    return None;
                };
                Read {
                    value: lent,
                    at: Self::disjunction_at(context, level, &owner, DisjunctionAt::Lent),
                }
            }
            other => {
                self.violation(context, &at, format!("not reached: {other:?}"));
                return None;
            }
        };
        let holds = self.flows(context, &at, &read, &binding_read, true);
        let read = &read.value;
        self.require(context, &at, holds, format!("{read} <= {binding} fails"));
        Some(())
    }

    /// A call of a function value.
    fn call_value(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        callee: &Name,
        args: &[Name],
        result: &Name,
    ) -> Option<()> {
        let at = format!("the call {name}");
        let DecidedAt {
            decision,
            level,
            owner,
        } = self.decided(context, &at, name)?;
        let Decision::Calls(lambdas) = decision else {
            self.violation(context, &at, format!("not reached: {decision:?}"));
            return None;
        };
        let arg_reads = args
            .iter()
            .map(|arg| self.text_read(context, &at, arg))
            .collect::<Option<Vec<Read>>>()?;
        let result_read = self.text_read(context, &at, result)?;
        let callee_value = self.text(context, &at, callee)?;
        let called = match callee_value.term() {
            Term::Ref(_, target) => target.clone(),
            _ => callee_value.clone(),
        };
        let Some(values) = called.lambdas.clone() else {
            self.violation(context, &at, format!("{callee_value} is no function value"));
            return None;
        };
        for lambda in values.difference(&lambdas) {
            self.violation(
                context,
                &at,
                format!("{callee}'s Values name {lambda}, which the call does not call"),
            );
        }
        for lambda in lambdas.difference(&values) {
            self.violation(
                context,
                &at,
                format!("the call calls {lambda}, which {callee}'s Values do not name"),
            );
        }
        for lambda in &lambdas {
            let copied = context.sheet(level).lambdas.get(lambda).cloned();
            let function = self.lambda_function(lambda, copied.as_ref());
            let made_at = Name::new(&format!("{owner}#{lambda}"));
            let Some(made) = context.sheet(level).instance(&made_at) else {
                self.violation(context, &at, format!("no instance of {lambda} is made"));
                continue;
            };
            let Some(params) = function
                .params
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    if self.cut.in_environment(param) {
                        let ty =
                            self.environment(context, &at, level, copied.as_ref(), i, param)?;
                        let enters = self.text.unknown(param);
                        return Some(Param { ty, enters });
                    }
                    let ty = self.copy(context, &at, level, made, &function, param)?;
                    let enters = self.text.enters_copy(param);
                    Some(Param { ty, enters })
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let returned = if self.cut.in_environment(&function.result) {
                let i = function.params.len();
                self.environment(context, &at, level, copied.as_ref(), i, &function.result)
            } else {
                self.copy(context, &at, level, made, &function, &function.result)
            };
            let Some(returned) = returned else {
                continue;
            };
            self.passes(context, &at, &arg_reads, &params);
            self.returns(context, &at, &returned, &result_read);
        }
        Some(())
    }
}

/// The paths of the function values in the value.
fn function_paths(value: &Value) -> Vec<Path> {
    if let Term::FunctionValue { .. } = value.term() {
        return vec![Vec::new()];
    }
    value
        .term()
        .parts()
        .into_iter()
        .flat_map(|(key, part)| {
            function_paths(part).into_iter().map(move |mut path| {
                path.insert(0, key.clone());
                path
            })
        })
        .collect()
}

/// The value at `path` in the value.
fn value_at<'v>(value: &'v Value, path: &[PartKey]) -> Option<&'v Value> {
    path.iter().try_fold(value, |at, key| at.part(key))
}

/// Where each function value of `from` reaches in `to` along `coercion`, as
/// the decomposition carries it, each as its path in `from` and its path in
/// `to`: at the same position where no stage moves it, a reborrow keeping the
/// target; along a cast's or a conversion's signature, from where its
/// parameter writes a variable to where its result writes it (`image`); and
/// at each part both have, by that part's coercion, where the structural stage
/// changes the members, one in a member dropped reaching nothing.
fn reach(
    decls: &Decls,
    from: &Value,
    to: &Value,
    coercion: &Coercion,
) -> Result<Vec<(Path, Path)>, String> {
    let same = |value: &Value| -> Vec<(Path, Path)> {
        function_paths(value)
            .into_iter()
            .map(|path| (path.clone(), path))
            .collect()
    };
    match coercion {
        Coercion::Never => Ok(Vec::new()),
        Coercion::Identity | Coercion::Undecided => Ok(same(from)),
        Coercion::Conversion(id) => image(&decls.decl(id).sig, from)
            .map(|(_, moved)| moved)
            .ok_or_else(|| format!("{from} does not pass into the conversion {id:?}'s parameter")),
        Coercion::Composition {
            reborrow,
            casts,
            structural,
        } => {
            let mut at = match (reborrow, from.term()) {
                (true, Term::Ref(_, target)) => {
                    Value::from_term(Term::Ref(Mutability::Shared, target.clone()))
                        .map_err(|error| format!("{from} is reborrowed to no type: {error:?}"))?
                }
                _ => from.clone(),
            };
            let mut carried = same(from);
            for cast in casts {
                let (up, moved) = image(&decls.decl(cast).sig, &at).ok_or_else(|| {
                    format!("{at} does not pass into the cast {cast:?}'s parameter")
                })?;
                carried = carried
                    .into_iter()
                    .flat_map(|(origin, path)| {
                        moved
                            .iter()
                            .filter(move |(before, _)| *before == path)
                            .map(move |(_, after)| (origin.clone(), after.clone()))
                    })
                    .collect();
                at = up;
            }
            let Some(Structural::Shape { parts, .. }) = structural else {
                return Ok(carried);
            };
            let mut reached = Vec::new();
            for (key, part_coercion) in parts {
                let (Some(at_part), Some(to_part)) = (at.part(key), to.part(key)) else {
                    return Err(format!(
                        "a structural stage changes the part {key:?} that {at} and {to} share"
                    ));
                };
                for (inner_from, inner_to) in reach(decls, at_part, to_part, part_coercion)? {
                    for (origin, path) in &carried {
                        if path.first() == Some(key) && path[1..] == inner_from[..] {
                            let mut to_path = vec![key.clone()];
                            to_path.extend(inner_to.iter().cloned());
                            reached.push((origin.clone(), to_path));
                        }
                    }
                }
            }
            Ok(reached)
        }
    }
}

/// What a cast, a conversion or a view gives where `value` passes into its
/// parameter, as the decomposition links them through its signature, and as a conversion
/// carries it: its result with each
/// variable its parameter writes bound to the part of `value` at the first
/// place it writes it, and where each function value of `value` at each place
/// the parameter writes a variable lies in it, from its path in `value`. None
/// where `value` is not of the parameter's shape.
fn image(sig: &Signature, value: &Value) -> Option<(Value, Vec<(Path, Path)>)> {
    let mut bound = BTreeMap::new();
    if !bind_sig(&sig.params()[0], value, &mut Vec::new(), &mut bound) {
        return None;
    }
    let mut moved = Vec::new();
    let image = build_sig(sig.result(), &bound, &mut Vec::new(), &mut moved)?;
    Some((image, moved))
}

/// Binds each variable `ty` writes to the part of `value` at the first place
/// it writes it, with the path of each place; false where `value` is not of
/// `ty`'s shape. A variable written twice takes one type, each place
/// holding its own Values.
fn bind_sig(
    ty: &SigTy,
    value: &Value,
    path: &mut Path,
    bound: &mut BTreeMap<SigVar, (Value, Vec<Path>)>,
) -> bool {
    match ty {
        SigTy::Var(var) => match bound.get_mut(var) {
            Some((held, paths)) => {
                paths.push(path.clone());
                held.ty() == value.ty()
            }
            None => {
                bound.insert(*var, (value.clone(), vec![path.clone()]));
                true
            }
        },
        SigTy::Term(term) => {
            let written = value.term();
            term.constructor() == written.constructor()
                && term.members() == written.members()
                && term.parts().into_iter().all(|(key, part)| {
                    let Some(at) = written.part(&key) else {
                        return false;
                    };
                    path.push(key);
                    let held = bind_sig(part, at, path, bound);
                    path.pop();
                    held
                })
        }
    }
}

/// The value `ty` writes with its variables bound, each function value of a
/// bound part kept in `moved` with its path at each place bound and its path
/// here.
fn build_sig(
    ty: &SigTy,
    bound: &BTreeMap<SigVar, (Value, Vec<Path>)>,
    path: &mut Path,
    moved: &mut Vec<(Path, Path)>,
) -> Option<Value> {
    match ty {
        SigTy::Var(var) => {
            let (value, froms) = bound.get(var)?;
            for from in froms {
                for inner in function_paths(value) {
                    let before = from.iter().chain(&inner).cloned().collect();
                    let after = path.iter().chain(&inner).cloned().collect();
                    moved.push((before, after));
                }
            }
            Some(value.clone())
        }
        SigTy::Term(term) => {
            let term = (**term)
                .clone()
                .try_map(|key, part| {
                    path.push(key);
                    let built = build_sig(&part, bound, path, moved);
                    path.pop();
                    built.ok_or(())
                })
                .ok()?;
            Value::from_term(term).ok()
        }
    }
}

/// Whether the value has the member or part the projection names, of the
/// value `part`.
fn has(decls: &Decls, value: &Value, projection: &Projection, part: &Value) -> bool {
    match (projection, value.term()) {
        (_, Term::Never) => true,
        (Projection::Deref, Term::Ref(_, target)) => target == part,
        (Projection::Field(field), Term::Object(fields)) => fields.get(field) == Some(part),
        (Projection::Field(field), Term::Struct(module, args)) => decls
            .types()
            .struct_decl(module)
            .and_then(|decl| decl.fields.iter().position(|name| name == field))
            .is_some_and(|i| args.get(i) == Some(part)),
        (Projection::Component(i), Term::Tuple(components)) => components.get(*i) == Some(part),
        (Projection::Elem, Term::Array(_, element) | Term::Slice(element)) => element == part,
        (Projection::Elem, Term::Extern { .. }) => {
            view_element(decls, value).as_ref() == Some(part)
        }
        (Projection::Payload { enum_name, variant }, term) => {
            let constructor = term.constructor();
            let named = enum_name
                .as_ref()
                .is_none_or(|name| constructor == Constructor::Enum(name.clone()));
            named
                && constructor
                    .payload_part(variant)
                    .and_then(|key| term.part(&key))
                    == Some(part)
        }
        _ => false,
    }
}

/// The element an extern-defined container indexes along its view, the cast
/// going up from a shared reference to it, with the Values of each function
/// value in it carried along the view's signature; none where it
/// has no view.
fn view_element(decls: &Decls, value: &Value) -> Option<Value> {
    let view = decls
        .tree()
        .up(&Node::Ref(Mutability::Shared, value.term().constructor()))?;
    let reference = Value::from_term(Term::Ref(Mutability::Shared, value.clone())).ok()?;
    let (returned, _) = image(&decls.decl(view).sig, &reference)?;
    match returned.term() {
        Term::Ref(_, slice) => match slice.term() {
            Term::Slice(element) => Some(element.clone()),
            _ => None,
        },
        _ => None,
    }
}
