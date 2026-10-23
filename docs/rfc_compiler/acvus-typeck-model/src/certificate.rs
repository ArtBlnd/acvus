//! The certificate an accepted order of solving leaves, and its checker
//! (`design/14-certificate.md`).
//!
//! The checker reads the program and the certificate alone, on closed types:
//! every relation the text and the decisions give is regenerated from the
//! program's items, never read from a state's log.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::class::{ClassId, ClassName};
use crate::coerce::{Coercion, Structural, along_le, coerce};
use crate::cut::Cut;
use crate::decl::{Decls, Node};
use crate::explore::Final;
use crate::instance::Made;
use crate::node::common::CommonPart;
use crate::node::{Decision, LendTarget, NodeAt};
use crate::program::{ENTRY, Function, Item, Origin, Part, Place, Position, Program, Projection};
use crate::relation::{Recorder, copy_name};
use crate::sig::{SigArgs, SigTy, SigVar, Signature};
use crate::solve::{LambdaCopy, State};
use crate::ty::{Constructor, HasTerm, Mutability, Name, PartKey, Term, Ty, TyError, check_term};

/// A closed value as the certificate gives it: a type each of whose function
/// values carries the lambdas its Values names, as a function value carries
/// its facts on its type (l.181, l.184).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Value {
    term: Rc<Term<Value>>,
    /// At a function value, the lambdas its Values names; none elsewhere. No
    /// declaration, text or context writes a function value (l.184), so every
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
/// carrying the lambdas its Values names (l.181, l.184); none where the class
/// has no closed value.
fn certified(state: &State, root: ClassId) -> Option<Value> {
    let ty = state.value(root)?;
    let (kind, _) = state.classes.held(root).kind.clone()?;
    let term = kind
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
    ids: BTreeMap<ClassName, ClassId>,
    roots: BTreeMap<ClassId, ClassId>,
    /// Each root's value.
    values: BTreeMap<ClassId, Option<Value>>,
    /// Each root's parts, by their roots, where it holds a kind: where the
    /// split linked a part's two sides (l.278).
    parts: BTreeMap<ClassId, Vec<(PartKey, ClassId)>>,
    /// The roots that are unknowns of the state's group (l.211), which a value
    /// passing into meets (l.219).
    unknowns: BTreeSet<ClassId>,
    decisions: BTreeMap<Name, Option<Decision>>,
    instances: Vec<Made>,
    /// The lambdas an instance made anew here, with their environments
    /// (l.215).
    lambdas: BTreeMap<Name, LambdaCopy>,
}

impl Sheet {
    fn of(state: &State) -> Sheet {
        let ids: BTreeMap<ClassName, ClassId> = state
            .arena
            .classes()
            .map(|(id, name)| (name.clone(), id))
            .collect();
        let roots: BTreeMap<ClassId, ClassId> =
            ids.values().map(|&id| (id, state.root(id))).collect();
        let distinct: BTreeSet<ClassId> = roots.values().copied().collect();
        let parts = distinct
            .iter()
            .map(|&root| {
                let parts = match &state.classes.held(root).kind {
                    Some((kind, _)) => kind
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
        let decisions = state
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.decision.clone()))
            .collect();
        Sheet {
            group: state.group.clone(),
            ids,
            roots,
            values,
            parts,
            unknowns,
            decisions,
            instances: state.instances.clone(),
            lambdas: state.lambdas.clone(),
        }
    }

    fn is_entry(&self) -> bool {
        self.group.as_str() == ENTRY
    }

    fn id(&self, name: &ClassName) -> Option<ClassId> {
        self.ids.get(name).copied()
    }

    fn root(&self, id: ClassId) -> ClassId {
        self.roots[&id]
    }

    /// A name of a member of the root.
    fn name_of(&self, root: ClassId) -> &ClassName {
        self.ids
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
    /// solved cut the lambdas' environments (l.186).
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
    /// instance or a node made is named as well as the text's.
    pub(crate) fn set_value(&mut self, group: &str, name: &str, value: Value) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let id = sheet
            .ids
            .iter()
            .find(|(class, _)| class.to_string() == name)
            .map(|(_, &id)| id)
            .unwrap_or_else(|| panic!("{group} has the class {name}"));
        let root = sheet.root(id);
        sheet.values.insert(root, Some(value));
    }

    /// Gives the node `node` of `group`'s state the decision `decision`.
    pub(crate) fn set_decision(&mut self, group: &str, node: &str, decision: Decision) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let decided = sheet
            .decisions
            .get_mut(&Name::new(node))
            .unwrap_or_else(|| panic!("{group} has the node {node}"));
        *decided = Some(decision);
    }
}

/// A relation the text or a decision gives that the certificate's values do
/// not satisfy, or a condition of l.290 they do not meet.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Violation {
    /// The binding group whose text gives it, and the instances through which
    /// the context reaches the state its values are read in.
    pub context: String,
    /// The item, node or class that gives it.
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
/// are read in (`design/14-certificate.md`, Domain). Level 0 is the group's
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
    /// with no value there: nothing is checked of it here (l.290).
    Absent,
    /// A class of the entry's state with no value.
    Missing,
}

/// A node's decision, the level it is made at, and the node's name there.
struct DecidedAt {
    decision: Decision,
    level: usize,
    owner: Name,
}

/// What a context gives a node of the group.
enum Decided {
    At(DecidedAt),
    /// Left undecided in C at the last level, which is not the entry's.
    InC,
}

/// Where a function value passes into a function type (l.197): the level of
/// the sheet that split it, the roots there of the two sides, and whether the
/// type's result is an unknown the lambda's result meets.
struct IntoFunctionType {
    level: usize,
    from: ClassId,
    to: ClassId,
    ret_meets: bool,
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

/// A parameter's type, and whether an argument meets it or passes into it by a
/// boundary (l.280).
struct Param {
    ty: Read,
    meets: bool,
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

    /// The decision of the group's node `name`: its own, or its copy's in an
    /// instance on the way.
    fn decision(&self, name: &Name) -> Result<Decided, String> {
        let mut name = name.clone();
        for level in 0..=self.last() {
            let sheet = self.sheet(level);
            let Some(decision) = sheet.decisions.get(&name) else {
                return Err(format!("{} has no node {name}", sheet.group));
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
            Err(format!("{name} is not decided (l.290)"))
        } else {
            Ok(Decided::InC)
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
/// equations equate, and which its text decides or makes universal (l.211).
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
    /// decided, and not universal (l.211).
    fn unknown(&self, name: &Name) -> bool {
        let root = self.root(name);
        !self.decided.contains(&root) && !self.universal.contains(&root)
    }

    /// Whether a value passing into `into` at `place` meets it (l.219;
    /// `design/09-group.md`, reading 8).
    fn meets(&self, program: &Program, cut: &Cut, group: &Name, into: &Name, place: Place) -> bool {
        match place {
            Place::ThroughRef => false,
            Place::ByName => self.unknown(into),
            Place::Slot => cut.owner(program, into) == group && self.unknown(into),
        }
    }

    /// Whether an argument meets the copy of a parameter's type in an
    /// instance: the copy is an unknown of the group the instance is recorded
    /// in unless the text decides the type (l.215).
    fn meets_copy(&self, param: &Name) -> bool {
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
                self.violation(context, at, format!("{of} has no value (l.290)"));
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

    /// The decided node's decision, none where it is left in C, recording a
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

    /// Where the class a node `owner` made at `made` is read at `level`, none
    /// where it made none there.
    fn node_at(context: &Context<'_>, level: usize, owner: &Name, made: NodeAt) -> Option<At> {
        let name = ClassName::Node(Rc::new(owner.clone()), made);
        let class = context.sheet(level).id(&name)?;
        Some(At { level, class })
    }

    /// A value passing into a type (l.219, l.280): `from ≤ to` where it meets
    /// it, `from ⇝ to` by a coercion otherwise, each function value in `from`
    /// carried along the coercion as the split carries it (`reach`, l.278).
    /// Where it reaches a function value, its lambdas are among that one's
    /// (l.181, l.184); where it reaches a function type, an instance of each of
    /// its lambdas was made where the split met the two, at the classes they
    /// are read at, as `function_type` checks (l.197). Returns whether the
    /// coercion and the Values hold; what an instance breaks is recorded there.
    fn flows(
        &mut self,
        context: &Context<'_>,
        at: &str,
        from: &Read,
        to: &Read,
        meets: bool,
    ) -> bool {
        let (from_ty, to_ty) = (from.value.ty(), to.value.ty());
        let coercion = if meets {
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
                             their Values (l.184)"
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

    /// Where the function value at `from_path` of `from` meets the function
    /// type at `to_path` of `to`, by the classes the two are read at, which the
    /// split related there (l.197, l.278).
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
                 (design/14-certificate.md, Open)",
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
                 function type at yet (l.197; design/14-certificate.md, Open)",
                from.value, to.value
            );
        }
        let ret_meets = context
            .part_at(ty, &[PartKey::Return])
            .is_some_and(|ret| context.sheet(ret.level).unknowns.contains(&ret.class));
        IntoFunctionType {
            level: value.level,
            from: value.class,
            to: ty.class,
            ret_meets,
        }
    }

    /// l.290: every class of the entry has a value, every node of the entry
    /// is decided, and every decided node is reached.
    fn acceptance(&mut self) {
        for (group, sheet) in &self.certificate.sheets {
            if sheet.is_entry() {
                for (&root, value) in &sheet.values {
                    if value.is_none() {
                        self.violations.push(Violation {
                            context: group.to_string(),
                            at: sheet.name_of(root).to_string(),
                            what: "no value (l.290)".to_string(),
                        });
                    }
                }
            }
            for (name, decision) in &sheet.decisions {
                if decision.is_none() && sheet.is_entry() {
                    self.violations.push(Violation {
                        context: group.to_string(),
                        at: name.to_string(),
                        what: "undecided (l.290)".to_string(),
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
                self.require(
                    context,
                    &at,
                    holds,
                    format!("{value} does not admit {lit} (l.127)"),
                );
            }
            Origin::Lambda(lambda) => {
                let arity = self
                    .program
                    .function(lambda)
                    .expect("a lambda is a function")
                    .params
                    .len();
                let holds = *value.term() == Term::FunctionValue { arity };
                self.require(
                    context,
                    &at,
                    holds,
                    format!("{value} is not Fn/{arity} (l.181)"),
                );
                // Each instance on the way that made the lambda anew names
                // its copy (l.215).
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
                    format!("{value}'s Values do not name {named_as} (l.181, l.215)"),
                );
            }
        }
        Some(())
    }

    /// The item's part the context's group records (l.186).
    fn item(&mut self, context: &Context<'_>, i: usize, item: &Item, part: &Part) -> Option<()> {
        match item {
            Item::Pass { value, into, place } => {
                let at = format!("the pass of {value} into {into}");
                let from = self.text_read(context, &at, value)?;
                let to = self.text_read(context, &at, into)?;
                let meets = self
                    .text
                    .meets(self.program, self.cut, &context.group, into, *place);
                let holds = self.flows(context, &at, &from, &to, meets);
                let way = if meets { "<=" } else { "~>" };
                let (from, to) = (&from.value, &to.value);
                self.require(
                    context,
                    &at,
                    holds,
                    format!("{from} {way} {to} fails (l.280)"),
                );
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
            self.violation(context, at, format!("{} copies no {name} (l.215)", made.at));
            return None;
        };
        let lookup = context.lookup(level, copy);
        let value = self.get(context, at, &copy_name(&made.at, name).to_string(), lookup)?;
        Some(Read {
            value,
            at: Some(At { level, class: copy }),
        })
    }

    /// The lambda of the program a lambda is, or an instance copied (l.215).
    fn lambda_function(&self, lambda: &Name, copied: Option<&LambdaCopy>) -> Function {
        let of = copied.map_or(lambda, |copied| &copied.of);
        self.program
            .function(of)
            .expect("a lambda is a function")
            .clone()
    }

    /// The value of the parameter or result type at `i` in a lambda's
    /// environment: the text's class, or, for a lambda an instance copied,
    /// that instance's copy (l.215), with the class it is read at.
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
        let holds = self.flows(context, at, arg, &param.ty, param.meets);
        let (arg, ty) = (&arg.value, &param.ty.value);
        self.require(
            context,
            at,
            holds,
            format!("argument {k}: {arg} into {ty} fails"),
        );
    }

    /// A function value passing into a function type (l.197): each lambda
    /// its Values names has an instance made where that boundary was split,
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
            unreachable!("{value} is a function value of a class, which names its Values (l.184)");
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
                    format!(
                        "no instance of {lambda} is made where {value} passes into {ty} (l.197)"
                    ),
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
                        let meets = self.text.unknown(param);
                        return Some(Param { ty, meets });
                    }
                    let ty = self.copy(context, at, into.level, made, &function, param)?;
                    let meets = self.text.meets_copy(param);
                    Some(Param { ty, meets })
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
            let holds = self.flows(context, at, &returned, &ret, into.ret_meets);
            let (returned, ret) = (&returned.value, &ret.value);
            self.require(
                context,
                at,
                holds,
                format!("the result of {lambda}: {returned} into {ret} fails (l.197)"),
            );
        }
    }

    /// The result type into the result, as a lower bound.
    fn returns(&mut self, context: &Context<'_>, at: &str, returned: &Read, result: &Read) {
        let holds = self.flows(context, at, returned, result, true);
        let (returned, result) = (&returned.value, &result.value);
        self.require(context, at, holds, format!("{returned} <= {result} fails"));
    }

    /// A use of a function (l.192–195), at the positions of its constraints
    /// the context's group records (l.186): in its own group or an enclosing
    /// one, each argument passes into its universal parameter's type by a
    /// boundary (l.508); from another group, into its copy in the instance
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
                self.violation(context, &at, "no instance is made (l.215)".to_string());
                return None;
            };
            Some(made)
        };
        let param = |checker: &mut Self, name: &Name| -> Option<Param> {
            match made {
                None => {
                    let ty = checker.text_read(context, &at, name)?;
                    Some(Param { ty, meets: false })
                }
                Some(made) => {
                    let ty = checker.copy(context, &at, 0, made, function, name)?;
                    let meets = checker.text.meets_copy(name);
                    Some(Param { ty, meets })
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

    /// A use of declarations (l.320–333), at the positions of its constraints
    /// the context's group records (l.186): where its candidates' common part
    /// writes one closed type, by that type, whatever the use becomes
    /// (l.246); at the others, which its node holds, by its decision. A use of
    /// one candidate records that candidate at once, where its text is
    /// recorded, as its common part, nothing left to decide (l.246, l.320).
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
        let common = (candidates.len() >= 2).then(|| CommonPart::of(&candidates));
        let closed = |position: Position| {
            common
                .as_ref()
                .is_some_and(|common| common.is_closed_at(position))
        };
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
        if let Some(common) = &common {
            self.closed(context, &at, name, common, &arg_reads, result_read.as_ref())?;
        }
        // The positions the node holds.
        let node_args: Vec<Option<Read>> = arg_reads
            .iter()
            .enumerate()
            .map(|(k, read)| read.clone().filter(|_| !closed(Position::Arg(k))))
            .collect();
        let node_result = result_read.filter(|_| !closed(Position::Result));
        let node_exit = exit.filter(|_| part.holds(Position::Exit));
        if node_args.iter().all(Option::is_none) && node_result.is_none() && node_exit.is_none() {
            return Some(());
        }
        let (decision, level, owner) = match decls.as_slice() {
            [one] => (Decision::Use(one.id.clone()), 0, name.clone()),
            _ => {
                let DecidedAt {
                    decision,
                    level,
                    owner,
                } = self.decided(context, &at, name)?;
                (decision, level, owner)
            }
        };
        // The common part is recorded where the use is, before it is decided
        // (l.246), so its classes are the group's own, at level 0.
        let common_values = match &common {
            Some(part) => Some(self.common(
                context,
                &at,
                0,
                name,
                part,
                &node_args,
                node_result.as_ref(),
            )?),
            None => None,
        };
        match decision {
            Decision::Use(id) => {
                let decl = self.decls.decl(&id);
                let candidate = decl.id.name == *callee && decl.sig.params().len() == args.len();
                self.require(
                    context,
                    &at,
                    candidate,
                    format!("{id:?} is no candidate (l.320)"),
                );
                let sig = decl.sig.clone();
                let written: Vec<&SigTy> = sig
                    .params()
                    .iter()
                    .zip(&node_args)
                    .filter(|(_, read)| read.is_some())
                    .map(|(param, _)| param)
                    .chain(node_result.as_ref().map(|_| sig.result()))
                    .collect();
                let bound = self.bound(context, &at, level, &owner, &sig, &written)?;
                for (k, param) in sig.params().iter().enumerate() {
                    let Some(arg) = &node_args[k] else {
                        continue;
                    };
                    let Ok(ty) = param.substitute(&bound) else {
                        self.violation(context, &at, format!("parameter {k} has no type"));
                        return None;
                    };
                    let ty = Read {
                        value: ty,
                        at: Self::node_at(context, level, &owner, NodeAt::Param(k)),
                    };
                    let meets = matches!(param, SigTy::Var(_));
                    self.pass(context, &at, k, arg, &Param { ty, meets });
                }
                if let Some(result) = &node_result {
                    let Ok(returned) = sig.result().substitute(&bound) else {
                        self.violation(context, &at, "the result has no type".to_string());
                        return None;
                    };
                    let returned = Read {
                        value: returned,
                        at: Self::node_at(context, level, &owner, NodeAt::Result),
                    };
                    self.returns(context, &at, &returned, result);
                }
                if let (Some(part), Some(values)) = (&common, &common_values) {
                    let candidate = decls
                        .iter()
                        .position(|decl| decl.id == id)
                        .expect("a candidate is among the candidates");
                    for k in CommonPart::vars_in(&part.result) {
                        let Ok(chosen) = part.vars[k][candidate].substitute(&bound) else {
                            self.violation(context, &at, format!("variable {k} has no type"));
                            return None;
                        };
                        let chosen = Read {
                            value: chosen,
                            at: Self::node_at(context, level, &owner, NodeAt::Chosen(k)),
                        };
                        let holds = self.flows(context, &at, &chosen, &values[k], true);
                        let (chosen, value) = (&chosen.value, &values[k].value);
                        self.require(
                            context,
                            &at,
                            holds,
                            format!("{chosen} <= {value} at the common part's {k} fails (l.246)"),
                        );
                    }
                }
                if let Some(exit) = node_exit {
                    self.exit(context, &at, level, &owner, &sig, exit)?;
                }
            }
            Decision::Never => {
                let never = State::differing(&candidates, args.len())
                    .into_iter()
                    .filter(|&k| k < args.len())
                    .any(|k| {
                        node_args[k]
                            .as_ref()
                            .is_some_and(|read| *read.value.term() == Term::Never)
                    });
                self.require(
                    context,
                    &at,
                    never,
                    "no argument it reads is `!` (l.333)".to_string(),
                );
            }
            other => self.violation(context, &at, format!("not reached: {other:?} (l.290)")),
        }
        Some(())
    }

    /// A use's constraints at the positions where its candidates' common part
    /// writes one closed type, which hold whatever the use becomes (l.186,
    /// l.246): each argument there passing into that type, at the class the
    /// common part made for it, and that type entering the result.
    fn closed(
        &mut self,
        context: &Context<'_>,
        at: &str,
        name: &Name,
        common: &CommonPart,
        args: &[Option<Read>],
        result: Option<&Read>,
    ) -> Option<()> {
        let none: SigArgs<Value> = SigArgs {
            types: vec![None; common.vars.len()],
            identity: None,
        };
        let closed = |ty: &SigTy| {
            ty.substitute(&none)
                .expect("a closed type of the common part writes no variable")
        };
        for (k, arg) in args.iter().enumerate() {
            let Some(arg) = arg else {
                continue;
            };
            if !common.is_closed_at(Position::Arg(k)) {
                continue;
            }
            let ty = Read {
                value: closed(&common.params[k]),
                at: Self::node_at(context, 0, name, NodeAt::CommonParam(k)),
            };
            self.pass(context, at, k, arg, &Param { ty, meets: false });
        }
        if let Some(result) = result
            && common.is_closed_at(Position::Result)
        {
            let returned = Read {
                value: closed(&common.result),
                at: Self::node_at(context, 0, name, NodeAt::CommonResult),
            };
            self.returns(context, at, &returned, result);
        }
        Some(())
    }

    /// `e?`'s exit (l.337): its value is `Option⟨K⟩` where the use is decided
    /// as `?(Option⟨T⟩)`, and `Result⟨O, X⟩` where it is decided as
    /// `?(Result⟨T, E⟩)`, `E`'s value wrapped into `X`: passing into it as it
    /// is where it is an `Error` or `!`, into the payload of an `Error` whose
    /// one member is its name where it has one, and into that of an
    /// `Error⟨..⟩` otherwise; and that value passes into the class the text
    /// gives the exit, a meeting where that is an unknown (l.219).
    fn exit(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
        exit: &Name,
    ) -> Option<()> {
        let node = |at: NodeAt| ClassName::Node(Rc::new(owner.clone()), at);
        let class = |checker: &mut Self, name: ClassName| -> Option<Value> {
            let Some(id) = context.sheet(level).id(&name) else {
                checker.violation(context, at, format!("the decision made no class {name}"));
                return None;
            };
            let lookup = context.lookup(level, id);
            checker.get(context, at, &name.to_string(), lookup)
        };
        let value = class(self, node(NodeAt::Exit))?;
        let exit_read = self.text_read(context, at, exit)?;
        let value_read = Read {
            value: value.clone(),
            at: Self::node_at(context, level, owner, NodeAt::Exit),
        };
        let meets = self.text.unknown(exit);
        let holds = self.flows(context, at, &value_read, &exit_read, meets);
        let exit_value = &exit_read.value;
        self.require(
            context,
            at,
            holds,
            format!("the exit {value} into {exit_value} fails (l.337)"),
        );
        match (sig.params(), value.term()) {
            ([SigTy::Term(param)], Term::Option(_)) if matches!(**param, Term::Option(_)) => {}
            ([SigTy::Term(param)], Term::Result { err, .. })
                if matches!(**param, Term::Result { .. }) =>
            {
                let error = class(self, node(NodeAt::Var(SigVar::Type(1))))?;
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
                    format!("{error} is not wrapped into {err} (l.337)"),
                );
            }
            _ => self.violation(
                context,
                at,
                format!("the exit {value} is not the decided candidate's (l.337)"),
            ),
        }
        Some(())
    }

    /// A use's common part (l.246): the values of the classes it made for its
    /// variables, each a reference or none where the common part records it
    /// so; each argument passing into the common part's parameter, as at a
    /// decided candidate's, a function value into a function type left to that
    /// check; and the common part's result entering the use's result. Returns
    /// the variables' values.
    #[allow(clippy::too_many_arguments)]
    fn common(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        part: &CommonPart,
        args: &[Option<Read>],
        result: Option<&Read>,
    ) -> Option<Vec<Read>> {
        let mut values = Vec::new();
        for k in 0..part.vars.len() {
            let name = ClassName::Node(Rc::new(owner.clone()), NodeAt::Common(k));
            let Some(id) = context.sheet(level).id(&name) else {
                self.violation(context, at, format!("the use made no class {name} (l.246)"));
                return None;
            };
            let lookup = context.lookup(level, id);
            let value = self.get(context, at, &name.to_string(), lookup)?;
            if let Some(reference) = part.is_reference(k) {
                let holds = *value.term() == Term::Never
                    || matches!(value.term(), Term::Ref(..)) == reference;
                let is = if reference {
                    "a reference"
                } else {
                    "no reference"
                };
                self.require(
                    context,
                    at,
                    holds,
                    format!("{name} is recorded as {is}, and its value is {value} (l.246)"),
                );
            }
            values.push(Read {
                value,
                at: Some(At { level, class: id }),
            });
        }
        let bound = SigArgs {
            types: values.iter().map(|read| Some(read.value.clone())).collect(),
            identity: None,
        };
        for (k, (param, arg)) in part.params.iter().zip(args).enumerate() {
            let Some(arg) = arg else {
                continue;
            };
            let Ok(ty) = param.substitute(&bound) else {
                self.violation(
                    context,
                    at,
                    format!("the common part's parameter {k} has no type"),
                );
                return None;
            };
            let ty = Read {
                value: ty,
                at: Self::node_at(context, level, owner, NodeAt::CommonParam(k)),
            };
            let meets = matches!(param, SigTy::Var(_));
            self.pass(context, at, k, arg, &Param { ty, meets });
        }
        if let Some(result) = result {
            let Ok(returned) = part.result.substitute(&bound) else {
                self.violation(
                    context,
                    at,
                    "the common part's result has no type".to_string(),
                );
                return None;
            };
            let returned = Read {
                value: returned,
                at: Self::node_at(context, level, owner, NodeAt::CommonResult),
            };
            self.returns(context, at, &returned, result);
        }
        Some(values)
    }

    /// The values of the variables the signature writes at the positions the
    /// node holds, `written`, those of the classes the decision made for them
    /// (l.320).
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
            let name = ClassName::Node(Rc::new(owner.clone()), NodeAt::Var(var));
            let Some(id) = context.sheet(level).id(&name) else {
                self.violation(context, at, format!("the decision made no class {name}"));
                return None;
            };
            let lookup = context.lookup(level, id);
            let ty = self.get(context, at, &name.to_string(), lookup)?;
            match var {
                SigVar::Type(k) => bound.types[k] = Some(ty),
                SigVar::Identity => bound.identity = Some(ty),
            }
        }
        Some(bound)
    }

    /// `&ₘ p` (l.296).
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
            self.violation(context, &at, format!("not reached: {decision:?} (l.290)"));
            return None;
        };
        let place = self.text(context, &at, place)?;
        let result = self.text(context, &at, result)?;
        let (expected, target_ty) = match (target, place.term()) {
            (LendTarget::Place, _) => (mutability, place.clone()),
            (LendTarget::Through(read), Term::Ref(of, inner)) => {
                let agrees = read.is_none_or(|read| read == *of);
                self.require(context, &at, agrees, format!("{place} is not of {read:?}"));
                (mutability.min(*of), inner.clone())
            }
            (LendTarget::Through(_), _) => {
                self.violation(context, &at, format!("{place} is no reference (l.296)"));
                return None;
            }
        };
        self.require(
            context,
            &at,
            lent == expected,
            format!("{lent:?} is not {expected:?} (l.296)"),
        );
        let holds = Value::from_term(Term::Ref(lent, target_ty)).is_ok_and(|lent| lent == result);
        self.require(
            context,
            &at,
            holds,
            format!("{result} is not the reference lent (l.296)"),
        );
        Some(())
    }

    /// A projection (l.144–154, l.298).
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
            Decision::AppliedToTarget(read) => match operand.term() {
                Term::Ref(of, target) => {
                    read.is_none_or(|read| read == *of)
                        && has(self.decls, target, projection, &result)
                }
                _ => false,
            },
            other => {
                self.violation(context, &at, format!("not reached: {other:?} (l.290)"));
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

    /// A binding (l.306).
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
                    (LendTarget::Through(_), Term::Ref(_, inner)) => inner.clone(),
                    (LendTarget::Through(_), _) => {
                        self.violation(context, &at, format!("{position} is no reference (l.306)"));
                        return None;
                    }
                };
                let Ok(lent) = Value::from_term(Term::Ref(mutability, target)) else {
                    self.violation(context, &at, "no reference is lent".to_string());
                    return None;
                };
                Read {
                    value: lent,
                    at: Self::node_at(context, level, &owner, NodeAt::Lent),
                }
            }
            other => {
                self.violation(context, &at, format!("not reached: {other:?} (l.290)"));
                return None;
            }
        };
        let holds = self.flows(context, &at, &read, &binding_read, true);
        let read = &read.value;
        self.require(
            context,
            &at,
            holds,
            format!("{read} <= {binding} fails (l.306)"),
        );
        Some(())
    }

    /// A call of a function value (l.181, l.196).
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
            self.violation(context, &at, format!("not reached: {decision:?} (l.290)"));
            return None;
        };
        let arg_reads = args
            .iter()
            .map(|arg| self.text_read(context, &at, arg))
            .collect::<Option<Vec<Read>>>()?;
        let result_read = self.text_read(context, &at, result)?;
        let callee_value = self.text(context, &at, callee)?;
        let Some(values) = callee_value.lambdas.clone() else {
            self.violation(
                context,
                &at,
                format!("{callee_value} is no function value (l.196)"),
            );
            return None;
        };
        for lambda in values.difference(&lambdas) {
            self.violation(
                context,
                &at,
                format!("{callee}'s Values name {lambda}, which the call does not call (l.196)"),
            );
        }
        for lambda in lambdas.difference(&values) {
            self.violation(
                context,
                &at,
                format!("the call calls {lambda}, which {callee}'s Values do not name (l.181)"),
            );
        }
        for lambda in &lambdas {
            let copied = context.sheet(level).lambdas.get(lambda).cloned();
            let function = self.lambda_function(lambda, copied.as_ref());
            let made_at = Name::new(&format!("{owner}#{lambda}"));
            let Some(made) = context.sheet(level).instance(&made_at) else {
                self.violation(
                    context,
                    &at,
                    format!("no instance of {lambda} is made (l.196)"),
                );
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
                        let meets = self.text.unknown(param);
                        return Some(Param { ty, meets });
                    }
                    let ty = self.copy(context, &at, level, made, &function, param)?;
                    let meets = self.text.meets_copy(param);
                    Some(Param { ty, meets })
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
/// the split carries it (l.278), each as its path in `from` and its path in
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
/// parameter, as the split links them through its signature (l.78, l.154,
/// l.278), and as a conversion carries it (l.280): its result with each
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
/// `ty`'s shape. A variable written twice takes one type (types §7), each place
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
/// value `part` (l.146–154).
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
/// value in it carried along the view's signature (l.78, l.154); none where it
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
