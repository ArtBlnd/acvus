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
use crate::decl::{Decls, Node};
use crate::explore::Final;
use crate::instance::Made;
use crate::node::{Decision, LendTarget, NodeAt};
use crate::program::{ENTRY, Function, Item, Origin, Place, Program, Projection};
use crate::relation::{Recorder, copy_name};
use crate::sig::{SigArgs, SigTy, SigVar, Signature};
use crate::solve::State;
use crate::ty::{Constructor, HasTerm, Mutability, Name, Term, Ty, TyError, check_term};

/// A closed value as the certificate gives it: a type each of whose function
/// values carries the lambdas its Values names, as a function value carries
/// its facts on its type (l.183, l.186).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Value {
    term: Rc<Term<Value>>,
    /// At a function value, the lambdas its Values names; none elsewhere. No
    /// declaration, text or context writes a function value (l.186), so every
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

    /// Whether a function value lies in it.
    fn holds_function(&self) -> bool {
        matches!(*self.term, Term::FunctionValue { .. })
            || self
                .term
                .parts()
                .into_iter()
                .any(|(_, part)| part.holds_function())
    }

    /// A value a test plants, which the case writes as a type and so with no
    /// Values: a function value in it is not decided yet.
    #[cfg(test)]
    fn planted(ty: &Ty) -> Value {
        if let Term::FunctionValue { .. } = ty.term() {
            panic!(
                "a planted {ty} names no Values, which a case does not write yet \
                 (design/14-certificate.md, Open)"
            );
        }
        Value {
            term: Rc::new(ty.term().clone().map(|_, part| Value::planted(&part))),
            lambdas: None,
        }
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
/// carrying the lambdas its Values names (l.183, l.186); none where the class
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
    decisions: BTreeMap<Name, Option<Decision>>,
    instances: Vec<Made>,
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
            decisions,
            instances: state.instances.clone(),
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
    accepted: bool,
}

impl Certificate {
    pub fn of(end: &Final<'_>) -> Certificate {
        let sheets = end
            .groups
            .iter()
            .map(|(name, state)| (name.clone(), Sheet::of(state)))
            .chain([(Name::new(ENTRY), Sheet::of(end.entry))])
            .collect();
        Certificate {
            sheets,
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
    pub(crate) fn set_value(&mut self, group: &str, name: &ClassName, value: Ty) {
        let sheet = self
            .sheets
            .get_mut(&Name::new(group))
            .unwrap_or_else(|| panic!("the certificate has a sheet of {group}"));
        let id = sheet
            .id(name)
            .unwrap_or_else(|| panic!("{group} has the class {name}"));
        let root = sheet.root(id);
        sheet.values.insert(root, Some(Value::planted(&value)));
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
/// not satisfy, or a condition of l.292 they do not meet.
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
    let text = Text::of(program);
    let mut checker = Checker {
        program,
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
    /// with no value there: nothing is checked of it here (l.292).
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

/// Where a function value passes into a function type (l.199): the level of
/// the sheet that split it, the roots there of the two sides, and whether the
/// type's result is an unknown the lambda's result meets.
struct IntoFunctionType {
    level: usize,
    from: ClassId,
    to: ClassId,
    ret_meets: bool,
}

/// A parameter's type, and whether an argument meets it or passes into it by a
/// boundary (l.282).
struct Param {
    ty: Value,
    meets: bool,
}

impl<'a> Context<'a> {
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
            Err(format!("{name} is not decided (l.292)"))
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
/// equations equate, and which its text decides or makes universal (l.213).
struct Text {
    parent: BTreeMap<Name, Name>,
    /// The roots of the classes that hold a type the text writes, gives from
    /// the host, or makes a lambda's value.
    decided: BTreeSet<Name>,
    /// The roots of the classes that are universal.
    universal: BTreeSet<Name>,
}

impl Text {
    fn of(program: &Program) -> Text {
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
                Origin::Written(_) | Origin::Host(_) | Origin::Lambda(_)
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
            .filter(|name| program.is_universal(name));
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
    /// decided, and not universal (l.213).
    fn unknown(&self, name: &Name) -> bool {
        let root = self.root(name);
        !self.decided.contains(&root) && !self.universal.contains(&root)
    }

    /// Whether a value passing into `into` at `place` meets it (l.221;
    /// `design/09-group.md`, reading 8).
    fn meets(&self, program: &Program, group: &Name, into: &Name, place: Place) -> bool {
        match place {
            Place::ThroughRef => false,
            Place::ByName => self.unknown(into),
            Place::Slot => program.owner(into) == group && self.unknown(into),
        }
    }

    /// Whether an argument meets the copy of a parameter's type in an
    /// instance: the copy is an unknown of the group the instance is recorded
    /// in unless the text decides the type (l.217).
    fn meets_copy(&self, param: &Name) -> bool {
        !self.decided.contains(&self.root(param))
    }
}

struct Checker<'a> {
    program: &'a Program,
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
                self.violation(context, at, format!("{of} has no value (l.292)"));
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

    /// `from ≤ to` (l.282), each function value's lambdas among those of the
    /// function value it reaches (l.183, l.186).
    fn lower(&self, from: &Value, to: &Value) -> bool {
        along_le(self.decls, &from.ty(), &to.ty())
            .is_ok_and(|coercion| carries(from, to, &coercion))
    }

    /// `from ⇝ to` by a coercion (l.282), each function value's lambdas among
    /// those of the function value it reaches (l.183, l.186).
    fn boundary(&self, from: &Value, to: &Value) -> bool {
        coerce(self.decls, &from.ty(), &to.ty()).is_ok_and(|coercion| carries(from, to, &coercion))
    }

    /// l.292: every class of the entry has a value, every node of the entry
    /// is decided, and every decided node is reached.
    fn acceptance(&mut self) {
        for (group, sheet) in &self.certificate.sheets {
            if sheet.is_entry() {
                for (&root, value) in &sheet.values {
                    if value.is_none() {
                        self.violations.push(Violation {
                            context: group.to_string(),
                            at: sheet.name_of(root).to_string(),
                            what: "no value (l.292)".to_string(),
                        });
                    }
                }
            }
            for (name, decision) in &sheet.decisions {
                if decision.is_none() && sheet.is_entry() {
                    self.violations.push(Violation {
                        context: group.to_string(),
                        at: name.to_string(),
                        what: "undecided (l.292)".to_string(),
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
            .filter(|class| *program.owner(&class.name) == context.group)
        {
            self.origin(context, &class.name, &class.origin);
        }
        for (i, (_, item)) in program.items.iter().enumerate() {
            if *program.item_group(i) == context.group {
                self.item(context, i, item);
            }
        }
    }

    fn origin(&mut self, context: &Context<'_>, name: &Name, origin: &Origin) -> Option<()> {
        let at = format!("the class {name}");
        let value = self.text(context, &at, name)?;
        match origin {
            Origin::Var => {}
            Origin::Written(term) | Origin::Host(term) => {
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
                    format!("{value} does not admit {lit} (l.129)"),
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
                    format!("{value} is not Fn/{arity} (l.183)"),
                );
                let named = value
                    .lambdas
                    .as_ref()
                    .is_some_and(|lambdas| lambdas.contains(lambda));
                self.require(
                    context,
                    &at,
                    named,
                    format!("{value}'s Values do not name {lambda} (l.183)"),
                );
            }
        }
        Some(())
    }

    fn item(&mut self, context: &Context<'_>, i: usize, item: &Item) -> Option<()> {
        match item {
            Item::Pass { value, into, place } => {
                let at = format!("the pass of {value} into {into}");
                let from = self.text(context, &at, value)?;
                let to = self.text(context, &at, into)?;
                if let (Term::FunctionValue { .. }, Term::Function { .. }) =
                    (from.term(), to.term())
                {
                    let Origin::Written(Term::Function { ret, .. }) =
                        &self.program.class(into).origin
                    else {
                        panic!(
                            "{into} holds a function type its text does not write, which no rule \
                             links a function value to yet (l.199; design/14-certificate.md, Open)"
                        );
                    };
                    let ret_meets = self.text.unknown(ret);
                    let value_id = context.own.id(&ClassName::Text(value.clone()));
                    let into_id = context.own.id(&ClassName::Text(into.clone()));
                    let (Some(value_id), Some(into_id)) = (value_id, into_id) else {
                        panic!("{} names the classes {value} and {into}", context.group);
                    };
                    let (level, from_root) = context.locate(0, value_id);
                    let (into_level, to_root) = context.locate(0, into_id);
                    if level != into_level {
                        panic!(
                            "{value} and {into} are read at two levels, which no rule links a \
                             function value to a function type at yet (l.199; \
                             design/14-certificate.md, Open)"
                        );
                    }
                    let into_type = IntoFunctionType {
                        level,
                        from: from_root,
                        to: to_root,
                        ret_meets,
                    };
                    self.function_type(context, &at, &into_type, &from, &to);
                } else if self.text.meets(self.program, &context.group, into, *place) {
                    let holds = self.lower(&from, &to);
                    self.require(context, &at, holds, format!("{from} <= {to} fails (l.282)"));
                } else {
                    let holds = self.boundary(&from, &to);
                    self.require(context, &at, holds, format!("{from} ~> {to} fails (l.282)"));
                }
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
            } => match self.program.function(callee) {
                Some(function) => {
                    let function = function.clone();
                    self.call(context, i, &function, args, result)?;
                }
                None => self.use_(context, name, callee, args, result)?,
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
    /// `name` of the group `function` lies in.
    fn copy(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        made: &Made,
        function: &Function,
        name: &Name,
    ) -> Option<Value> {
        let from = &self.certificate.sheets[&function.group];
        let id = from
            .id(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("{} names the class {name}", function.group));
        let Some(&copy) = made.made.get(&from.root(id)) else {
            self.violation(context, at, format!("{} copies no {name} (l.217)", made.at));
            return None;
        };
        let lookup = context.lookup(level, copy);
        self.get(context, at, &copy_name(&made.at, name).to_string(), lookup)
    }

    /// Each argument into its parameter's type.
    fn passes(&mut self, context: &Context<'_>, at: &str, args: &[Value], params: &[Param]) {
        for (k, (arg, param)) in args.iter().zip(params).enumerate() {
            self.pass(context, at, k, arg, param);
        }
    }

    /// The `k`th argument into its parameter's type.
    fn pass(&mut self, context: &Context<'_>, at: &str, k: usize, arg: &Value, param: &Param) {
        let holds = if param.meets {
            self.lower(arg, &param.ty)
        } else {
            self.boundary(arg, &param.ty)
        };
        let ty = &param.ty;
        self.require(
            context,
            at,
            holds,
            format!("argument {k}: {arg} into {ty} fails"),
        );
    }

    /// Where a use's `k`th argument, the text class `arg`, passes into the
    /// function type its decided candidate writes there, whose result the
    /// candidate writes as `ret`.
    fn argument_into_type(
        &mut self,
        context: &Context<'_>,
        level: usize,
        owner: &Name,
        k: usize,
        arg: &Name,
        ret: &SigTy,
    ) -> Option<IntoFunctionType> {
        let arg_id = context
            .own
            .id(&ClassName::Text(arg.clone()))
            .unwrap_or_else(|| panic!("{} names the class {arg}", context.group));
        let (arg_level, from) = context.locate(0, arg_id);
        if arg_level != level {
            panic!(
                "{arg} is read at another level than the use that passes it, which no rule links \
                 a function value to a function type at yet (l.199; design/14-certificate.md, Open)"
            );
        }
        let sheet = context.sheet(level);
        let param = ClassName::Node(Rc::new(owner.clone()), NodeAt::Param(k));
        let to = sheet.root(sheet.id(&param)?);
        Some(IntoFunctionType {
            level,
            from,
            to,
            ret_meets: matches!(ret, SigTy::Var(_)),
        })
    }

    /// A function value passing into a function type (l.199): each lambda
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
            unreachable!("{value} is a function value of a class, which names its Values (l.186)");
        };
        let sheet = context.sheet(into.level);
        for lambda in lambdas {
            let function = self
                .program
                .function(lambda)
                .expect("a lambda is a function")
                .clone();
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
                        "no instance of {lambda} is made where {value} passes into {ty} (l.199)"
                    ),
                );
                continue;
            };
            let Some(lambda_params) = function
                .params
                .iter()
                .map(|param| {
                    if self.program.in_environment(param) {
                        let ty = self.text(context, at, param)?;
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
            let returned = if self.program.in_environment(&function.result) {
                self.text(context, at, &function.result)
            } else {
                self.copy(context, at, into.level, made, &function, &function.result)
            };
            let Some(returned) = returned else {
                continue;
            };
            self.passes(context, at, params, &lambda_params);
            let holds = if into.ret_meets {
                self.lower(&returned, ret)
            } else {
                self.boundary(&returned, ret)
            };
            self.require(
                context,
                at,
                holds,
                format!("the result of {lambda}: {returned} into {ret} fails (l.199)"),
            );
        }
    }

    /// The result type into the result, as a lower bound.
    fn returns(&mut self, context: &Context<'_>, at: &str, returned: &Value, result: &Value) {
        let holds = self.lower(returned, result);
        self.require(context, at, holds, format!("{returned} <= {result} fails"));
    }

    /// A use of a function (l.194–197): in its own group or an enclosing one,
    /// each argument passes into its universal parameter's type by a boundary
    /// (l.510); from another group, into its copy in the instance.
    fn call(
        &mut self,
        context: &Context<'_>,
        i: usize,
        function: &Function,
        args: &[Name],
        result: &Name,
    ) -> Option<()> {
        let at = format!("the use {}@{i}", function.name);
        let arg_values = args
            .iter()
            .map(|arg| self.text(context, &at, arg))
            .collect::<Option<Vec<Value>>>()?;
        let result_value = self.text(context, &at, result)?;
        let own = function.group == context.group
            || self
                .program
                .enclosing(&context.group)
                .contains(&function.group);
        let (params, returned) = if own {
            let params = function
                .params
                .iter()
                .map(|param| {
                    let ty = self.text(context, &at, param)?;
                    Some(Param { ty, meets: false })
                })
                .collect::<Option<Vec<_>>>()?;
            (params, self.text(context, &at, &function.result)?)
        } else {
            let made_at = Name::new(&format!("{}@{i}", function.name));
            let Some(made) = context.own.instance(&made_at) else {
                self.violation(context, &at, "no instance is made (l.217)".to_string());
                return None;
            };
            let params = function
                .params
                .iter()
                .map(|param| {
                    let ty = self.copy(context, &at, 0, made, function, param)?;
                    let meets = self.text.meets_copy(param);
                    Some(Param { ty, meets })
                })
                .collect::<Option<Vec<_>>>()?;
            let returned = self.copy(context, &at, 0, made, function, &function.result)?;
            (params, returned)
        };
        self.passes(context, &at, &arg_values, &params);
        self.returns(context, &at, &returned, &result_value);
        Some(())
    }

    /// A use of declarations (l.322–335), by its decision.
    fn use_(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        callee: &Name,
        args: &[Name],
        result: &Name,
    ) -> Option<()> {
        let at = format!("the use {name}");
        let DecidedAt {
            decision,
            level,
            owner,
        } = self.decided(context, &at, name)?;
        let arg_values = args
            .iter()
            .map(|arg| self.text(context, &at, arg))
            .collect::<Option<Vec<Value>>>()?;
        let result_value = self.text(context, &at, result)?;
        let decided = match &decision {
            Decision::Use(id) => Some(&self.decls.decl(id).sig),
            _ => None,
        };
        self.references(context, &at, callee, decided, &arg_values, &result_value);
        match decision {
            Decision::Use(id) => {
                let decl = self.decls.decl(&id);
                let candidate = decl.id.name == *callee && decl.sig.params().len() == args.len();
                self.require(
                    context,
                    &at,
                    candidate,
                    format!("{id:?} is no candidate (l.322)"),
                );
                let sig = decl.sig.clone();
                let bound = self.bound(context, &at, level, &owner, &sig)?;
                for (k, (param, arg)) in sig.params().iter().zip(args).enumerate() {
                    let Ok(ty) = param.substitute(&bound) else {
                        self.violation(context, &at, format!("parameter {k} has no type"));
                        return None;
                    };
                    let arg_value = &arg_values[k];
                    match (arg_value.term(), param) {
                        (Term::FunctionValue { .. }, SigTy::Term(written))
                            if matches!(**written, Term::Function { .. }) =>
                        {
                            let Term::Function { ret, .. } = &**written else {
                                unreachable!("the parameter writes a function type");
                            };
                            let into_type =
                                self.argument_into_type(context, level, &owner, k, arg, ret)?;
                            self.function_type(context, &at, &into_type, arg_value, &ty);
                        }
                        _ => {
                            let meets = matches!(param, SigTy::Var(_));
                            self.pass(context, &at, k, arg_value, &Param { ty, meets });
                        }
                    }
                }
                let Ok(returned) = sig.result().substitute(&bound) else {
                    self.violation(context, &at, "the result has no type".to_string());
                    return None;
                };
                self.returns(context, &at, &returned, &result_value);
            }
            Decision::Never => {
                let candidates: Vec<&Signature> = self
                    .decls
                    .candidates(callee)
                    .iter()
                    .filter(|decl| decl.sig.params().len() == args.len())
                    .map(|decl| &decl.sig)
                    .collect();
                let never = State::differing(&candidates, args.len())
                    .into_iter()
                    .filter(|&k| k < args.len())
                    .any(|k| *arg_values[k].term() == Term::Never);
                self.require(
                    context,
                    &at,
                    never,
                    "no argument it reads is `!` (l.335)".to_string(),
                );
            }
            other => self.violation(context, &at, format!("not reached: {other:?} (l.292)")),
        }
        Some(())
    }

    /// l.248: a use records that a class is, or is not, a reference where
    /// every candidate would record a reference, or none would, into it, and
    /// otherwise as its decided candidate would. A value `!` holds either, as
    /// the solver reads it (`Classes::normalize`).
    fn references(
        &mut self,
        context: &Context<'_>,
        at: &str,
        callee: &Name,
        decided: Option<&Signature>,
        args: &[Value],
        result: &Value,
    ) {
        let per_candidate: Vec<Option<bool>> = self
            .decls
            .candidates(callee)
            .iter()
            .filter(|decl| decl.sig.params().len() == args.len())
            .map(|decl| would_record(&decl.sig))
            .collect();
        let Some((&first, rest)) = per_candidate.split_first() else {
            return;
        };
        let agreed = first.filter(|reference| rest.iter().all(|other| *other == Some(*reference)));
        let recorded = decided.and_then(would_record).or(agreed);
        if let Some(reference) = recorded {
            let holds = *result.term() == Term::Never
                || matches!(result.term(), Term::Ref(..)) == reference;
            let is = if reference {
                "a reference"
            } else {
                "no reference"
            };
            self.require(
                context,
                at,
                holds,
                format!("Result is recorded as {is}, and its value is {result} (l.248)"),
            );
        }
    }

    /// The values of the variables the signature writes, those of the classes
    /// the decision made for them (l.322).
    fn bound(
        &mut self,
        context: &Context<'_>,
        at: &str,
        level: usize,
        owner: &Name,
        sig: &Signature,
    ) -> Option<SigArgs<Value>> {
        let mut bound: SigArgs<Value> = sig.unbound();
        let vars = (0..sig.type_params().len())
            .map(SigVar::Type)
            .chain(sig.names_identity().then_some(SigVar::Identity));
        for var in vars {
            let written = sig
                .params()
                .iter()
                .chain([sig.result()])
                .any(|ty| ty.mentions(var));
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

    /// `&ₘ p` (l.298).
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
            self.violation(context, &at, format!("not reached: {decision:?} (l.292)"));
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
                self.violation(context, &at, format!("{place} is no reference (l.298)"));
                return None;
            }
        };
        self.require(
            context,
            &at,
            lent == expected,
            format!("{lent:?} is not {expected:?} (l.298)"),
        );
        let holds = Value::from_term(Term::Ref(lent, target_ty)).is_ok_and(|lent| lent == result);
        self.require(
            context,
            &at,
            holds,
            format!("{result} is not the reference lent (l.298)"),
        );
        Some(())
    }

    /// A projection (l.146–156, l.300).
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
                self.violation(context, &at, format!("not reached: {other:?} (l.292)"));
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

    /// A binding (l.308).
    fn bind(
        &mut self,
        context: &Context<'_>,
        name: &Name,
        position: &Name,
        binding: &Name,
    ) -> Option<()> {
        let at = format!("the binding {name}");
        let DecidedAt { decision, .. } = self.decided(context, &at, name)?;
        let position = self.text(context, &at, position)?;
        let binding = self.text(context, &at, binding)?;
        let read = match decision {
            Decision::Read => position,
            Decision::BindLend { target, mutability } => {
                let target = match (target, position.term()) {
                    (LendTarget::Place, _) => position.clone(),
                    (LendTarget::Through(_), Term::Ref(_, inner)) => inner.clone(),
                    (LendTarget::Through(_), _) => {
                        self.violation(context, &at, format!("{position} is no reference (l.308)"));
                        return None;
                    }
                };
                let Ok(lent) = Value::from_term(Term::Ref(mutability, target)) else {
                    self.violation(context, &at, "no reference is lent".to_string());
                    return None;
                };
                lent
            }
            other => {
                self.violation(context, &at, format!("not reached: {other:?} (l.292)"));
                return None;
            }
        };
        let holds = self.lower(&read, &binding);
        self.require(
            context,
            &at,
            holds,
            format!("{read} <= {binding} fails (l.308)"),
        );
        Some(())
    }

    /// A call of a function value (l.183, l.198).
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
            self.violation(context, &at, format!("not reached: {decision:?} (l.292)"));
            return None;
        };
        let arg_values = args
            .iter()
            .map(|arg| self.text(context, &at, arg))
            .collect::<Option<Vec<Value>>>()?;
        let result_value = self.text(context, &at, result)?;
        let callee_value = self.text(context, &at, callee)?;
        let Some(values) = callee_value.lambdas.clone() else {
            self.violation(
                context,
                &at,
                format!("{callee_value} is no function value (l.198)"),
            );
            return None;
        };
        for lambda in values.difference(&lambdas) {
            self.violation(
                context,
                &at,
                format!("{callee}'s Values name {lambda}, which the call does not call (l.198)"),
            );
        }
        for lambda in lambdas.difference(&values) {
            self.violation(
                context,
                &at,
                format!("the call calls {lambda}, which {callee}'s Values do not name (l.183)"),
            );
        }
        for lambda in &lambdas {
            let function = self
                .program
                .function(lambda)
                .expect("a lambda is a function")
                .clone();
            let made_at = Name::new(&format!("{owner}#{lambda}"));
            let Some(made) = context.sheet(level).instance(&made_at) else {
                self.violation(
                    context,
                    &at,
                    format!("no instance of {lambda} is made (l.198)"),
                );
                continue;
            };
            let Some(params) = function
                .params
                .iter()
                .map(|param| {
                    if self.program.in_environment(param) {
                        let ty = self.text(context, &at, param)?;
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
            let returned = if self.program.in_environment(&function.result) {
                self.text(context, &at, &function.result)
            } else {
                self.copy(context, &at, level, made, &function, &function.result)
            };
            let Some(returned) = returned else {
                continue;
            };
            self.passes(context, &at, &arg_values, &params);
            self.returns(context, &at, &returned, &result_value);
        }
        Some(())
    }
}

/// Whether a candidate of signature `sig` would record a reference into the
/// use's result, where the type it would record there is one it writes
/// (l.248; note 08, reading 13). The result is the one place a candidate
/// records a type it writes into: each argument passes into its parameter's
/// type, a boundary recording into that type, and each part of the
/// argument's passes into that type's part, a meeting where the signature
/// writes a variable, so that what is recorded into the argument's part is
/// an equation with the variable, and a boundary into a type it writes
/// elsewhere (l.280).
fn would_record(sig: &Signature) -> Option<bool> {
    match sig.result() {
        SigTy::Term(term) => Some(matches!(**term, Term::Ref(..))),
        SigTy::Var(_) => None,
    }
}

/// Whether each function value in `from` has its lambdas among those of the
/// function value of `to` it reaches along `coercion`, a function value's
/// Values being part of its type (l.183, l.186): at the same position where no
/// stage changes it, a reborrow keeping the target; at each part both have, by
/// that part's coercion, where the structural stage changes the members; and
/// at the payload where the value is wrapped. Where Values would cross a cast
/// or a conversion, or a position decides nothing, no rule is decided yet.
fn carries(from: &Value, to: &Value, coercion: &Coercion) -> bool {
    let undecided = |what: String| -> ! {
        panic!("{what} carries Values by no rule yet (l.186; design/14-certificate.md, Open)")
    };
    match coercion {
        Coercion::Identity => within(from, to),
        Coercion::Never => true,
        Coercion::Undecided => undecided(format!(
            "{from} reaching {to} where a position decides nothing"
        )),
        Coercion::Conversion(id) => {
            if from.holds_function() || to.holds_function() {
                undecided(format!("the conversion {id:?} from {from} to {to}"));
            }
            true
        }
        Coercion::Composition {
            casts, structural, ..
        } => {
            if !casts.is_empty() {
                if from.holds_function() || to.holds_function() {
                    undecided(format!("the casts {casts:?} from {from} to {to}"));
                }
                return true;
            }
            match structural {
                None => within(from, to),
                Some(Structural::Shape { parts, .. }) => parts.iter().all(|(key, coercion)| {
                    let (Some(from), Some(to)) = (from.part(key), to.part(key)) else {
                        panic!(
                            "a structural stage changes the part {key:?} that {from} and {to} share"
                        );
                    };
                    carries(from, to, coercion)
                }),
                Some(Structural::Wrap(variant, inner)) => {
                    let Some(payload) = to
                        .term()
                        .constructor()
                        .payload_part(variant)
                        .and_then(|key| to.part(&key))
                    else {
                        panic!("{to} wraps a value in {variant}, which it has no payload of");
                    };
                    carries(from, payload, inner)
                }
            }
        }
    }
}

/// Whether each function value in `from` has its lambdas among those of the
/// function value at the same position of `to`, where the two are of one
/// type but for a reborrowed reference's mutability.
fn within(from: &Value, to: &Value) -> bool {
    if let Term::FunctionValue { .. } = from.term() {
        let (Some(lambdas), Some(reached)) = (&from.lambdas, &to.lambdas) else {
            unreachable!(
                "{from} and {to} are function values of classes, which name their Values (l.186)"
            );
        };
        return lambdas.is_subset(reached);
    }
    from.term()
        .parts()
        .into_iter()
        .all(|(key, part)| to.part(&key).is_some_and(|reached| within(part, reached)))
}

/// Whether the value has the member or part the projection names, of the
/// value `part` (l.148–156).
fn has(decls: &Decls, value: &Value, projection: &Projection, part: &Value) -> bool {
    match (projection, value.term()) {
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
            if value.holds_function() || part.holds_function() {
                panic!(
                    "the element of {value} along its view, a cast, carries Values by no rule \
                     yet (l.186; design/14-certificate.md, Open)"
                );
            }
            view_element(decls, &value.ty()) == Some(part.ty())
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
/// going up from a shared reference to it (l.78, l.156); none where it has no
/// view.
fn view_element(decls: &Decls, ty: &Ty) -> Option<Ty> {
    let view = decls
        .tree()
        .up(&Node::Ref(Mutability::Shared, ty.term().constructor()))?;
    let sig = &decls.decl(view).sig;
    let mut bound: SigArgs<Ty> = sig.unbound();
    let reference = Ty::new(Term::Ref(Mutability::Shared, ty.clone())).ok()?;
    if !sig.params().first()?.match_ty(&reference, &mut bound) {
        return None;
    }
    let returned = sig.result().substitute(&bound).ok()?;
    match returned.term() {
        Term::Ref(_, slice) => match slice.term() {
            Term::Slice(element) => Some(element.clone()),
            _ => None,
        },
        _ => None,
    }
}
