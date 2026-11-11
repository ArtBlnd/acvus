//! Recording a program: each statement as the operation of the store it writes.
//!
//! Nothing is judged here. A name is the class it was first written as, a path
//! names parts on it, a type gives the class that has it, and a statement records
//! one constraint between the classes its operands give.
//!
//! What a declaration writes is recorded before any other statement, as it comes
//! from outside the text: so the order of the statements is the order
//! constraints are recorded in, and nothing else.

use std::collections::{BTreeMap, BTreeSet};

use crate::operation::Outcome;
use crate::operation::language::Language;
use crate::operation::language::functions::Functions;
use crate::operation::language::lambdas::{Defined, Lambda, Lambdas};
use crate::operation::settle::settle;
use crate::primitive::class::{ClassKind, Live, Origin, Side, State, Template};
use crate::primitive::constraint::{Bound, Constraint, Suspended};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{Mutability, PartKey, Scalar, Shape, Ty, VariantOf};
use crate::primitive::scope::Scopes;
use crate::primitive::{ClassId, ItemId, LambdaId, Name, ScopeId, TemplateId};
use crate::store::Store;
use crate::store::classes::Classes;

use super::ast::{self, Assertion, Op, Path, Program, Segment, Type, Variant};
use super::print;

/// A program recorded, and then solved.
#[derive(Debug)]
pub struct Recorded {
    pub store: Store<Live>,
    pub law: Language,
    pub outcome: Outcome<Live>,
    items: Items,
    names: BTreeMap<String, ClassId>,
    /// The classes the text made, in the order it made them.
    made: Vec<ClassId>,
}

/// The items of a text, each with what it is found by: the label of its
/// statement, or the statement's number where it has none, and for a projection
/// that and the path it names, as `first:r.x`. A statement of a body is numbered
/// within the statement that writes the lambda, as `3.1`.
#[derive(Debug, Default)]
struct Items {
    found_by: Arena<ItemId, String>,
    labelled: BTreeMap<String, ItemId>,
}

impl Items {
    fn item(&mut self, found_by: String) -> ItemId {
        let item = self.found_by.push(found_by.clone());
        self.labelled.entry(found_by).or_insert(item);
        item
    }
}

/// What the class of a lambda written in some classes is: where the program is
/// written it has the function value of a closure named there, and in a body it
/// is a class the body's template says a lambda is defined at.
type Define<'a, K> = dyn FnMut(&mut Classes<K>, LambdaId, Vec<<K as ClassKind>::Id>, ItemId) -> <K as ClassKind>::Id + 'a;

/// Statements being written into classes: those of the program, or those of the
/// template of a lambda's body. The two are written the same way, and differ in
/// what a lambda written there is ([`Define`]).
struct Writer<'a, K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    classes: &'a mut Classes<K>,
    scope: ScopeId,
    /// The class each name stands for: the one it was first written as.
    names: &'a mut BTreeMap<String, K::Id>,
    /// The classes written, in the order they were.
    made: &'a mut Vec<K::Id>,
    /// Each part named on a class decided as a head with no place for it.
    not_held: &'a mut Vec<Refused<K>>,
    items: &'a mut Items,
    lambdas: &'a mut Lambdas,
    define: &'a mut Define<'a, K>,
}

impl<K> Writer<'_, K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    fn undecided(&mut self) -> K::Id {
        let class = self.classes.undecided(self.scope);
        self.made.push(class);
        class
    }

    /// The class `name` stands for: the one it was first written as.
    fn named(&mut self, name: &str) -> K::Id {
        if let Some(class) = self.names.get(name) {
            return *class;
        }
        let class = self.undecided();
        self.names.insert(name.to_owned(), class);
        class
    }

    /// The class `path` gives, each of its segments naming a part on `side`, by
    /// an item found by `label` and the path so far.
    ///
    /// A part a decided class has no place for is refused as solving refuses one,
    /// and the path goes on from a class of its own.
    fn path(&mut self, path: &Path, side: Side, label: &str) -> K::Id {
        let mut class = self.named(&path.root);
        let mut so_far = Path {
            root: path.root.clone(),
            segments: Vec::new(),
        };
        for segment in &path.segments {
            so_far.segments.push(segment.clone());
            let item = self.items.item(format!("{label}:{so_far}"));
            class = match self.classes.part(class, key(segment), side, item) {
                Some(part) => {
                    self.made.push(part);
                    part
                }
                None => {
                    self.not_held.push(Refused {
                        cause: Cause::NotHeld,
                        items: BTreeSet::from([item]),
                        of: vec![class],
                    });
                    self.undecided()
                }
            };
        }
        class
    }

    /// The class that has the type `ty`, written by `origin`, by `item`.
    fn class_of(&mut self, ty: &Type, origin: Origin, item: ItemId, label: &str) -> K::Id {
        let literal = |writer: &mut Self, scalars: &[Scalar]| {
            let class = writer.undecided();
            let pending = writer.classes.domain(class, Domain(scalars.iter().copied().collect()), item);
            debug_assert_eq!(pending, None, "a class just made is undecided");
            class
        };
        let shape = match ty {
            Type::Class(path) => return self.path(path, Side::Upper, label),
            Type::Integer => return literal(self, &INTEGERS),
            Type::Float => return literal(self, &[Scalar::F32, Scalar::F64]),
            written => shape(written).map(|part| self.class_of(part, origin, item, label)),
        };
        let by = BTreeSet::from([item]);
        match origin {
            Origin::Free => self.classes.decided(self.scope, shape, by),
            Origin::External => self.classes.external(self.scope, shape, by),
        }
    }

    /// `callee(arguments)`, giving `result`: the callee is passed to the function
    /// type the call writes.
    fn call(&mut self, callee: K::Id, arguments: Vec<K::Id>, result: K::Id, item: ItemId) {
        let written = self.classes.decided(self.scope, Shape::Fn(arguments, result), BTreeSet::from([item]));
        self.classes.constrain(
            Bound::Change {
                lower: callee,
                upper: written,
            }
            .into(),
            item,
        );
    }

    /// Writes the template of `lambda`, and gives it with the classes written
    /// here that it captures.
    fn lambda(&mut self, lambda: &ast::Lambda, label: &str) -> Result<(LambdaId, Vec<K::Id>), String> {
        let mut template = Classes::<Template>::default();
        let mut names = BTreeMap::new();
        let scope = self.scope;
        let mut open = |names_of: &[String]| -> Result<Vec<TemplateId>, String> {
            let mut classes = Vec::new();
            for name in names_of {
                let class = template.undecided(scope);
                if names.insert(name.clone(), class).is_some() {
                    return Err(format!("`{name}` is named twice among what a lambda captures, takes and gives"));
                }
                classes.push(class);
            }
            Ok(classes)
        };
        let captures = open(&lambda.captures)?;
        let parameters = open(&lambda.parameters)?;
        let result = open(std::slice::from_ref(&lambda.result))?[0];

        let (mut made, mut not_held, mut defines) = (Vec::new(), Vec::new(), Vec::new());
        let mut define = |classes: &mut Classes<Template>, lambda: LambdaId, captures: Vec<TemplateId>, by: ItemId| {
            let value = classes.undecided(scope);
            defines.push(Defined {
                lambda,
                captures,
                value,
                by,
            });
            value
        };
        let mut body = Writer {
            classes: &mut template,
            scope,
            names: &mut names,
            made: &mut made,
            not_held: &mut not_held,
            items: &mut *self.items,
            lambdas: &mut *self.lambdas,
            define: &mut define,
        };
        for (index, statement) in lambda.body.iter().enumerate() {
            let label = statement.label.clone().unwrap_or_else(|| format!("{label}.{}", index + 1));
            body.statement(&statement.op, &label)?;
        }
        debug_assert!(not_held.is_empty(), "a body names parts on classes not decided");

        let written = self.lambdas.write(Lambda {
            template,
            parameters,
            result,
            captures,
            defines,
        });
        let captured = lambda.captures.iter().map(|name| self.named(name)).collect();
        Ok((written, captured))
    }

    /// Records the one operation `op` writes. A declaration is no statement of
    /// what is solved: what it writes comes from outside the text.
    fn statement(&mut self, op: &Op, label: &str) -> Result<(), String> {
        match op {
            Op::Enter { upper, lower, alone } => {
                let item = self.items.item(label.to_owned());
                let lower = self.class_of(lower, Origin::Free, item, label);
                let upper = self.path(upper, Side::Lower, label);
                let constraint = if *alone {
                    Suspended::Alone { lower, upper }.into()
                } else {
                    Bound::Enter { lower, upper }.into()
                };
                self.classes.constrain(constraint, item);
            }
            Op::Pass { argument, given } => {
                let item = self.items.item(label.to_owned());
                let lower = self.path(argument, Side::Upper, label);
                let upper = self.path(given, Side::Upper, label);
                self.classes.constrain(Bound::Change { lower, upper }.into(), item);
            }
            Op::Exit { exits_with, error } => {
                let item = self.items.item(label.to_owned());
                let lower = self.path(exits_with, Side::Upper, label);
                let upper = self.path(error, Side::Lower, label);
                self.classes.constrain(Suspended::Exit { lower, upper }.into(), item);
            }
            Op::Define { upper, lambda } => {
                let item = self.items.item(label.to_owned());
                let (written, captured) = self.lambda(lambda, label)?;
                let value = (self.define)(self.classes, written, captured, item);
                let upper = self.path(upper, Side::Lower, label);
                self.classes.constrain(Bound::Enter { lower: value, upper }.into(), item);
                // A definition is a use that passes nothing: a call with
                // arguments nothing enters.
                let nothing = lambda.parameters.iter().map(|_| self.undecided()).collect();
                let result = self.undecided();
                self.call(value, nothing, result, item);
            }
            Op::Call {
                callee,
                arguments,
                result,
            } => {
                let item = self.items.item(label.to_owned());
                let callee = self.path(callee, Side::Upper, label);
                let arguments = arguments.iter().map(|argument| self.path(argument, Side::Upper, label)).collect();
                let result = self.path(result, Side::Lower, label);
                self.call(callee, arguments, result, item);
            }
            Op::Declare { .. } | Op::Row { .. } => return Err("a declaration is written where a statement is".to_owned()),
        }
        Ok(())
    }
}

impl Recorded {
    fn new() -> Self {
        Self {
            store: Store::default(),
            law: Language::default(),
            outcome: Outcome::default(),
            items: Items::default(),
            names: BTreeMap::new(),
            made: Vec::new(),
        }
    }

    /// Solves what is recorded, asking the classes the text made in the order
    /// `order` names: as made, reversed, or shuffled by it.
    pub fn settle(&mut self, order: u64) {
        let mut asked = self.made.clone();
        match order {
            0 => {}
            1 => asked.reverse(),
            seed => {
                let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                for i in (1..asked.len()).rev() {
                    state = state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                    let j = usize::try_from(state >> 33).expect("a shifted word fits") % (i + 1);
                    asked.swap(i, j);
                }
            }
        }
        let at = self.items.item("settling".to_owned());
        let visit = |classes: &Classes<Live>| -> Vec<ClassId> {
            asked.iter().copied().chain(classes.roots().map(|root| root.id())).collect()
        };
        settle(&mut self.store, &self.law, &mut self.outcome, at, &visit);
    }

    /// The class `path` gives as the store stands, recording nothing: none where
    /// a name was never written, or a class has no part of a segment's name.
    pub fn class(&self, path: &Path) -> Option<ClassId> {
        let mut class = *self.names.get(&path.root)?;
        for segment in &path.segments {
            class = self.store.classes.get(self.store.classes.find(class)).state.part(&key(segment))?;
        }
        Some(class)
    }

    /// The type of `class` as the text writes one, `?` standing for a class not
    /// decided.
    pub fn written(&self, class: ClassId) -> String {
        self.written_to(class, 16)
    }

    fn written_to(&self, class: ClassId, depth: usize) -> String {
        if depth == 0 {
            return "...".to_owned();
        }
        let State::Decided(decided) = &self.store.classes.get(self.store.classes.find(class)).state else {
            return "?".to_owned();
        };
        let at = |part: &ClassId| self.written_to(*part, depth - 1);
        let members = |members: &BTreeMap<Name, ClassId>| {
            let members: Vec<String> = members.iter().map(|(variant, payload)| format!("{}({})", variant.0, at(payload))).collect();
            format!("{{{}}}", members.join(", "))
        };
        let listed = |parts: &[ClassId]| parts.iter().map(at).collect::<Vec<_>>().join(", ");
        match &decided.shape {
            Shape::Ref(Mutability::Shared, target) => format!("&{}", at(target)),
            Shape::Ref(Mutability::Exclusive, target) => format!("&mut {}", at(target)),
            Shape::Tuple(components) if components.len() == 1 => format!("({},)", listed(components)),
            Shape::Tuple(components) => format!("({})", listed(components)),
            Shape::Unit => "()".to_owned(),
            Shape::Array(length, element) => format!("[{}; {length}]", at(element)),
            Shape::Slice(element) => format!("[{}]", at(element)),
            Shape::Option(payload) => format!("Option<{}>", at(payload)),
            Shape::Result(ok, error) => format!("Result<{}, {}>", at(ok), at(error)),
            Shape::Error(variants) => format!("Error{}", members(variants)),
            Shape::ErrorOf(error) => format!("Error<{}>", at(error)),
            Shape::Object(fields) => {
                let fields: Vec<String> = fields.iter().map(|(field, part)| format!("{}: {}", field.0, at(part))).collect();
                format!("{{{}}}", fields.join(", "))
            }
            Shape::Enum(name, variants) => format!("{}{}", name.0, members(variants)),
            Shape::Struct(name, arguments) if arguments.is_empty() => name.0.clone(),
            Shape::Struct(name, arguments) => format!("{}<{}>", name.0, listed(arguments)),
            Shape::Scalar(scalar) => print::scalar(*scalar).to_owned(),
            Shape::String => "String".to_owned(),
            Shape::Str => "str".to_owned(),
            Shape::Never => "!".to_owned(),
            // The text writes none of these, so no class it records has one.
            other @ (Shape::Fn(..) | Shape::FnValue(_) | Shape::Extern(..)) => format!("{:?}", other.layer()),
        }
    }

    /// Whether solving came to what `assertion` says, and what it came to where
    /// it did not.
    pub fn check(&self, assertion: &Assertion) -> Result<(), String> {
        let item = |label: &String| self.items.labelled.get(label).copied().ok_or(format!("no statement is labelled `{label}`"));
        match assertion {
            Assertion::Type { class, ty } => {
                let found = self.class(class).ok_or(format!("`{class}` is no class"))?;
                let (written, expected) = (self.written(found), ty.to_string());
                if written == expected { Ok(()) } else { Err(format!("`{class}` is `{written}`")) }
            }
            Assertion::Accepted | Assertion::Refused { .. } => {
                let refused: BTreeSet<Cause> = self.outcome.refusals.iter().map(|refusal| refusal.cause).collect();
                let holds = match assertion {
                    Assertion::Refused { causes, others: false } => refused == *causes,
                    Assertion::Refused { causes, others: true } => !refused.is_empty() && refused.is_superset(causes),
                    _ => refused.is_empty(),
                };
                match (holds, refused.is_empty()) {
                    (true, _) => Ok(()),
                    (false, true) => Err("accepted".to_owned()),
                    (false, false) => Err(format!("refused by {refused:?}")),
                }
            }
            Assertion::Names { cause, labels } => {
                let mut refusals = self.outcome.refusals.iter().filter(|refusal| refusal.cause == *cause);
                let (Some(refusal), None) = (refusals.next(), refusals.next()) else {
                    return Err(format!("not one refusal by {cause:?}"));
                };
                let expected = labels.iter().map(item).collect::<Result<BTreeSet<ItemId>, String>>()?;
                if refusal.connections == expected {
                    Ok(())
                } else {
                    let named: Vec<&String> = refusal.connections.iter().map(|at| self.items.found_by.get(*at)).collect();
                    Err(format!("the refusal by {cause:?} names {named:?}"))
                }
            }
            Assertion::Takes { label, at, adjustments } => {
                let found_by = at.as_ref().map_or(label.clone(), |at| format!("{label}:{at}"));
                let taken = self.outcome.adjustments.get(&item(&found_by)?).cloned().unwrap_or_default();
                if taken == *adjustments { Ok(()) } else { Err(format!("`{found_by}` takes {taken:?}")) }
            }
        }
    }
}

const INTEGERS: [Scalar; 9] = [
    Scalar::I8,
    Scalar::I16,
    Scalar::I32,
    Scalar::I64,
    Scalar::U8,
    Scalar::U16,
    Scalar::U32,
    Scalar::U64,
    Scalar::Usize,
];

fn key(segment: &Segment) -> PartKey {
    let payload = |of: VariantOf, variant: &str| PartKey::Payload(of, Name::new(variant));
    match segment {
        Segment::Field(field) => PartKey::Field(Name::new(field)),
        Segment::Component(index) => PartKey::Component(*index),
        Segment::Target => PartKey::Target,
        Segment::Element => PartKey::Elem,
        Segment::Payload(Variant::Some) => payload(VariantOf::Option, "Some"),
        Segment::Payload(Variant::Ok) => payload(VariantOf::Result, "Ok"),
        Segment::Payload(Variant::Err) => payload(VariantOf::Result, "Err"),
        Segment::Payload(Variant::Error(variant)) => payload(VariantOf::Error, variant),
        Segment::Payload(Variant::Enum(of, variant)) => payload(VariantOf::Enum(Name::new(of)), variant),
    }
}

/// The head `ty` writes, with the types of its parts. A path and a literal write
/// no head: each gives a class of its own.
fn shape(ty: &Type) -> Shape<&Type> {
    fn named(members: &BTreeMap<String, Type>) -> BTreeMap<Name, &Type> {
        members.iter().map(|(name, ty)| (Name::new(name), ty)).collect()
    }
    match ty {
        Type::Class(_) | Type::Integer | Type::Float => unreachable!("a path and a literal write no head"),
        Type::Never => Shape::Never,
        Type::Scalar(scalar) => Shape::Scalar(*scalar),
        Type::String => Shape::String,
        Type::Str => Shape::Str,
        Type::Ref(mutability, target) => Shape::Ref(*mutability, target),
        Type::Tuple(components) if components.is_empty() => Shape::Unit,
        Type::Tuple(components) => Shape::Tuple(components.iter().collect()),
        Type::Array(length, element) => Shape::Array(*length, element),
        Type::Slice(element) => Shape::Slice(element),
        Type::Option(payload) => Shape::Option(payload),
        Type::Result(ok, error) => Shape::Result(ok, error),
        Type::Error(variants) => Shape::Error(named(variants)),
        Type::ErrorOf(error) => Shape::ErrorOf(error),
        Type::Object(fields) => Shape::Object(named(fields)),
        Type::Enum(name, variants) => Shape::Enum(Name::new(name), named(variants)),
        Type::Named(name, arguments) => Shape::Struct(Name::new(name), arguments.iter().collect()),
    }
}

/// The type a row writes: a name among `holes` is a hole. A row is a signature
/// of its own, so it names no class of the program and holds no literal.
fn signature(ty: &Type, holes: &[String]) -> Result<Ty, String> {
    match ty {
        Type::Class(path) => Err(format!("a row names the class `{path}`")),
        Type::Integer | Type::Float => Err("a row holds a literal".to_owned()),
        Type::Named(name, arguments) if arguments.is_empty() && holes.contains(name) => Ok(Ty::Hole(Name::new(name))),
        written => {
            let mut failed = None;
            let shape = shape(written).map(|part| match signature(part, holes) {
                Ok(part) => part,
                Err(error) => {
                    failed.get_or_insert(error);
                    Ty::Hole(Name::new(""))
                }
            });
            failed.map_or(Ok(Ty::Node(Box::new(shape))), Err)
        }
    }
}

/// Records `program`: what its declarations write, and then each constraint in
/// the order of its statements. It is no program to record where a name is
/// declared twice or by a type that names it, or a row names a class or holds a
/// literal.
pub fn record(program: &Program) -> Result<Recorded, String> {
    let mut recorded = Recorded::new();
    let labelled: Vec<String> = program
        .statements
        .iter()
        .enumerate()
        .map(|(index, statement)| statement.label.clone().unwrap_or_else(|| (index + 1).to_string()))
        .collect();
    let Recorded {
        store: Store { classes, scopes, closures },
        law: Language { functions, lambdas },
        outcome,
        items,
        names,
        made,
    } = &mut recorded;
    let root = scopes.root();
    let mut not_held = Vec::new();
    // Where the program is written a lambda is a closure named at once, over the
    // classes it captures, and its class has that function value.
    let mut define = |classes: &mut Classes<Live>, lambda: LambdaId, captures: Vec<ClassId>, by: ItemId| {
        let closure = closures.name(lambda, captures);
        classes.decided(root, Shape::FnValue(BTreeSet::from([closure])), BTreeSet::from([by]))
    };
    let mut writer = Writer {
        classes,
        scope: root,
        names,
        made,
        not_held: &mut not_held,
        items,
        lambdas,
        define: &mut define,
    };
    for (statement, label) in program.statements.iter().zip(&labelled) {
        declare(&mut writer, functions, scopes, &statement.op, label)?;
    }
    for (statement, label) in program.statements.iter().zip(&labelled) {
        if !matches!(statement.op, Op::Declare { .. } | Op::Row { .. }) {
            writer.statement(&statement.op, label)?;
        }
    }
    for refused in not_held {
        outcome.refuse(refused);
    }
    Ok(recorded)
}

/// Records what a declaration writes: a class of the name it declares, or a row
/// of the table of functions. Any other statement declares nothing.
fn declare(writer: &mut Writer<'_, Live>, functions: &mut Functions, scopes: &Scopes, op: &Op, label: &str) -> Result<(), String> {
    match op {
        Op::Declare { name, ty } => {
            if writer.names.contains_key(name) {
                return Err(format!("`{name}` is declared twice"));
            }
            let item = writer.items.item(label.to_owned());
            let class = writer.class_of(ty, Origin::External, item, label);
            // Writing the type made a class of the name where the type names
            // it: a declaration writes a finite type, and none that holds
            // itself.
            if writer.names.insert(name.clone(), class).is_some() {
                return Err(format!("`{name}` is declared as a type that names it"));
            }
        }
        Op::Row {
            name,
            holes,
            parameter,
            result,
        } => {
            let item = writer.items.item(label.to_owned());
            let (parameter, result) = (signature(parameter, holes)?, signature(result, holes)?);
            functions.declare(Name::new(name), &parameter, &result, scopes, item);
        }
        Op::Enter { .. } | Op::Pass { .. } | Op::Exit { .. } | Op::Define { .. } | Op::Call { .. } => {}
    }
    Ok(())
}

/// Why a text did not run to what it asserts.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Failed {
    /// It is no program: it does not parse, or cannot be recorded.
    Unread(String),
    /// It is a program, and solving it comes to something else than it asserts,
    /// or to one thing in one order and another in another.
    Differs(String),
}

/// Reads `text`, and solves it in each of `orders` orders of asking: what it
/// asserts holds in each, and every order comes to the same refusals. A program
/// that asserts neither that it is accepted nor that it is refused is to be
/// accepted.
pub fn run(text: &str, orders: u64) -> Result<(), Failed> {
    let program = super::parse(text).map_err(Failed::Unread)?;
    let accepted = Assertion::Accepted;
    let says = program
        .assertions
        .iter()
        .any(|assertion| matches!(assertion, Assertion::Accepted | Assertion::Refused { .. }));
    let assertions = program.assertions.iter().chain((!says).then_some(&accepted));
    let mut first = None;
    for order in 0..orders {
        let mut recorded = record(&program).map_err(Failed::Unread)?;
        recorded.settle(order);
        if let Err(violations) = recorded.store.classes.verify() {
            return Err(Failed::Differs(format!("in order {order}, the store is broken: {violations:?}")));
        }
        for assertion in assertions.clone() {
            recorded
                .check(assertion)
                .map_err(|found| Failed::Differs(format!("in order {order}, `assert {assertion};` does not hold: {found}")))?;
        }
        let mut refusals: Vec<_> = recorded
            .outcome
            .refusals
            .iter()
            .map(|refusal| (refusal.cause, refusal.items.clone(), refusal.connections.clone()))
            .collect();
        refusals.sort();
        let first = first.get_or_insert_with(|| refusals.clone());
        if refusals != *first {
            return Err(Failed::Differs(format!("order {order} is refused otherwise than order 0: {refusals:?} and {first:?}")));
        }
    }
    Ok(())
}
