//! A program, as the constraints its text gives (l.219–231): the types its
//! text writes and makes, split into equivalence classes (l.120), and each
//! value passing into a type at a place, each in the binding group its text
//! lies in (l.203–215).

use std::collections::{BTreeMap, BTreeSet};

use crate::decl::Decls;
use crate::lit::LitVar;
use crate::ty::{Mutability, Name, Term};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct Program {
    pub decls: Decls,
    /// The binding groups other than the run's entry, which holds the text
    /// outside them (l.199), each with the group it lies inside, a lambda's
    /// being its enclosing body's (l.23).
    pub groups: Vec<(Name, Option<Name>)>,
    pub functions: Vec<Function>,
    /// The classes the text makes universal besides its parameters' types: a
    /// receiver's `T̄` (l.314).
    pub universals: Vec<Name>,
    pub classes: Vec<TextClass>,
    pub items: Vec<(Name, Item)>,
}

/// A position of a use (l.186): an argument, the result, or `e?`'s exit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Position {
    Arg(usize),
    Result,
    Exit,
}

/// What of an item a binding group records (l.186): an item other than a use,
/// whole, or the constraints a use gives at these positions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Part {
    Whole,
    At(BTreeSet<Position>),
}

impl Part {
    /// Whether the part holds the use's position.
    pub fn holds(&self, position: Position) -> bool {
        match self {
            Part::Whole => true,
            Part::At(positions) => positions.contains(&position),
        }
    }
}

/// The binding group of the function the run starts at (l.199). Its inputs
/// are the types the embedding declares, so it has no universal, and the
/// run's instance of it is its own solving continued.
pub const ENTRY: &str = "run";

/// A free fn or a method: the binding group it lies in, its parameters'
/// types, which are universal, and its result type, an unknown of its body
/// (l.210, l.219).
#[derive(Clone, Debug)]
pub struct Function {
    pub name: Name,
    pub group: Name,
    pub params: Vec<Name>,
    pub result: Name,
}

impl Program {
    pub fn function(&self, name: &Name) -> Option<&Function> {
        self.functions
            .iter()
            .find(|function| function.name == *name)
    }

    /// The groups enclosing `group`, innermost first, the entry last.
    pub fn enclosing(&self, group: &Name) -> Vec<Name> {
        let mut enclosing = Vec::new();
        let mut at = group.clone();
        while at.as_str() != ENTRY {
            at = self
                .groups
                .iter()
                .find(|(name, _)| *name == at)
                .and_then(|(_, parent)| parent.clone())
                .unwrap_or_else(|| Name::new(ENTRY));
            enclosing.push(at.clone());
        }
        enclosing
    }

    pub fn class(&self, name: &Name) -> &TextClass {
        self.classes
            .iter()
            .find(|class| class.name == *name)
            .unwrap_or_else(|| panic!("the program names the class {name}"))
    }

    /// The groups other than the entry, each before the groups whose uses
    /// make instances of it, and before the group it lies inside (l.213). A
    /// use's constraints are recorded where its text lies or in a group
    /// enclosing it (l.186), which lies after it, so the group its text lies
    /// in is solved after the groups it makes instances of.
    pub fn group_order(&self) -> Vec<Name> {
        let mut before: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
        for (group, item) in &self.items {
            if let Item::Use { callee, .. } = item
                && let Some(function) = self.function(callee)
                && function.group != *group
                && !self.enclosing(group).contains(&function.group)
            {
                before
                    .entry(group.clone())
                    .or_default()
                    .insert(function.group.clone());
            }
        }
        for (group, parent) in &self.groups {
            let parent = parent.clone().unwrap_or_else(|| Name::new(ENTRY));
            before.entry(parent).or_default().insert(group.clone());
        }
        let mut order: Vec<Name> = Vec::new();
        while order.len() < self.groups.len() {
            let next = self
                .groups
                .iter()
                .map(|(group, _)| group)
                .find(|group| {
                    !order.contains(group)
                        && before
                            .get(*group)
                            .into_iter()
                            .flatten()
                            .all(|earlier| order.contains(earlier))
                })
                .unwrap_or_else(|| panic!("the binding groups' instances form no cycle"));
            order.push(next.clone());
        }
        order
    }
}

/// A type the text gives, named, in the binding group its text lies in.
#[derive(Clone, Debug)]
pub struct TextClass {
    pub name: Name,
    pub group: Name,
    pub origin: Origin,
}

#[derive(Clone, Debug)]
pub enum Origin {
    /// A type not yet known (l.126): a binding's type, a block's value, a
    /// payload a constructor makes anew (l.316).
    Var,
    /// A literal's type (l.127).
    Lit(LitVar),
    /// A type the text or a declaration writes, or a value's constructor
    /// makes, its parts named by other text classes.
    Written(Term<Name>),
    /// A type the host gives (l.37, l.276), its parts named by other text
    /// classes.
    Host(Term<Name>),
    /// A type the embedding declares for the run's result, where a value
    /// leaves the program, so it may hold an `Error` (l.91, l.197), its parts
    /// named by other text classes.
    Outward(Term<Name>),
    /// The value of the lambda whose function is named: a function value
    /// whose Values names that lambda alone (l.181, l.186).
    Lambda(Name),
}

#[derive(Clone, Debug)]
pub enum Item {
    /// A value of the type `value` passes into the type `into` (l.219).
    Pass {
        value: Name,
        into: Name,
        place: Place,
    },
    /// An invariant place (l.229).
    Equation(Name, Name),
    /// A use that calls the declarations or the function `callee` names
    /// (l.192–195, l.320), its arguments' types and its result's; for `e?`,
    /// also the type of its exit's value, which passes into the innermost
    /// body's result as the text gives it (l.219, l.337).
    Use {
        name: Name,
        callee: Name,
        args: Vec<Name>,
        result: Name,
        exit: Option<Name>,
    },
    /// `&ₘ p`, `p`'s type `place`, giving a reference of the type `result`
    /// (l.296).
    Lend {
        name: Name,
        mutability: Mutability,
        place: Name,
        result: Name,
    },
    /// A projection of `operand`'s type, naming the part of the type `result`
    /// (l.144).
    Project {
        name: Name,
        projection: Projection,
        operand: Name,
        result: Name,
    },
    /// A use that calls the function value of the type `callee` (l.196).
    CallValue {
        name: Name,
        callee: Name,
        args: Vec<Name>,
        result: Name,
    },
    /// A binding at a pattern's position of the type `position`, reached from
    /// the scrutinee by the projections `path`, of the type `binding`
    /// (l.300–306).
    Bind {
        name: Name,
        path: Vec<Name>,
        position: Name,
        binding: Name,
    },
}

impl Item {
    /// A use's positions: its arguments', its result's and its exit's; none
    /// for another item.
    pub fn positions(&self) -> Vec<Position> {
        match self {
            Item::Use { args, exit, .. } => (0..args.len())
                .map(Position::Arg)
                .chain([Position::Result])
                .chain(exit.as_ref().map(|_| Position::Exit))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The class at a use's position.
    pub fn class_at(&self, position: Position) -> &Name {
        let Item::Use {
            args, result, exit, ..
        } = self
        else {
            panic!("a position is a use's");
        };
        match position {
            Position::Arg(k) => &args[k],
            Position::Result => result,
            Position::Exit => exit.as_ref().expect("the use is `e?`, which has an exit"),
        }
    }

    /// The classes the part of the item names.
    pub fn classes_in(&self, part: &Part) -> Vec<&Name> {
        match part {
            Part::Whole => self.classes(),
            Part::At(positions) => positions
                .iter()
                .map(|&position| self.class_at(position))
                .collect(),
        }
    }

    /// The classes the item names.
    pub fn classes(&self) -> Vec<&Name> {
        match self {
            Item::Pass { value, into, .. } => vec![value, into],
            Item::Equation(a, b) => vec![a, b],
            Item::Use {
                args, result, exit, ..
            } => args.iter().chain([result]).chain(exit).collect(),
            Item::CallValue {
                callee,
                args,
                result,
                ..
            } => [callee].into_iter().chain(args).chain([result]).collect(),
            Item::Lend { place, result, .. } => vec![place, result],
            Item::Project {
                operand, result, ..
            } => vec![operand, result],
            Item::Bind {
                position, binding, ..
            } => vec![position, binding],
        }
    }
}

/// A projection (l.146–152).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Projection {
    Deref,
    Field(Name),
    Component(usize),
    Elem,
    /// `?0 as V`; `enum_name` is the enum the text writes the variant of, `E`
    /// of `E::V`, and none for the language's `Some`, `None`, `Ok` and `Err`.
    Payload {
        enum_name: Option<Name>,
        variant: Name,
    },
}

/// How the place a value passes into is reached (l.219).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    /// Neither of the others: a block's value, an argument, a binding of the
    /// place's own group.
    Slot,
    /// A binding of an enclosing group reached by name, or its fields and
    /// parts.
    ByName,
    /// Through a reference, from a binding of an enclosing group.
    ThroughRef,
}

/// A program written item by item, as the examples' encodings write it.
pub struct ProgramBuilder {
    decls: Decls,
    groups: Vec<(Name, Option<Name>)>,
    functions: Vec<Function>,
    universals: Vec<Name>,
    classes: Vec<TextClass>,
    items: Vec<(Name, Item)>,
    /// The group the classes and items written next lie in.
    group: Name,
}

impl ProgramBuilder {
    pub fn new(decls: Decls) -> ProgramBuilder {
        ProgramBuilder {
            decls,
            groups: Vec::new(),
            functions: Vec::new(),
            universals: Vec::new(),
            classes: Vec::new(),
            items: Vec::new(),
            group: Name::new(ENTRY),
        }
    }

    /// Writes what follows in the binding group `name`, declared where first
    /// named.
    pub fn group(&mut self, name: &str) {
        self.group_in(name, None);
    }

    /// As `group`, the group lying inside `parent`, as a lambda lies inside
    /// its enclosing body (l.23).
    pub fn group_in(&mut self, name: &str, parent: Option<&str>) {
        let name = Name::new(name);
        let declared = self.groups.iter().any(|(group, _)| *group == name);
        if name.as_str() != ENTRY && !declared {
            self.groups.push((name.clone(), parent.map(Name::new)));
        }
        self.group = name;
    }

    /// Makes the class universal, as a receiver's `T̄` is (l.314).
    pub fn universal(&mut self, class: &Name) {
        self.universals.push(class.clone());
    }

    pub fn call_value(&mut self, name: &str, callee: &Name, args: &[&Name], result: &Name) {
        self.item(Item::CallValue {
            name: Name::new(name),
            callee: callee.clone(),
            args: args.iter().map(|&arg| arg.clone()).collect(),
            result: result.clone(),
        });
    }

    /// Declares a function of the current group, its parameters' types and
    /// its result type named by classes of it.
    pub fn function(&mut self, name: &str, params: &[&Name], result: &Name) {
        self.functions.push(Function {
            name: Name::new(name),
            group: self.group.clone(),
            params: params.iter().map(|&param| param.clone()).collect(),
            result: result.clone(),
        });
    }

    fn item(&mut self, item: Item) {
        self.items.push((self.group.clone(), item));
    }

    /// A text class of the origin. No type the text writes or a context
    /// gives is a function value, a lambda's value alone (l.184), and none
    /// holds an `Error`, which a value may hold only where it leaves the
    /// program (l.91); a function type in the type the embedding declares for
    /// the result, whose parameters a value enters the program by, is not
    /// encoded.
    pub fn class(&mut self, name: &str, origin: Origin) -> Name {
        match &origin {
            Origin::Written(term) | Origin::Host(term) => {
                assert!(
                    !matches!(term, Term::FunctionValue { .. }),
                    "{name}: the text writes, and a context gives, no function value (l.184)"
                );
                assert!(
                    !crate::join::is_error(term),
                    "{name}: the text writes, and a context gives, no `Error` (l.91)"
                );
            }
            Origin::Outward(term) => assert!(
                !matches!(term, Term::FunctionValue { .. } | Term::Function { .. }),
                "{name}: a function type or value in the embedding's result is not encoded"
            ),
            Origin::Var | Origin::Lit(_) | Origin::Lambda(_) => {}
        }
        let name = Name::new(name);
        self.classes.push(TextClass {
            name: name.clone(),
            group: self.group.clone(),
            origin,
        });
        name
    }

    pub fn var(&mut self, name: &str) -> Name {
        self.class(name, Origin::Var)
    }

    pub fn lit(&mut self, name: &str, lit: LitVar) -> Name {
        self.class(name, Origin::Lit(lit))
    }

    /// A class holding a type the text writes, which is no function value: a
    /// function value is a lambda's value alone (l.184).
    pub fn written(&mut self, name: &str, term: Term<Name>) -> Name {
        self.class(name, Origin::Written(term))
    }

    /// A class holding a type the embedding declares for the run's result,
    /// where a value leaves the program (l.91, l.197).
    pub fn outward(&mut self, name: &str, term: Term<Name>) -> Name {
        self.class(name, Origin::Outward(term))
    }

    /// A class holding a type a context gives, which is no function value
    /// (l.184).
    pub fn host(&mut self, name: &str, term: Term<Name>) -> Name {
        self.class(name, Origin::Host(term))
    }

    pub fn pass(&mut self, value: &Name, into: &Name) {
        self.pass_at(value, into, Place::Slot);
    }

    pub fn pass_at(&mut self, value: &Name, into: &Name, place: Place) {
        self.item(Item::Pass {
            value: value.clone(),
            into: into.clone(),
            place,
        });
    }

    pub fn equation(&mut self, a: &Name, b: &Name) {
        self.item(Item::Equation(a.clone(), b.clone()));
    }

    pub fn call(&mut self, name: &str, callee: &str, args: &[&Name], result: &Name) {
        self.item(Item::Use {
            name: Name::new(name),
            callee: Name::new(callee),
            args: args.iter().map(|&arg| arg.clone()).collect(),
            result: result.clone(),
            exit: None,
        });
    }

    /// `e?`, the type of `e` `operand`, giving `result`, its exit's value of
    /// the type `exit` (l.337).
    pub fn try_(&mut self, name: &str, operand: &Name, result: &Name, exit: &Name) {
        self.item(Item::Use {
            name: Name::new(name),
            callee: Name::new(crate::decl::TRY),
            args: vec![operand.clone()],
            result: result.clone(),
            exit: Some(exit.clone()),
        });
    }

    pub fn lend(&mut self, name: &str, mutability: Mutability, place: &Name, result: &Name) {
        self.item(Item::Lend {
            name: Name::new(name),
            mutability,
            place: place.clone(),
            result: result.clone(),
        });
    }

    pub fn project(&mut self, name: &str, projection: Projection, operand: &Name, result: &Name) {
        self.item(Item::Project {
            name: Name::new(name),
            projection,
            operand: operand.clone(),
            result: result.clone(),
        });
    }

    pub fn bind(&mut self, name: &str, path: &[&str], position: &Name, binding: &Name) {
        self.item(Item::Bind {
            name: Name::new(name),
            path: path.iter().map(|step| Name::new(step)).collect(),
            position: position.clone(),
            binding: binding.clone(),
        });
    }

    pub fn build(self) -> Program {
        Program {
            decls: self.decls,
            groups: self.groups,
            functions: self.functions,
            universals: self.universals,
            classes: self.classes,
            items: self.items,
        }
    }
}
