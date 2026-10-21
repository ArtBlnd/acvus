//! A program, as the constraints its text gives (l.219–231): the types its
//! text writes and makes, split into equivalence classes (l.120), and each
//! value passing into a type at a place, each in the binding group its text
//! lies in (l.203–215).

use std::collections::{BTreeMap, BTreeSet};

use crate::decl::Decls;
use crate::lit::LitVar;
use crate::ty::{Mutability, Name, Term};

#[derive(Clone, Debug)]
pub struct Program {
    pub decls: Decls,
    /// The binding groups other than the run's entry, which holds the text
    /// outside them (l.199), each with the group it lies inside, a lambda's
    /// being its enclosing body's (l.23).
    pub groups: Vec<(Name, Option<Name>)>,
    pub functions: Vec<Function>,
    /// The classes the text makes universal besides its parameters' types: a
    /// receiver's `T̄` (l.311).
    pub universals: Vec<Name>,
    pub classes: Vec<TextClass>,
    pub items: Vec<(Name, Item)>,
    pub readings: Readings,
    /// Which group solves each class and item, where `readings.captures`
    /// moves it from the one it lies in.
    cut: Cut,
}

/// The classes and items a lambda's captured environment holds, each with the
/// group that solves it, its enclosing one's (`design/17-environment.md`).
#[derive(Clone, Debug, Default)]
struct Cut {
    classes: BTreeMap<Name, Name>,
    items: BTreeMap<usize, Name>,
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

    /// The group that solves the class: the one it lies in, or the group
    /// enclosing the lambda whose environment holds it.
    pub fn owner(&self, class: &Name) -> &Name {
        self.cut
            .classes
            .get(class)
            .unwrap_or_else(|| &self.class(class).group)
    }

    /// The group that records the item at `i` in `items`.
    pub fn item_group(&self, i: usize) -> &Name {
        self.cut.items.get(&i).unwrap_or_else(|| &self.items[i].0)
    }

    /// Whether a lambda's environment holds the class.
    pub fn in_environment(&self, class: &Name) -> bool {
        self.cut.classes.contains_key(class)
    }

    /// The classes the lambdas' environments hold.
    pub fn environment(&self) -> BTreeSet<Name> {
        self.cut.classes.keys().cloned().collect()
    }

    /// Whether the class is universal: a function's parameter type that no
    /// environment holds, or a receiver's `T̄` (l.210, l.311).
    pub fn is_universal(&self, name: &Name) -> bool {
        let parameter = self
            .functions
            .iter()
            .any(|function| function.params.contains(name));
        self.universals.contains(name) || parameter && !self.in_environment(name)
    }

    /// Whether the class is an unknown of the group that solves it, as its
    /// text gives it (l.211): not written, and not universal.
    pub fn is_text_unknown(&self, name: &Name) -> bool {
        let class = self.class(name);
        matches!(class.origin, Origin::Var | Origin::Lit(_)) && !self.is_universal(name)
    }

    /// Each lambda's environment, moved to its enclosing group, the innermost
    /// lambdas first, so that what a lambda's environment records into an
    /// enclosing lambda is part of that lambda's text
    /// (`design/17-environment.md`, Definitions).
    fn cut_environments(&self) -> Cut {
        let mut cut = Cut::default();
        let mut lambdas: Vec<(&Name, &Name)> = self
            .groups
            .iter()
            .filter_map(|(group, parent)| Some((group, parent.as_ref()?)))
            .collect();
        lambdas.sort_by_key(|(group, _)| std::cmp::Reverse(self.enclosing(group).len()));
        for (lambda, parent) in lambdas {
            let owner = |cut: &Cut, class: &Name| -> Name {
                cut.classes
                    .get(class)
                    .cloned()
                    .unwrap_or_else(|| self.class(class).group.clone())
            };
            let items: Vec<usize> = (0..self.items.len())
                .filter(|&i| cut.items.get(&i).unwrap_or(&self.items[i].0) == lambda)
                .collect();
            let mut joined = Joined::default();
            for &i in &items {
                let named = self.joined_by(i, &items);
                for pair in named.windows(2) {
                    joined.join(pair[0], pair[1]);
                }
                for name in named {
                    joined.add(name);
                }
            }
            for class in &self.classes {
                if owner(&cut, &class.name) != *lambda {
                    continue;
                }
                joined.add(&class.name);
                if let Origin::Written(term) | Origin::Host(term) = &class.origin {
                    for (_, part) in term.parts() {
                        joined.join(&class.name, part);
                    }
                }
            }
            let captures: BTreeSet<Name> = joined
                .names()
                .filter(|name| owner(&cut, name) != *lambda)
                .map(|name| joined.root(name))
                .collect();
            let held: Vec<Name> = joined
                .names()
                .filter(|name| captures.contains(&joined.root(name)))
                .cloned()
                .collect();
            for name in &held {
                if owner(&cut, name) == *lambda {
                    cut.classes.insert(name.clone(), parent.clone());
                }
            }
            for i in items {
                if self
                    .joined_by(i, &[])
                    .into_iter()
                    .any(|name| captures.contains(&joined.root(name)))
                {
                    cut.items.insert(i, parent.clone());
                }
            }
        }
        cut
    }

    /// The classes the item at `i` joins: those it names, the parameter and
    /// result types of a function of an enclosing group it uses, and, for a
    /// binding, those its path's projections among `items` name.
    fn joined_by(&self, i: usize, items: &[usize]) -> Vec<&Name> {
        let item = &self.items[i].1;
        let mut named = item.classes();
        match item {
            Item::Use { callee, .. } => {
                if let Some(function) = self.function(callee)
                    && self.enclosing(&self.items[i].0).contains(&function.group)
                {
                    named.extend(function.params.iter().chain([&function.result]));
                }
            }
            Item::Bind { path, .. } => {
                for &j in items {
                    if let Item::Project { name, .. } = &self.items[j].1
                        && path.contains(name)
                    {
                        named.extend(self.items[j].1.classes());
                    }
                }
            }
            _ => {}
        }
        named
    }

    /// The groups other than the entry, each before the groups whose uses
    /// make instances of it, and before the group it lies inside (l.213).
    pub fn group_order(&self) -> Vec<Name> {
        let mut before: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
        for (i, (_, item)) in self.items.iter().enumerate() {
            let group = self.item_group(i);
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

/// The readings the model takes where the spec leaves one open (I1), each
/// stated in `design/holes.md`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Readings {
    pub visibility: Visibility,
    pub binding_path: BindingPath,
    pub deferral: Deferral,
    pub captures: Captures,
}

/// Which part of a lambda's text its binding group solves (l.203–215;
/// `holes.md`, H5 and O4; `design/17-environment.md`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Captures {
    /// All of it, its captures shared with the enclosing group and recorded
    /// into by each instance (l.215).
    #[default]
    AsStated,
    /// All but its environment, the classes joined to a capture and the items
    /// naming them, which the enclosing group solves once (functions §2).
    Environment,
}

/// When the relations of a node that a broken dependency pointed to are
/// recorded (l.262; `holes.md`, O2).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Deferral {
    /// Once the node that broke the dependency is decided.
    #[default]
    Recording,
    /// When their node is decided; the value the node that broke the
    /// dependency reads leaves them out.
    Reading,
}

/// Which members a kind gives (l.275; `holes.md`, T5).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Visibility {
    /// A field, and a variant of an enum the text names, whatever the class's
    /// other sources.
    #[default]
    ByMember,
    /// As `ByMember`, but none where a lower bound of the class is a type the
    /// host gives, which does not grow (patterns §2).
    BySources,
}

/// What decides whether a binding's path passes a reference (l.303;
/// `holes.md`, O1).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum BindingPath {
    /// The binding depends on what l.241 states alone, and a projection on its
    /// path not yet decided has passed no reference.
    #[default]
    AsStated,
    /// The binding depends on the projections on its path as well.
    AfterPath,
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
    /// payload a constructor makes anew (l.313).
    Var,
    /// A literal's type (l.127).
    Lit(LitVar),
    /// A type the text or a declaration writes, or a value's constructor
    /// makes, its parts named by other text classes.
    Written(Term<Name>),
    /// A type the host gives (l.37, l.275), its parts named by other text
    /// classes.
    Host(Term<Name>),
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
    /// (l.192–195, l.317), its arguments' types and its result's.
    Use {
        name: Name,
        callee: Name,
        args: Vec<Name>,
        result: Name,
    },
    /// `&ₘ p`, `p`'s type `place`, giving a reference of the type `result`
    /// (l.293).
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
    /// (l.297–303).
    Bind {
        name: Name,
        path: Vec<Name>,
        position: Name,
        binding: Name,
    },
}

impl Item {
    /// The classes the item names.
    pub fn classes(&self) -> Vec<&Name> {
        match self {
            Item::Pass { value, into, .. } => vec![value, into],
            Item::Equation(a, b) => vec![a, b],
            Item::Use { args, result, .. } => args.iter().chain([result]).collect(),
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
    readings: Readings,
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
            readings: Readings::default(),
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

    /// Makes the class universal, as a receiver's `T̄` is (l.311).
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

    pub fn readings(&mut self, readings: Readings) {
        self.readings = readings;
    }

    pub fn class(&mut self, name: &str, origin: Origin) -> Name {
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

    pub fn written(&mut self, name: &str, term: Term<Name>) -> Name {
        self.class(name, Origin::Written(term))
    }

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
        let mut program = Program {
            decls: self.decls,
            groups: self.groups,
            functions: self.functions,
            universals: self.universals,
            classes: self.classes,
            items: self.items,
            readings: self.readings,
            cut: Cut::default(),
        };
        if program.readings.captures == Captures::Environment {
            program.cut = program.cut_environments();
        }
        program
    }
}

/// Classes joined by the items that name them together (`design/17-environment.md`).
#[derive(Default)]
struct Joined {
    parent: BTreeMap<Name, Name>,
}

impl Joined {
    fn add(&mut self, name: &Name) {
        self.parent
            .entry(name.clone())
            .or_insert_with(|| name.clone());
    }

    fn root(&self, name: &Name) -> Name {
        let mut at = name.clone();
        while let Some(next) = self.parent.get(&at).filter(|next| **next != at) {
            at = next.clone();
        }
        at
    }

    fn join(&mut self, a: &Name, b: &Name) {
        self.add(a);
        self.add(b);
        let (a, b) = (self.root(a), self.root(b));
        if a != b {
            self.parent.insert(a, b);
        }
    }

    fn names(&self) -> impl Iterator<Item = &Name> {
        self.parent.keys()
    }
}
