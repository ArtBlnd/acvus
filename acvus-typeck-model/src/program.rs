//! A program, as the constraints its text gives (l.226–238): the types its
//! text writes and makes, split into equivalence classes (l.127), and each
//! value passing into a type at a place, each in the binding group its text
//! lies in (l.210–222).

use std::collections::{BTreeMap, BTreeSet};

use crate::decl::Decls;
use crate::lit::LitVar;
use crate::ty::{Mutability, Name, Term};

#[derive(Clone, Debug)]
pub struct Program {
    pub decls: Decls,
    /// The binding groups other than the run's entry, which holds the text
    /// outside them (l.206).
    pub groups: Vec<Name>,
    pub functions: Vec<Function>,
    pub classes: Vec<TextClass>,
    pub items: Vec<(Name, Item)>,
    pub readings: Readings,
}

/// The binding group of the function the run starts at (l.206). Its inputs
/// are the types the embedding declares, so it has no universal, and the
/// run's instance of it is its own solving continued.
pub const ENTRY: &str = "run";

/// A free fn or a method: the binding group it lies in, its parameters'
/// types, which are universal, and its result type, an unknown of its body
/// (l.217, l.226).
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

    /// The groups other than the entry, each before the groups whose uses
    /// make instances of it (l.220).
    pub fn group_order(&self) -> Vec<Name> {
        let mut before: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
        for (group, item) in &self.items {
            if let Item::Use { callee, .. } = item
                && let Some(function) = self.function(callee)
                && function.group != *group
            {
                before
                    .entry(group.clone())
                    .or_default()
                    .insert(function.group.clone());
            }
        }
        let mut order: Vec<Name> = Vec::new();
        while order.len() < self.groups.len() {
            let next = self
                .groups
                .iter()
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
}

/// When the relations of a node that a broken dependency pointed to are
/// recorded (l.269; `holes.md`, H6).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Deferral {
    /// Once the node that broke the dependency is decided.
    #[default]
    Recording,
    /// When their node is decided; the value the node that broke the
    /// dependency reads leaves them out.
    Reading,
}

/// Which members a kind gives (l.282; `holes.md`, T5).
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

/// What decides whether a binding's path passes a reference (l.317;
/// `holes.md`, H5).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum BindingPath {
    /// The binding depends on what l.248 states alone, and a projection on its
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
    /// A type not yet known (l.133): a binding's type, a block's value, a
    /// payload a constructor makes anew (l.327).
    Var,
    /// A literal's type (l.134).
    Lit(LitVar),
    /// A type the text or a declaration writes, or a value's constructor
    /// makes, its parts named by other text classes.
    Written(Term<Name>),
    /// A type the host gives (l.37, l.282), its parts named by other text
    /// classes.
    Host(Term<Name>),
}

#[derive(Clone, Debug)]
pub enum Item {
    /// A value of the type `value` passes into the type `into` (l.226).
    Pass {
        value: Name,
        into: Name,
        place: Place,
    },
    /// An invariant place (l.236).
    Equation(Name, Name),
    /// A use that calls the declarations or the function `callee` names
    /// (l.199–202, l.331), its arguments' types and its result's.
    Use {
        name: Name,
        callee: Name,
        args: Vec<Name>,
        result: Name,
    },
    /// `&ₘ p`, `p`'s type `place`, giving a reference of the type `result`
    /// (l.307).
    Lend {
        name: Name,
        mutability: Mutability,
        place: Name,
        result: Name,
    },
    /// A projection of `operand`'s type, naming the part of the type `result`
    /// (l.151).
    Project {
        name: Name,
        projection: Projection,
        operand: Name,
        result: Name,
    },
    /// A binding at a pattern's position of the type `position`, reached from
    /// the scrutinee by the projections `path`, of the type `binding`
    /// (l.311–317).
    Bind {
        name: Name,
        path: Vec<Name>,
        position: Name,
        binding: Name,
    },
}

/// A projection (l.153–159).
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

/// How the place a value passes into is reached (l.226).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    /// By neither a binding's name nor a reference: a block's value, an
    /// argument.
    Slot,
    /// A binding reached by name, or its fields and parts.
    ByName,
    /// Through a reference.
    ThroughRef,
}

/// A program written item by item, as the examples' encodings write it.
pub struct ProgramBuilder {
    decls: Decls,
    groups: Vec<Name>,
    functions: Vec<Function>,
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
            classes: Vec::new(),
            items: Vec::new(),
            readings: Readings::default(),
            group: Name::new(ENTRY),
        }
    }

    /// Writes what follows in the binding group `name`, declared where first
    /// named.
    pub fn group(&mut self, name: &str) {
        let name = Name::new(name);
        if name.as_str() != ENTRY && !self.groups.contains(&name) {
            self.groups.push(name.clone());
        }
        self.group = name;
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
        Program {
            decls: self.decls,
            groups: self.groups,
            functions: self.functions,
            classes: self.classes,
            items: self.items,
            readings: self.readings,
        }
    }
}
