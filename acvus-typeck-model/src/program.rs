//! A program, as the constraints its text gives (l.226–238): the types its
//! text writes and makes, split into equivalence classes (l.127), and each
//! value passing into a type at a place.

use crate::decl::Decls;
use crate::lit::LitVar;
use crate::ty::{Mutability, Name, Term};

#[derive(Clone, Debug)]
pub struct Program {
    pub decls: Decls,
    pub classes: Vec<TextClass>,
    pub items: Vec<Item>,
    pub readings: Readings,
}

/// The readings the model takes where the spec leaves one open (I1), each
/// stated in `design/holes.md`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Readings {
    pub visibility: Visibility,
    pub binding_path: BindingPath,
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

/// A type the text gives, named.
#[derive(Clone, Debug)]
pub struct TextClass {
    pub name: Name,
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
    /// A use that calls the declarations `callee` names (l.331), its arguments'
    /// types and its result's.
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
    classes: Vec<TextClass>,
    items: Vec<Item>,
    readings: Readings,
}

impl ProgramBuilder {
    pub fn new(decls: Decls) -> ProgramBuilder {
        ProgramBuilder {
            decls,
            classes: Vec::new(),
            items: Vec::new(),
            readings: Readings::default(),
        }
    }

    pub fn readings(&mut self, readings: Readings) {
        self.readings = readings;
    }

    pub fn class(&mut self, name: &str, origin: Origin) -> Name {
        let name = Name::new(name);
        self.classes.push(TextClass {
            name: name.clone(),
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
        self.items.push(Item::Pass {
            value: value.clone(),
            into: into.clone(),
            place,
        });
    }

    pub fn equation(&mut self, a: &Name, b: &Name) {
        self.items.push(Item::Equation(a.clone(), b.clone()));
    }

    pub fn call(&mut self, name: &str, callee: &str, args: &[&Name], result: &Name) {
        self.items.push(Item::Use {
            name: Name::new(name),
            callee: Name::new(callee),
            args: args.iter().map(|&arg| arg.clone()).collect(),
            result: result.clone(),
        });
    }

    pub fn lend(&mut self, name: &str, mutability: Mutability, place: &Name, result: &Name) {
        self.items.push(Item::Lend {
            name: Name::new(name),
            mutability,
            place: place.clone(),
            result: result.clone(),
        });
    }

    pub fn project(&mut self, name: &str, projection: Projection, operand: &Name, result: &Name) {
        self.items.push(Item::Project {
            name: Name::new(name),
            projection,
            operand: operand.clone(),
            result: result.clone(),
        });
    }

    pub fn bind(&mut self, name: &str, path: &[&str], position: &Name, binding: &Name) {
        self.items.push(Item::Bind {
            name: Name::new(name),
            path: path.iter().map(|step| Name::new(step)).collect(),
            position: position.clone(),
            binding: binding.clone(),
        });
    }

    pub fn build(self) -> Program {
        Program {
            decls: self.decls,
            classes: self.classes,
            items: self.items,
            readings: self.readings,
        }
    }
}
