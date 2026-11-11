//! A program as the text writes it: statements, each one operation of recording.
//!
//! Every value of these types is a program: none has to be checked before it is
//! recorded. A name is a class from where it is first written, so there is no
//! name that is not bound.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::Adjustment;
use crate::primitive::refusal::Cause;
use crate::primitive::ty::{Mutability, Scalar};

/// What is recorded, and what solving it is to come to. The two are apart: the
/// statements are the program, in the order they are recorded, and the
/// assertions say nothing to solving.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Program {
    pub statements: Vec<Statement>,
    pub assertions: Vec<Assertion>,
}

/// What solving a program is to come to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Assertion {
    /// `assert a: T;`: the class has the type, decided at every depth.
    Type { class: Path, ty: Type },
    /// `assert accepted;`: the program is refused by no cause. A program that
    /// asserts neither this nor a refusal is held to this.
    Accepted,
    /// `assert refused A, B;`: the program is refused by these causes and no
    /// other. `assert refused A, ..;`: by these, and by others or none besides.
    /// `assert refused ..;`: by some cause.
    Refused { causes: BTreeSet<Cause>, others: bool },
    /// `assert refused A names first, second;`: the connections the one refusal
    /// by the cause names are those of the statements labelled so.
    Names { cause: Cause, labels: BTreeSet<String> },
    /// `assert first takes Deref, Reborrow;`: the adjustments the constraint of
    /// the statement labelled so takes, in their order. A declared function
    /// called is written by its name. `assert first at r.x takes Deref;`: those
    /// the projection that names `r.x` in that statement takes.
    Takes {
        label: String,
        at: Option<Path>,
        adjustments: Vec<Adjustment>,
    },
}

/// One operation, with the label its items are found by, where it has one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Statement {
    pub label: Option<String>,
    pub op: Op,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Op {
    /// `upper <- lower;`: `lower` enters `upper`. `upper <- lower as _;`: what
    /// `lower` comes to by itself enters `upper`.
    Enter { upper: Path, lower: Type, alone: bool },
    /// `pass argument to given;`
    Pass { argument: Path, given: Path },
    /// `exit exits_with to error;`: the exit of a `?`.
    Exit { exits_with: Path, error: Path },
    /// `declare name = ty;`: `name` is a class a declaration writes.
    Declare { name: String, ty: Type },
    /// `row name<holes>: parameters -> result;`: a declared function.
    Row {
        name: String,
        holes: Vec<String>,
        parameters: Vec<Type>,
        result: Type,
    },
    /// `struct Name { fields };`: a struct, whose type `Name<T, U>` has one type
    /// parameter for each field, in their order.
    Struct { name: String, fields: Vec<String> },
    /// `upper <- fn [captures] (parameters) -> result { body };`: the function
    /// value of a closure of the lambda enters `upper`.
    Define { upper: Path, lambda: Lambda },
    /// `call callee(arguments) to result;`: `callee` is passed to the function
    /// type of those arguments and that result.
    Call {
        callee: Path,
        arguments: Vec<Path>,
        result: Path,
    },
    /// `module Name [fields] { body };`: a struct of that name and those fields,
    /// and the functions of it. The body is the body of a lambda that takes
    /// nothing: a field is a class of it, and each lambda a statement of the body
    /// assigns to a name is a function of the module, which captures the fields
    /// and the other functions. The module is what that lambda gives, the
    /// functions by their names, so each use of the module copies it whole.
    Module {
        name: String,
        fields: Vec<String>,
        body: Vec<Statement>,
    },
    /// `call Name::function(arguments) to result;`: the module is used, and that
    /// function of it called.
    CallOf {
        module: String,
        name: String,
        arguments: Vec<Path>,
        result: Path,
    },
    /// `call receiver::name(arguments) to result;`: the function of that name
    /// that takes the receiver is called.
    CallNamed {
        receiver: Path,
        name: String,
        arguments: Vec<Path>,
        result: Path,
    },
}

/// A lambda as the text writes it. In its body a name is a class of the lambda:
/// a parameter, the result, a class the lambda captures from where it is written,
/// or a class of its own from where the body first writes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lambda {
    /// The names of the classes captured, each a name where the lambda is
    /// written.
    pub captures: Vec<String>,
    pub parameters: Vec<Parameter>,
    pub result: String,
    /// A body declares nothing: what a declaration writes comes from outside
    /// the text.
    pub body: Vec<Statement>,
}

/// A parameter of a lambda: `x`, which takes what is passed as it comes, or
/// `x: T`, to which what is passed is passed as to a declared type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parameter {
    pub name: String,
    pub ty: Option<Type>,
}

/// A class: one a name stands for, or a part of one, named by each segment in
/// turn. A path left of `<-` is written into, and any other is read.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Path {
    pub root: String,
    pub segments: Vec<Segment>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Segment {
    /// `.x`
    Field(String),
    /// `.0`
    Component(usize),
    /// `.*`: what a reference refers to.
    Target,
    /// `.[]`
    Element,
    /// `.Some`, `.Ok`, `.Err`, `.Error::V`, `.E::V`
    Payload(Variant),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Variant {
    Some,
    Ok,
    Err,
    /// A variant of an `Error{..}`.
    Error(String),
    /// A variant of the enum of the first name.
    Enum(String, String),
}

/// A type as the text writes one. Where a class is to be given, a type gives the
/// class that has it: a path gives its class, a literal a class of its own, and
/// any other a class with that head, whose parts are the classes its parts give.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Type {
    Class(Path),
    /// `{integer}`: a literal with no suffix.
    Integer,
    /// `{float}`
    Float,
    /// `!`
    Never,
    Scalar(Scalar),
    /// `String`
    String,
    /// `str`
    Str,
    /// `&T`, `&mut T`
    Ref(Mutability, Box<Type>),
    /// `()`, `(T,)`, `(T, U)`
    Tuple(Vec<Type>),
    /// `[T; n]`
    Array(usize, Box<Type>),
    /// `[T]`
    Slice(Box<Type>),
    /// `Option<T>`
    Option(Box<Type>),
    /// `Result<T, E>`
    Result(Box<Type>, Box<Type>),
    /// `Error{V(T), W(U)}`
    Error(BTreeMap<String, Type>),
    /// `Error<E>`
    ErrorOf(Box<Type>),
    /// `{x: T, y: U}`
    Object(BTreeMap<String, Type>),
    /// `E{V(T), W(U)}`
    Enum(String, BTreeMap<String, Type>),
    /// `N`, `N<T, U>`: a type of a name, or, in a row, one of its holes.
    Named(String, Vec<Type>),
}
