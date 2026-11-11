//! A program written out. What is written is read back as the same program.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use crate::primitive::adjustment::Adjustment;
use crate::primitive::ty::{Mutability, Scalar};

use super::ast::{Assertion, Brand, Lambda, Op, Path, Program, Segment, Statement, Type, Variant};

/// `items`, each written by `write`, with `, ` between them.
fn separated<I, T, F>(f: &mut Formatter<'_>, items: I, mut write: F) -> fmt::Result
where
    I: IntoIterator<Item = T>,
    F: FnMut(&mut Formatter<'_>, T) -> fmt::Result,
{
    for (index, item) in items.into_iter().enumerate() {
        if index > 0 {
            f.write_str(", ")?;
        }
        write(f, item)?;
    }
    Ok(())
}

fn payloads(f: &mut Formatter<'_>, variants: &BTreeMap<String, Type>) -> fmt::Result {
    f.write_str("{")?;
    separated(f, variants, |f, (variant, payload)| write!(f, "{variant}({payload})"))?;
    f.write_str("}")
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for statement in &self.statements {
            writeln!(f, "{statement}")?;
        }
        for assertion in &self.assertions {
            writeln!(f, "assert {assertion};")?;
        }
        Ok(())
    }
}

impl Display for Assertion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Assertion::Type { class, ty } => write!(f, "{class}: {ty}"),
            Assertion::Accepted => f.write_str("accepted"),
            Assertion::Refused { causes, others } => {
                f.write_str("refused ")?;
                separated(f, causes, |f, cause| write!(f, "{cause:?}"))?;
                match (*others, causes.is_empty()) {
                    (true, true) => f.write_str(".."),
                    (true, false) => f.write_str(", .."),
                    (false, _) => Ok(()),
                }
            }
            Assertion::Names { cause, labels } => {
                write!(f, "refused {cause:?} names")?;
                if !labels.is_empty() {
                    f.write_str(" ")?;
                }
                separated(f, labels, |f, label| f.write_str(label))
            }
            Assertion::Takes { label, at, adjustments } => {
                f.write_str(label)?;
                if let Some(at) = at {
                    write!(f, " at {at}")?;
                }
                f.write_str(" takes")?;
                if !adjustments.is_empty() {
                    f.write_str(" ")?;
                }
                separated(f, adjustments, |f, adjustment| match adjustment {
                    Adjustment::Deref => f.write_str("Deref"),
                    Adjustment::Lend(Mutability::Shared) => f.write_str("Lend"),
                    Adjustment::Lend(Mutability::Exclusive) => f.write_str("LendMut"),
                    Adjustment::Subtype(subtyping) => write!(f, "{subtyping:?}"),
                    Adjustment::Row(name) => f.write_str(&name.0),
                })
            }
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{label}: ")?;
        }
        write!(f, "{};", self.op)
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Op::Enter { upper, lower, alone } => {
                write!(f, "{upper} <- {lower}")?;
                if *alone {
                    f.write_str(" as _")?;
                }
                Ok(())
            }
            Op::Pass { argument, given } => write!(f, "pass {argument} to {given}"),
            Op::Exit { exits_with, error } => write!(f, "exit {exits_with} to {error}"),
            Op::Declare { name, ty } => write!(f, "declare {name} = {ty}"),
            Op::Row {
                name,
                holes,
                parameters,
                result,
            } => {
                write!(f, "row {name}")?;
                if !holes.is_empty() {
                    f.write_str("<")?;
                    separated(f, holes, |f, hole| f.write_str(hole))?;
                    f.write_str(">")?;
                }
                f.write_str(":")?;
                for (index, parameter) in parameters.iter().enumerate() {
                    write!(f, "{}{parameter}", if index == 0 { " " } else { ", " })?;
                }
                write!(f, " -> {result}")
            }
            Op::Module { name, fields, body } => {
                write!(f, "module {name} [")?;
                separated(f, fields, |f, field| f.write_str(field))?;
                f.write_str("] {")?;
                for statement in body {
                    write!(f, " {statement}")?;
                }
                f.write_str(" }")
            }
            Op::CallOf {
                module,
                name,
                arguments,
                result,
            } => {
                write!(f, "call {module}::{name}(")?;
                separated(f, arguments, |f, argument| write!(f, "{argument}"))?;
                write!(f, ") to {result}")
            }
            Op::Struct { name, fields } => {
                write!(f, "struct {name} {{")?;
                separated(f, fields, |f, field| f.write_str(field))?;
                f.write_str("}")
            }
            Op::Define { upper, lambda } => write!(f, "{upper} <- {lambda}"),
            Op::CallNamed {
                receiver,
                name,
                arguments,
                result,
            } => {
                write!(f, "call {receiver}::{name}(")?;
                separated(f, arguments, |f, argument| write!(f, "{argument}"))?;
                write!(f, ") to {result}")
            }
            Op::Call {
                callee,
                arguments,
                result,
            } => {
                write!(f, "call {callee}(")?;
                separated(f, arguments, |f, argument| write!(f, "{argument}"))?;
                write!(f, ") to {result}")
            }
        }
    }
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("fn ")?;
        if !self.captures.is_empty() {
            f.write_str("[")?;
            separated(f, &self.captures, |f, capture| f.write_str(capture))?;
            f.write_str("] ")?;
        }
        f.write_str("(")?;
        separated(f, &self.parameters, |f, parameter| match &parameter.ty {
            Some(ty) => write!(f, "{}: {ty}", parameter.name),
            None => f.write_str(&parameter.name),
        })?;
        write!(f, ") -> {} {{", self.result)?;
        for statement in &self.body {
            write!(f, " {statement}")?;
        }
        f.write_str(" }")
    }
}

impl Display for Path {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.root)?;
        for segment in &self.segments {
            write!(f, ".{segment}")?;
        }
        Ok(())
    }
}

impl Display for Segment {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Field(field) => f.write_str(field),
            Segment::Component(index) => write!(f, "{index}"),
            Segment::Target => f.write_str("*"),
            Segment::Element => f.write_str("[]"),
            Segment::Payload(Variant::Some) => f.write_str("Some"),
            Segment::Payload(Variant::Ok) => f.write_str("Ok"),
            Segment::Payload(Variant::Err) => f.write_str("Err"),
            Segment::Payload(Variant::Error(variant)) => write!(f, "Error::{variant}"),
            Segment::Payload(Variant::Enum(of, variant)) => write!(f, "{of}::{variant}"),
        }
    }
}

/// A scalar as the text writes it.
pub(super) fn scalar(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::I8 => "i8",
        Scalar::I16 => "i16",
        Scalar::I32 => "i32",
        Scalar::I64 => "i64",
        Scalar::U8 => "u8",
        Scalar::U16 => "u16",
        Scalar::U32 => "u32",
        Scalar::U64 => "u64",
        Scalar::Usize => "usize",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
        Scalar::Bool => "bool",
        Scalar::Char => "char",
    }
}

impl Display for Brand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Brand::Class(path) => write!(f, "{path}"),
            Brand::Hole(hole) => f.write_str(hole),
            Brand::Made(number) => write!(f, "#{number}"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Type::Class(path) => write!(f, "{path}"),
            Type::Integer => f.write_str("{integer}"),
            Type::Float => f.write_str("{float}"),
            Type::Never => f.write_str("!"),
            Type::Scalar(of) => f.write_str(scalar(*of)),
            Type::String => f.write_str("String"),
            Type::Str => f.write_str("str"),
            Type::Ref(Mutability::Shared, target) => write!(f, "&{target}"),
            Type::Ref(Mutability::Exclusive, target) => write!(f, "&mut {target}"),
            Type::Tuple(components) => {
                f.write_str("(")?;
                separated(f, components, |f, component| write!(f, "{component}"))?;
                if components.len() == 1 {
                    f.write_str(",")?;
                }
                f.write_str(")")
            }
            Type::Array(length, element) => write!(f, "[{element}; {length}]"),
            Type::Slice(element) => write!(f, "[{element}]"),
            Type::Option(payload) => write!(f, "Option<{payload}>"),
            Type::Result(ok, error) => write!(f, "Result<{ok}, {error}>"),
            Type::Error(variants) => {
                f.write_str("Error")?;
                payloads(f, variants)
            }
            Type::ErrorOf(error) => write!(f, "Error<{error}>"),
            Type::Object(fields) => {
                f.write_str("{")?;
                separated(f, fields, |f, (field, ty)| write!(f, "{field}: {ty}"))?;
                f.write_str("}")
            }
            Type::Enum(name, variants) => {
                f.write_str(name)?;
                payloads(f, variants)
            }
            Type::Branded(name, arguments, brand) => {
                write!(f, "{name}<")?;
                separated(f, arguments, |f, argument| write!(f, "{argument}"))?;
                write!(f, "; {brand}>")
            }
            Type::Fn(parameters, result) => {
                f.write_str("Fn(")?;
                separated(f, parameters, |f, parameter| write!(f, "{parameter}"))?;
                write!(f, ") -> {result}")
            }
            Type::Named(name, arguments) => {
                f.write_str(name)?;
                if !arguments.is_empty() {
                    f.write_str("<")?;
                    separated(f, arguments, |f, argument| write!(f, "{argument}"))?;
                    f.write_str(">")?;
                }
                Ok(())
            }
        }
    }
}
