//! The text of the model: a program written as the operations that record it.
//!
//! One statement is one operation over classes, and an operand is a class: a
//! name, or a part of one. So a program is read into the store with no judgment
//! of its own, and one is made by choosing operations and names.
//!
//! | Statement | What is recorded |
//! |---|---|
//! | `a <- b;` | `b` enters `a` |
//! | `a <- Option<b>;` | a class with that head, whose payload is `b`, enters `a` |
//! | `a <- {integer};` | a literal with no suffix enters `a` |
//! | `a <- b as _;` | what `b` comes to by itself enters `a` |
//! | `a.x <- b;`, `b <- a.x;` | the part `x` of `a` is written into, or read |
//! | `pass a to p;` | `a` is passed to `p` |
//! | `exit e to w;` | the exit of a `?`: `e` is what it exits with, `w` the error |
//! | `declare p = Option<t>;` | `p` is a class a declaration writes, and `t` a hole of it |
//! | `row as_slice<T>: &Vec<T> -> &[T];`, `row push<T>: &mut Vec<T>, T -> ();` | a declared function, of one parameter or of several |
//! | `f <- fn [a] (x) -> r { r <- x; };` | a closure of the lambda, over `a`, is named, and its function value enters `f` |
//! | `f <- fn (s: &Point<a, b>, x) -> r { .. };` | a parameter that writes a type: what is passed is passed to it as to a declared type |
//! | `row map<T, U>: Option<T>, Fn(T) -> U -> Option<U>;` | a function type: what a function value is passed to |
//! | `struct Point {x, y};` | a struct: `Point<T, U>` has the field `x` of the type `T` and `y` of `U` |
//! | `module Point [x, y] { new <- fn (a, b) -> r { .. }; };` | a struct and its functions: a lambda that takes nothing and gives them |
//! | `call Point::new(u, v) to p;` | the module is used, and that function of it called |
//! | `call f(u, v) to w;` | `f` is passed to the function type of those arguments and the result `w` |
//! | `call x::m(u) to w;` | the function of the name `m` that takes `x` as its receiver is called |
//! | `first: pass a to p;` | the statement's items are found by the label `first` |
//!
//! In the body of a lambda a name is a class of the lambda: one it captures, a
//! parameter, its result, or a class of its own from where the body first writes
//! it. The body is written into a template, by the statements a program is
//! written by, and a lambda written is called once with arguments nothing
//! enters.
//!
//! A word that begins with a small letter is a class, a field or a label, and one
//! that begins with a capital is a type or a variant. [`ast`] is the program,
//! [`parse`] reads one, a program is written out as it is read, and
//! [`record::record`] writes one into a store.
//!
//! | Assertion | What solving is to come to |
//! |---|---|
//! | `assert a: Option<u8>;` | the type of `a` |
//! | `assert accepted;` | the program is refused by no cause. A program that says neither this nor the next is held to this |
//! | `assert refused Mismatch, UndecidedClass;` | the causes it is refused by, and no other. `Mismatch, ..`: these at least. `..`: some cause |
//! | `assert refused Mismatch names first, second;` | the connections that refusal names |
//! | `assert first takes Deref, Reborrow;` | the adjustments the constraint of the statement labelled `first` takes |
//! | `assert first at r.x takes Deref;` | those the projection that names `r.x` in that statement takes |

pub mod ast;
mod print;
pub mod record;

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(
    #[allow(clippy::all, clippy::pedantic, unused)]
    grammar,
    "/text/grammar.rs"
);

/// Reads a program, or says where the text is none, by its line and column.
pub fn parse(text: &str) -> Result<ast::Program, String> {
    grammar::ProgramParser::new().parse(text).map_err(|error| {
        let at = |offset: usize| {
            let before = &text[..offset.min(text.len())];
            let line = before.matches('\n').count() + 1;
            let column = before.rfind('\n').map_or(before.len(), |start| before.len() - start - 1) + 1;
            format!("{line}:{column}")
        };
        error.map_location(at).to_string()
    })
}

#[cfg(test)]
mod tests;
