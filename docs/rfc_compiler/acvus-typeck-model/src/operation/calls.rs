//! Calls, as programs: each test is a ruling of `TERMS.md` on function values
//! run, with the templates written by hand.

use std::collections::BTreeSet;

use super::language::Language;
use super::language::lambdas::{Defined, Lambda};
use super::testing::{U8, World, reorder};
use crate::primitive::class::Template;
use crate::primitive::constraint::Bound;
use crate::primitive::refusal::Cause;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Scalar, Shape};
use crate::primitive::{ClassId, ItemId, LambdaId, TemplateId};
use crate::store::classes::Classes;

const U16: Shape<ClassId> = Shape::Scalar(Scalar::U16);

/// A template being written: its classes, and the item each thing it writes
/// goes by.
struct Body<'a> {
    template: Classes<Template>,
    scopes: Scopes,
    world: &'a mut World,
}

impl<'a> Body<'a> {
    fn new(world: &'a mut World) -> Self {
        Self {
            template: Classes::default(),
            scopes: Scopes::new(),
            world,
        }
    }

    fn undecided(&mut self) -> TemplateId {
        self.template.undecided(self.scopes.root())
    }

    fn text(&mut self, shape: Shape<TemplateId>) -> TemplateId {
        let by = self.world.items.push("written in a body");
        self.template.decided(self.scopes.root(), shape, BTreeSet::from([by]))
    }

    fn enter(&mut self, lower: TemplateId, upper: TemplateId) -> ItemId {
        let by = self.world.items.push("bound in a body");
        self.template.constrain(Bound::Enter { lower, upper }.into(), by);
        by
    }
}

/// `|x| x`
fn identity(world: &mut World, law: &mut Language) -> LambdaId {
    let mut body = Body::new(world);
    let (x, result) = (body.undecided(), body.undecided());
    body.enter(x, result);
    law.lambdas.write(Lambda {
        template: body.template,
        parameters: vec![x],
        result,
        captures: Vec::new(),
        defines: Vec::new(),
    })
}

/// `|x| v`, `v` a value of the type `of`.
fn constant(world: &mut World, law: &mut Language, of: Shape<TemplateId>) -> LambdaId {
    let mut body = Body::new(world);
    let (x, result) = (body.undecided(), body.undecided());
    let value = body.text(of);
    body.enter(value, result);
    law.lambdas.write(Lambda {
        template: body.template,
        parameters: vec![x],
        result,
        captures: Vec::new(),
        defines: Vec::new(),
    })
}

/// `|x| a = Some(x)`, `a` captured.
fn remember(world: &mut World, law: &mut Language) -> LambdaId {
    let mut body = Body::new(world);
    let (a, x, result) = (body.undecided(), body.undecided(), body.undecided());
    let some = body.text(Shape::Option(x));
    body.enter(some, a);
    let unit = body.text(Shape::Unit);
    body.enter(unit, result);
    law.lambdas.write(Lambda {
        template: body.template,
        parameters: vec![x],
        result,
        captures: vec![a],
        defines: Vec::new(),
    })
}

/// `|x| { let z: u8 = 'c'; x }`
fn broken(world: &mut World, law: &mut Language) -> LambdaId {
    let mut body = Body::new(world);
    let (x, z, result) = (body.undecided(), body.undecided(), body.undecided());
    let (declared, written) = (body.text(Shape::Scalar(Scalar::U8)), body.text(Shape::Scalar(Scalar::Char)));
    body.enter(declared, z);
    body.enter(written, z);
    body.enter(x, result);
    law.lambdas.write(Lambda {
        template: body.template,
        parameters: vec![x],
        result,
        captures: Vec::new(),
        defines: Vec::new(),
    })
}

/// `|x| |y| Some(x)`: a lambda that defines one over its parameter.
fn defining(world: &mut World, law: &mut Language) -> LambdaId {
    let inner = {
        let mut body = Body::new(world);
        let (x, y, result) = (body.undecided(), body.undecided(), body.undecided());
        let some = body.text(Shape::Option(x));
        body.enter(some, result);
        law.lambdas.write(Lambda {
            template: body.template,
            parameters: vec![y],
            result,
            captures: vec![x],
            defines: Vec::new(),
        })
    };
    let by = world.items.push("definition in a body");
    let mut body = Body::new(world);
    let (x, value, result) = (body.undecided(), body.undecided(), body.undecided());
    body.enter(value, result);
    law.lambdas.write(Lambda {
        template: body.template,
        parameters: vec![x],
        result,
        captures: Vec::new(),
        defines: vec![Defined {
            lambda: inner,
            captures: vec![x],
            value,
            by,
        }],
    })
}

fn behind(world: &mut World, mut value: ClassId, lets: usize) -> ClassId {
    for _ in 0..lets {
        value = world.meets(value);
    }
    value
}

/// The one outcome `build` comes to, the function value it is given a count of
/// `let`s for being read behind each count from none to three, and its classes
/// asked in each order. Every order names the same in each refusal.
fn comes_to<F>(expected: &str, build: F)
where
    F: Fn(&mut World, &mut Language, usize) -> ClassId,
{
    for lets in 0..4 {
        let mut first = None;
        for order in 0..24 {
            let mut world = World::new();
            let mut law = Language::default();
            let observed = build(&mut world, &mut law, lets);
            reorder(&mut world.open, order);
            world.settle(&law);
            assert_eq!(world.outcome_of(observed), expected, "behind {lets} lets, in order {order}");
            let mut refusals: Vec<(Cause, BTreeSet<ItemId>, BTreeSet<ItemId>)> = world
                .outcome
                .refusals
                .iter()
                .map(|refusal| (refusal.cause, refusal.items.clone(), refusal.connections.clone()))
                .collect();
            refusals.sort();
            let first = first.get_or_insert_with(|| refusals.clone());
            assert_eq!(&refusals, first, "what the refusals name behind {lets} lets, in order {order}");
        }
    }
}

// A call copies the template, and each call copies it anew.

#[test]
fn a_call_gives_what_the_body_gives_of_its_argument() {
    comes_to("u8", |w, law, lets| {
        let identity = identity(w, law);
        let f = w.define(identity, Vec::new());
        let f = behind(w, f, lets);
        let argument = w.text(U8);
        w.call(f, &[argument])
    });
}

#[test]
fn two_calls_give_a_parameter_two_types() {
    for (observed, expected) in [(0, "u8"), (1, "u16")] {
        comes_to(expected, |w, law, lets| {
            let identity = identity(w, law);
            let f = w.define(identity, Vec::new());
            let f = behind(w, f, lets);
            let (first, second) = (w.text(U8), w.text(U16));
            [w.call(f, &[first]), w.call(f, &[second])][observed]
        });
    }
}

#[test]
fn a_call_with_another_count_of_arguments_is_refused() {
    comes_to("refused: Mismatch", |w, law, lets| {
        let identity = identity(w, law);
        let f = w.define(identity, Vec::new());
        let f = behind(w, f, lets);
        let (first, second) = (w.text(U8), w.text(U8));
        w.call(f, &[first, second])
    });
}

#[test]
fn what_is_no_function_is_not_called() {
    comes_to("refused: NoChange", |w, _, lets| {
        let f = w.text(U8);
        let f = behind(w, f, lets);
        let argument = w.text(U8);
        w.call(f, &[argument])
    });
}

// A capture is one class, however many copies there are.

/// `let a = None; let f = |x| a = Some(x);`, then `f` called with each of
/// `arguments`: the class of `a`.
fn remembering(w: &mut World, law: &mut Language, lets: usize, arguments: &[Shape<ClassId>]) -> ClassId {
    let nothing = w.undecided();
    let a = w.text(Shape::Option(nothing));
    let remember = remember(w, law);
    let f = w.define(remember, vec![a]);
    let f = behind(w, f, lets);
    for argument in arguments {
        let argument = w.text(argument.clone());
        w.call(f, &[argument]);
    }
    a
}

#[test]
fn a_parameter_nothing_is_passed_to_adds_nothing_to_a_capture() {
    comes_to("Option<!>", |w, law, lets| remembering(w, law, lets, &[]));
}

#[test]
fn an_argument_enters_a_capture_through_the_copy() {
    comes_to("Option<u8>", |w, law, lets| remembering(w, law, lets, &[U8]));
    comes_to("Option<u8>", |w, law, lets| remembering(w, law, lets, &[U8, U8]));
}

/// What enters below the top of a type is one type with what is there, so the
/// parameter of each copy is the payload of the capture, and the two arguments
/// meet there.
#[test]
fn the_arguments_of_two_calls_meet_in_a_capture() {
    comes_to("refused: Mismatch, UndecidedClass", |w, law, lets| remembering(w, law, lets, &[U8, U16]));
}

// What a body refuses apart from its parameters.

#[test]
fn a_body_is_refused_where_it_is_defined_and_once_however_often_it_is_called() {
    for calls in 0..3 {
        let mut world = World::new();
        let mut law = Language::default();
        let broken = broken(&mut world, &mut law);
        let f = world.define(broken, Vec::new());
        for _ in 0..calls {
            let argument = world.text(U8);
            world.call(f, &[argument]);
        }
        world.settle(&law);
        let causes: Vec<Cause> = world.outcome.refusals.iter().map(|refusal| refusal.cause).collect();
        // The class that takes no head is refused as undecided besides, as anywhere.
        assert_eq!(causes, vec![Cause::Mismatch, Cause::UndecidedClass], "called {calls} times");
    }
}

// Where two lambdas meet.

/// `if c { |x| x } else { |x| 0u8 }`, called with a value of the type `argument`.
fn either(w: &mut World, law: &mut Language, lets: usize, argument: Shape<ClassId>) -> ClassId {
    let (identity, constant) = (identity(w, law), constant(w, law, Shape::Scalar(Scalar::U8)));
    let (first, second) = (w.define(identity, Vec::new()), w.define(constant, Vec::new()));
    let f = w.meets(first);
    w.enter(second, f);
    let f = behind(w, f, lets);
    let argument = w.text(argument);
    w.call(f, &[argument])
}

#[test]
fn the_results_of_two_lambdas_that_met_are_one_type() {
    comes_to("u8", |w, law, lets| either(w, law, lets, U8));
    comes_to("refused: Mismatch, UndecidedClass", |w, law, lets| either(w, law, lets, U16));
}

// A lambda defined in a body is a closure of each copy.

/// `let f = |x| |y| Some(x); let p = f(0u8); let q = f(0u16);`
fn two_closures(w: &mut World, law: &mut Language, lets: usize) -> (ClassId, ClassId) {
    let defining = defining(w, law);
    let f = w.define(defining, Vec::new());
    let f = behind(w, f, lets);
    let (first, second) = (w.text(U8), w.text(U16));
    (w.call(f, &[first]), w.call(f, &[second]))
}

#[test]
fn each_copy_defines_a_closure_over_its_own_classes() {
    for (observed, expected) in [(0, "Option<u8>"), (1, "Option<u16>")] {
        comes_to(expected, |w, law, lets| {
            let (p, q) = two_closures(w, law, lets);
            let argument = w.text(U8);
            w.call([p, q][observed], &[argument])
        });
    }
}

#[test]
fn two_closures_of_one_lambda_that_met_are_both_called() {
    comes_to("refused: Mismatch", |w, law, lets| {
        let (p, q) = two_closures(w, law, lets);
        let g = w.meets(p);
        w.enter(q, g);
        let argument = w.text(U8);
        w.call(g, &[argument])
    });
}
