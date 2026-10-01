//! `#[derive(Opaque)]`: an extern-defined type declared by the derive crosses
//! as itself, a generic one at its type variables' uniform values, and is
//! declared at its name, its arguments and its lifetimes.

use acvus_extern::decl::{DeclTy, Declaration, Var, kind};
use acvus_extern::extern_fn;
use acvus_extern::mode::ByValue;
use acvus_extern::register::Declared;
use acvus_extern::context::Returning;
use acvus_extern::runtime::{Crosses, Runtime};
use acvus_extern::{GenericParam, LayoutFact, Structural, Uniform};
use acvus_extern_test::{Cursor, Frame, Pair, Pass, Session, Stack, Word, WordRuntime};
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn stack<T: Var<kind::Type>>() -> Stack<T> {
    Stack { items: Vec::new() }
}

#[extern_fn(effect = pure)]
fn push<T: Var<kind::Type>>(s: &mut Stack<T>, x: T) {
    s.items.push(x)
}

#[extern_fn(effect = pure)]
fn pop<T: Var<kind::Type>>(s: &mut Stack<T>) -> Option<T> {
    s.items.pop()
}

#[extern_fn(effect = pure)]
fn peek<T: Var<kind::Type>>(s: &Stack<T>) -> Option<&T> {
    s.items.last()
}

#[extern_fn]
fn session<T: Var<kind::Type>, E: Var<kind::Effect>, I: Var<kind::Identity>>() -> Session<T, E, I> {
    Session { pending: Vec::new(), at: std::marker::PhantomData }
}

/// The parts of `text` between each `sep`, each held at its own type.
#[extern_fn(effect = pure)]
fn split<R: Runtime>(#[context] cx: &Returning<'_, R>, text: &str, sep: &str) -> Vec<Uniform<R, String>> {
    text.split(sep).map(|part| Uniform::new(cx, String::from(part))).collect()
}

/// The part at `at` of a sequence `split` gave.
#[extern_fn(effect = pure)]
fn part<R: Runtime>(#[context] cx: &Returning<'_, R>, parts: &Vec<Uniform<R, String>>, at: u64) -> String {
    parts[at as usize].open(cx).clone()
}

fn produce<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call([], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

fn read<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, variable: usize) -> T {
    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<T>,), ()>(|cx, (value,), ()| {
            read = Some(value);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

#[test]
fn a_generic_type_holds_its_variables_values() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let (new, push, pop, peek) =
        (frame.function(stack::new()), frame.function(push::new()), frame.function(pop::new()), frame.function(peek::new()));

    let s = frame.call_function(new, []);
    for n in [3i64, 5, 8] {
        let n = produce(&mut frame, n);
        frame.call_function(push, [Pass::Lend(s), Pass::Give(n)]);
    }
    let top = frame.call_function(peek, [Pass::Lend(s)]);
    let peeked = frame.option_referent::<Word>(top).map(Word::as_i64);
    let popped = frame.call_function(pop, [Pass::Lend(s)]);

    assert_eq!(peeked, Some(8));
    assert_eq!(read::<Option<i64>>(&mut frame, popped), Some(8));
}

#[test]
fn a_sequence_is_given_as_a_vec_of_uniform_values() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let (split, part) = (frame.function(split::new()), frame.function(part::new()));
    let text = produce(&mut frame, String::from("a,bc,d"));
    let sep = produce(&mut frame, String::from(","));

    let parts = frame.call_function(split, [Pass::Lend(text), Pass::Lend(sep)]);
    let at = produce(&mut frame, 1u64);
    let second = frame.call_function(part, [Pass::Lend(parts), Pass::Give(at)]);

    assert_eq!(read::<String>(&mut frame, second), "bc");
    assert_eq!(read::<Vec<Uniform<WordRuntime, String>>>(&mut frame, parts).len(), 3);

    let (names, split) = declaration_of(split::new());
    let vec = acvus_utils::QualifiedRef::qualified(names.namespace(&[names.intern("acvus_extern")]), names.intern("Vec"));
    assert_eq!(
        split.result,
        DeclTy::Extern { name: vec, args: vec![DeclTy::String], effect: None, identity: None, lifetimes: 0 }
    );
}

fn declaration_of<F: Declared<WordRuntime>>(f: F) -> (Interner, Declaration) {
    let names = Interner::new();
    let declaration = f.declaration(&names);
    (names, declaration)
}

#[test]
fn a_derived_type_is_declared_at_its_name_arguments_and_lifetimes() {
    let (names, stack) = declaration_of(stack::new());
    let DeclTy::Extern { name, args, effect, identity, lifetimes } = &stack.result else {
        panic!("a stack is extern-defined: {:?}", stack.result)
    };
    let module = names.namespace(&[names.intern("acvus_extern_test"), names.intern("types")]);
    assert_eq!(*name, acvus_utils::QualifiedRef::qualified(module, names.intern("Stack")));
    assert_eq!(*args, [DeclTy::Param(0)]);
    assert_eq!((effect, identity, *lifetimes), (&None, &None, 0));

    assert_eq!(<Cursor<'_, WordRuntime>>::FACTS.generics, [GenericParam::Lifetime]);
    assert_eq!(<Cursor<'_, WordRuntime>>::FACTS.layout, LayoutFact::CanonicalOf);
    assert_eq!(<Stack<Uniform<WordRuntime, i64>>>::FACTS.generics, [GenericParam::Type]);
    assert_eq!(<Stack<i64>>::FACTS.layout, LayoutFact::OwnCanonical);
    assert_eq!(<Pair>::FACTS.generics, []);

    let (_, session) = declaration_of(session::new());
    let DeclTy::Extern { args, effect, identity, .. } = &session.result else {
        panic!("a session is extern-defined: {:?}", session.result)
    };
    assert_eq!(*args, [DeclTy::Param(0)]);
    assert_eq!(*effect, Some(acvus_extern::decl::DeclEffect::Var(0)));
    assert_eq!(*identity, Some(0));
}
