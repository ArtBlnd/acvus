//! `#[extern_fn]`: each function declared by the attribute is registered at
//! `WordRuntime` and at `Adapter<WordRuntime>`, called through its dispatch
//! as a compiled call is, and its declaration read back.

use acvus_extern::Uniform;
use acvus_extern::adapter::Adapter;
use acvus_extern::callback::{Callback, CallsBack};
use acvus_extern::context::Returning;
use acvus_extern::decl::{DeclTy, Declaration, Var, kind};
use acvus_extern::done::Done;
use acvus_extern::extern_fn;
use acvus_extern::mode::{ByRef, ByStr, ByValue};
use acvus_extern::register::{Declared, Registers};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern_test::{Frame, Number, Pair, Pass, Word, WordRuntime};
use acvus_mir::ty::{Effect, IntTy};
use acvus_utils::Interner;

use Mutability::{Exclusive, Shared};

#[extern_fn(effect = pure)]
fn add(a: u64, b: u64) -> u64 {
    a + b
}

#[extern_fn(effect = pure)]
fn name_of(pair: &Pair) -> &str {
    &pair.name
}

#[extern_fn]
fn first<R: Runtime, T: Var<kind::Type>>(xs: &[Uniform<R, T>]) -> &Uniform<R, T> {
    &xs[0]
}

#[extern_fn]
fn push(text: &mut String, more: &str) {
    text.push_str(more);
}

#[extern_fn(effect = pure)]
fn tagged(#[capture] tag: &String, text: &str) -> String {
    format!("{tag}: {text}")
}

#[extern_fn]
fn apply<'c, R: CallsBack, E: Var<kind::Effect>>(
    #[context] cx: &Returning<'c, R>,
    f: Callback<'c, '_, (i64,), i64, E, R>,
    n: i64,
) -> i64 {
    f.call(cx, (n,)) + 1
}

#[extern_fn(effect = idempotent)]
async fn measure(text: String) -> i64 {
    text.len() as i64
}

/// Runs `case` on a `WordRuntime` and on an adapter over one.
macro_rules! on_both {
    ($($name:ident: $case:ident;)+) => {
        mod on_word_runtime {
            $(
                #[test]
                fn $name() {
                    super::$case(super::WordRuntime::default());
                }
            )+
        }

        mod on_adapter {
            $(
                #[test]
                fn $name() {
                    super::$case(super::Adapter::new(super::WordRuntime::default()));
                }
            )+
        }
    };
}

on_both! {
    a_function_by_value_is_called_through_its_dispatch: calls_add;
    a_borrowed_result_names_a_part_of_its_lent_argument: calls_name_of;
    a_generic_function_is_dispatched_at_its_stand_in: calls_first;
    lent_arguments_are_written_and_read: calls_push;
    a_capture_is_lent_to_every_call: calls_tagged;
    a_callback_and_the_context_are_passed: calls_apply;
    an_awaited_function_is_called_through_its_dispatch: calls_measure;
}

/// A variable holding `value`, made as a program makes one: a call's result.
fn produce<R, T>(frame: &mut Frame<'_, R>, value: T) -> usize
where
    R: Runtime<Value = Word>,
    T: Crosses<R>,
{
    frame.call(&[], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

/// A variable's value, given to a call that takes it by value.
fn read<R, T>(frame: &mut Frame<'_, R>, variable: usize) -> T
where
    R: Runtime<Value = Word>,
    T: Crosses<R>,
{
    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<T>,), ()>(|cx, (value,), ()| {
            read = Some(value);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// The text a reference value names.
fn read_text<R: Runtime<Value = Word>>(frame: &mut Frame<'_, R>, reference: usize) -> String {
    let mut read = None;
    frame.call(&[Pass::Same(reference, Shared)], |entry| {
        entry.open::<(ByStr,), ()>(|cx, (text,), ()| {
            read = Some(String::from(&*text));
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

fn calls_add<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(add::new()).expect("the path is free");
    let path = Declared::<R>::path(&add::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let a = produce(&mut frame, 2u64);
    let b = produce(&mut frame, 3u64);
    let sum = frame.call_registered(path, &[Pass::Give(a), Pass::Give(b)]);

    assert_eq!(read::<_, u64>(&mut frame, sum), 5);
}

fn calls_name_of<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(name_of::new()).expect("the path is free");
    let path = Declared::<R>::path(&name_of::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let pair = produce(&mut frame, Pair { name: String::from("named"), lead: Number(1), items: Vec::new() });
    let name = frame.call_registered(path, &[Pass::Lend(pair, Shared)]);

    assert_eq!(read_text(&mut frame, name), "named");
}

fn calls_first<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(first::new()).expect("the path is free");
    let path = Declared::<R>::path(&first::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let array = frame.array(&[3i64, 5, 8]);
    let element = frame.call_registered(path, &[Pass::Lend(array, Shared)]);

    let mut read = None;
    frame.call(&[Pass::Same(element, Shared)], |entry| {
        entry.open::<(ByRef<Uniform<R, i64>>,), ()>(|cx, (element,), ()| {
            read = Some(element.get(&cx));
            cx.finish(())
        })
    });
    assert_eq!(read, Some(3));
}

fn calls_push<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(push::new()).expect("the path is free");
    let path = Declared::<R>::path(&push::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("lent"));
    let more = produce(&mut frame, String::from(" and written"));
    frame.call_registered(path, &[Pass::Lend(text, Exclusive), Pass::Lend(more, Shared)]);

    assert_eq!(read::<_, String>(&mut frame, text), "lent and written");
}

fn calls_tagged<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(tagged::with(String::from("tag"))).expect("the path is free");
    let path = Declared::<R>::path(&tagged::with(String::new()), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("text"));
    let first = frame.call_registered(path, &[Pass::Lend(text, Shared)]);
    let second = frame.call_registered(path, &[Pass::Lend(text, Shared)]);

    assert_eq!(read::<_, String>(&mut frame, first), "tag: text");
    assert_eq!(read::<_, String>(&mut frame, second), "tag: text");
}

fn calls_apply<R: Registers<Value = Word> + CallsBack>(mut runtime: R) {
    runtime.register(apply::new()).expect("the path is free");
    let path = Declared::<R>::path(&apply::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let double = frame.function(Doubled);
    let n = produce(&mut frame, 20i64);
    let applied = frame.call_registered(path, &[Pass::Function(double), Pass::Give(n)]);

    assert_eq!(read::<_, i64>(&mut frame, applied), 41);
}

/// A function value that doubles its argument, as a program's lambda would.
struct Doubled;

// SAFETY: this stands in for the dispatch of a program's lambda: `call` opens
// its one argument, an `i64`, by value.
#[allow(unsafe_code)]
unsafe impl acvus_extern::seal::ExternFns for Doubled {}

impl<R: Runtime> acvus_extern::register::ExternFn<R> for Doubled {
    fn call(&self, runtime: &R, args: &[R::Value], site: &acvus_extern::SiteShape<R>) -> Done<R> {
        acvus_extern::Entry::new(runtime, args, site).open::<(ByValue<i64>,), ()>(|cx, (n,), ()| cx.finish(n * 2))
    }
}

fn calls_measure<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register_awaited(measure::new()).expect("the path is free");
    let path = Declared::<R>::path(&measure::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("awaited"));
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let length = executor.block_on(frame.call_registered_awaited(path, &[Pass::Give(text)]));

    assert_eq!(read::<_, i64>(&mut frame, length), 7);
}

fn declaration_of<F: Declared<WordRuntime>>(f: F) -> (Interner, Declaration) {
    let names = Interner::new();
    let declaration = f.declaration(&names);
    (names, declaration)
}

fn shared(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: false, target: Box::new(target) }
}

#[test]
fn a_declaration_is_read_off_the_signature() {
    let (names, add) = declaration_of(add::new());

    assert_eq!(add.path, acvus_extern::decl::path(&names, "extern_fn", "add"));
    assert_eq!(
        add.params,
        [(names.intern("a"), DeclTy::Int(IntTy::U64)), (names.intern("b"), DeclTy::Int(IntTy::U64))]
    );
    assert_eq!(add.result, DeclTy::Int(IntTy::U64));
    assert_eq!(add.effect, Effect::pure());
    assert!(add.binders.types.is_empty());
}

#[test]
fn a_generic_declaration_is_at_its_parameters() {
    let (names, first) = declaration_of(first::new());

    assert_eq!(first.binders.types.len(), 1);
    assert_eq!(first.params, [(names.intern("xs"), shared(DeclTy::Slice(Box::new(DeclTy::Param(0)))))]);
    assert_eq!(first.result, shared(DeclTy::Param(0)));
    assert_eq!(first.effect, Effect::opaque());
}

#[test]
fn captures_and_the_context_are_no_parameters() {
    let (names, tagged) = declaration_of(tagged::with(String::new()));
    assert_eq!(tagged.params, [(names.intern("text"), shared(DeclTy::Str))]);

    let (names, apply) = declaration_of(apply::new());
    assert_eq!(apply.binders.effects, 1);
    assert_eq!(apply.runs, [0]);
    assert_eq!(
        apply.params,
        [
            (
                names.intern("f"),
                DeclTy::Fn {
                    params: vec![DeclTy::Int(IntTy::I64)],
                    result: Box::new(DeclTy::Int(IntTy::I64)),
                    effect: acvus_extern::decl::DeclEffect::Var(0),
                }
            ),
            (names.intern("n"), DeclTy::Int(IntTy::I64)),
        ]
    );
}
