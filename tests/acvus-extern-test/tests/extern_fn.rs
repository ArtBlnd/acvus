//! `#[extern_fn]`: each function declared by the attribute is registered at
//! `WordRuntime` and at `Adapter<WordRuntime>`, called through its dispatch
//! as a compiled call is, and its declaration read back.

use acvus_extern::Uniform;
use acvus_extern::adapter::Adapter;
use acvus_extern::callback::{Callback, CallsBack};
use acvus_extern::context::Awaited;
use acvus_extern::decl::{DeclTy, Declaration, Var, kind};
use acvus_extern::done::Done;
use acvus_extern::extern_fn;
use acvus_extern::mode::ByValue;
use acvus_extern::register::{Declared, Registers};
use acvus_extern::runtime::{Crosses, Runtime};
use acvus_extern_test::{Frame, Pass, Word, WordRuntime};
use acvus_mir::ty::{Effect, IntTy};
use acvus_utils::Interner;


#[extern_fn(effect = pure)]
fn add(a: u64, b: u64) -> u64 {
    a + b
}

#[extern_fn]
fn first<T: Var<kind::Type>>(xs: &[T]) -> &T {
    &xs[0]
}

#[extern_fn(effect = pure)]
fn get<T: Var<kind::Type>>(xs: &[T], at: u64) -> Option<&T> {
    xs.get(at as usize)
}

#[extern_fn(effect = pure)]
fn halves(text: &str) -> (&str, &str) {
    text.split_at(text.len() / 2)
}

#[extern_fn]
fn fail(message: &str) -> ! {
    panic!("{message}")
}

#[extern_fn(effect = pure)]
fn tagged(#[capture] tag: &String, text: &str) -> String {
    format!("{tag}: {text}")
}

#[extern_fn(effect = E)]
async fn apply<'c, R: CallsBack, E: Var<kind::Effect>>(
    #[context] cx: &Awaited<'c, R>,
    f: Callback<'c, '_, (i64,), i64, E, R>,
    n: i64,
) -> i64 {
    f.call_await(cx, (n,)).await + 1
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

fn calls_apply<R: Registers<Value = Word> + CallsBack>(mut runtime: R) {
    runtime.register_awaited(apply::new()).expect("the path is free");
    let path = Declared::<R>::path(&apply::new(), runtime.interner());

    let mut frame = Frame::new(&runtime);
    let double = frame.function(Doubled);
    let n = produce(&mut frame, 20i64);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let applied = executor.block_on(frame.call_registered_awaited(path, &[Pass::Function(double), Pass::Give(n)]));

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

    let module = names.namespace(&[names.intern("extern_fn")]);
    assert_eq!(add.path, acvus_utils::QualifiedRef::qualified(module, names.intern("add")));
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
    assert_eq!(apply.effect, Effect::pure());
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

#[test]
fn a_borrowed_result_declares_its_references() {
    let (_, get) = declaration_of(get::new());
    assert_eq!(get.result, DeclTy::Option(Box::new(shared(DeclTy::Param(0)))));

    let (_, halves) = declaration_of(halves::new());
    assert_eq!(halves.result, DeclTy::Tuple(vec![shared(DeclTy::Str), shared(DeclTy::Str)]));
}

#[test]
fn a_call_that_never_returns_declares_no_value() {
    let (_, fail) = declaration_of(fail::new());
    assert_eq!(fail.result, DeclTy::Never);
}

/// A borrowed result's parts need outlive only the borrow they are made
/// under, whatever lifetimes their own type carries: no type variable is
/// bounded `'static` (limitations §3). This compiles, and is all it checks.
#[allow(dead_code)]
fn borrows_at_any_lifetime<'s, 'x, R: Runtime>(
    part: &'s [Uniform<R, &'x ()>],
    sink: &acvus_extern::borrowed::Sink<'s, R>,
) -> acvus_extern::borrowed::Made<'s, R> {
    acvus_extern::borrowed::Borrows::made(part.first(), sink)
}
