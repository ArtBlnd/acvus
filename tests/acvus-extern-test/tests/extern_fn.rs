//! `#[extern_fn]`: each function declared by the attribute is called through
//! its dispatch on `WordRuntime`, and its declaration read back.

use acvus_extern::Uniform;
use acvus_extern::callback::{Callable, CallsBack};
use acvus_extern::context::{Awaited, Returning};
use acvus_extern::decl::{DeclTy, Declaration, Var, kind};
use acvus_extern::done::Done;
use acvus_extern::extern_fn;
use acvus_extern::mode::ByValue;
use acvus_extern::register::Declared;
use acvus_extern::runtime::{Crosses, Runtime};
use acvus_extern::Loaned;
use acvus_extern_test::{Cursor, Frame, Pair, Pass, WordRuntime};
use acvus_mir::decl::SegmentLabel;
use acvus_mir::ty::{Effect, FlowEnd, IntTy};
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
fn first_positive<'a, R: Runtime>(#[context] cx: &Returning<'_, R>, xs: &'a [Uniform<R, i64>]) -> Option<&'a Uniform<R, i64>> {
    xs.iter().find(|x| x.get(cx) > 0)
}

#[extern_fn(effect = pure)]
fn tagged(#[capture] tag: &String, text: &str) -> String {
    format!("{tag}: {text}")
}

#[extern_fn(effect = E)]
async fn apply<'f, R: CallsBack, E: Var<kind::Effect>>(
    #[context] cx: &Awaited<'_, R>,
    f: Uniform<R, Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64 {
    f.call_await(cx, (n,)).await + 1
}

#[extern_fn(effect = E)]
async fn twice<'f, R: CallsBack, E: Var<kind::Effect>>(
    #[context] cx: &Awaited<'_, R>,
    f: &Uniform<R, Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64 {
    let once = f.call_await(cx, (n,)).await;
    f.call_await(cx, (once,)).await
}

#[extern_fn(effect = idempotent)]
async fn measure(text: String) -> i64 {
    text.len() as i64
}

/// A variable holding `value`, made as a program makes one: a call's result.
fn produce<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call([], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

/// A variable's value, given to a call that takes it by value.
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
fn a_callback_and_the_context_are_passed() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let double = frame.function(Doubled);
    let n = produce(&mut frame, 20i64);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let applied = executor.block_on(frame.call_dispatch_awaited(&apply::new(), [Pass::Give(double), Pass::Give(n)]));

    assert_eq!(read::<i64>(&mut frame, applied), 41);
}

#[test]
fn a_lent_function_is_called_and_stays_the_callers() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let double = frame.function(Doubled);
    let n = produce(&mut frame, 5i64);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");

    let first = executor.block_on(frame.call_dispatch_awaited(&twice::new(), [Pass::Lend(double), Pass::Give(n)]));
    let m = produce(&mut frame, 1i64);
    let second = executor.block_on(frame.call_dispatch_awaited(&twice::new(), [Pass::Lend(double), Pass::Give(m)]));

    assert_eq!(read::<i64>(&mut frame, first), 20);
    assert_eq!(read::<i64>(&mut frame, second), 4);
}

/// A function value that doubles its argument, as a program's lambda would.
struct Doubled;

// SAFETY: this stands in for the dispatch of a program's lambda: `call` opens
// its one argument, an `i64`, by value.
#[allow(unsafe_code)]
unsafe impl acvus_extern::seal::ExternFns for Doubled {}

impl<R: Runtime> acvus_extern::register::ExternFn<R, 1, 0> for Doubled {
    fn call(&self, runtime: &R, args: [R::Value; 1], site: &acvus_extern::SiteShape<R, 0>) -> Done<R> {
        acvus_extern::Entry::new(runtime, args, site).open::<(ByValue<i64>,), ()>(|cx, (n,), ()| cx.finish(n * 2))
    }
}

#[test]
fn an_awaited_function_is_called_through_its_dispatch() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("awaited"));
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let length = executor.block_on(frame.call_dispatch_awaited(&measure::new(), [Pass::Give(text)]));

    assert_eq!(read::<i64>(&mut frame, length), 7);
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

    let (names, twice) = declaration_of(twice::new());
    assert_eq!(
        twice.params[0],
        (
            names.intern("f"),
            shared(DeclTy::Fn {
                params: vec![DeclTy::Int(IntTy::I64)],
                result: Box::new(DeclTy::Int(IntTy::I64)),
                effect: acvus_extern::decl::DeclEffect::Var(0),
            })
        )
    );
}

#[test]
fn a_borrowed_result_declares_its_references() {
    let (_, get) = declaration_of(get::new());
    assert_eq!(get.result, DeclTy::Option(Box::new(shared(DeclTy::Param(0)))));

    let (_, halves) = declaration_of(halves::new());
    assert_eq!(halves.result, DeclTy::Tuple(vec![shared(DeclTy::Str), shared(DeclTy::Str)]));

    let (names, first_positive) = declaration_of(first_positive::new());
    assert_eq!(first_positive.params, [(names.intern("xs"), shared(DeclTy::Slice(Box::new(DeclTy::Int(IntTy::I64)))))]);
    assert_eq!(first_positive.result, DeclTy::Option(Box::new(shared(DeclTy::Int(IntTy::I64)))));
}

/// The labels a declaration states of one end.
fn labels(declaration: &Declaration, end: FlowEnd) -> Vec<SegmentLabel> {
    let signature = declaration.facts.lifetimes.as_ref().expect("a declaration states its lifetimes");
    signature.ends.iter().find(|labelled| labelled.end == end).expect("every end is labelled").labels.to_vec()
}

#[extern_fn(effect = pure)]
fn pick<'m, 'k, T: Var<kind::Type>>(from: &'m [T], at: &'k u64) -> Option<&'m T> {
    from.get(*at as usize)
}

#[test]
fn a_borrowed_result_takes_the_loan_its_lifetime_names() {
    let (_, first) = declaration_of(first::new());
    let xs = labels(&first, FlowEnd::Param(0));
    assert_eq!(labels(&first, FlowEnd::Result), xs);
    assert_eq!(xs[1], SegmentLabel::TypeVar(0));

    let (_, pick) = declaration_of(pick::new());
    let from = labels(&pick, FlowEnd::Param(0));
    let at = labels(&pick, FlowEnd::Param(1));
    let result = labels(&pick, FlowEnd::Result);
    assert_eq!(result[0], from[0]);
    assert_ne!(result[0], at[0]);
}

#[test]
fn a_function_value_lent_is_a_reference_and_one_position() {
    let (_, twice) = declaration_of(twice::new());
    let f = labels(&twice, FlowEnd::Param(0));
    assert_eq!(f.len(), 2);
    assert_ne!(f[0], f[1]);
    let outlives = &twice.facts.lifetimes.as_ref().expect("stated").outlives;
    let [order] = &**outlives else { panic!("what the function value holds outlives the loan of it: {outlives:?}") };
    assert_eq!((SegmentLabel::Lifetime(order.longer), SegmentLabel::Lifetime(order.shorter)), (f[1], f[0]));
}

/// A cursor over the pair an argument lends: the result keeps the loan.
#[extern_fn(effect = pure)]
fn cursor<'a, R: Runtime>(over: Loaned<'a, Pair, R>) -> Cursor<'a, R> {
    Cursor { over, at: 0 }
}

/// The name of the pair a cursor keeps a loan on, read through the loan
/// within this call.
#[extern_fn(effect = pure)]
fn name_under<'a, R: Runtime>(#[context] cx: &Returning<'a, R>, c: &Cursor<'a, R>) -> String {
    c.over.open(cx).name.clone()
}

fn pair(name: &str) -> Pair {
    Pair { name: String::from(name), lead: acvus_extern_test::Number(1), items: Vec::new() }
}

#[test]
fn a_result_keeps_a_loan_an_argument_lent_and_a_later_call_reads_through_it() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let held = produce(&mut frame, pair("held"));
    let make = frame.function(cursor::new());
    let name = frame.function(name_under::new());

    let kept = frame.call_function(make, [Pass::Lend(held)]);
    let named = frame.call_function(name, [Pass::Lend(kept)]);

    assert_eq!(read::<String>(&mut frame, named), "held");
}

#[test]
fn a_kept_loan_is_the_results_and_the_arguments_one_label() {
    let (_, cursor) = declaration_of(cursor::new());
    let over = labels(&cursor, FlowEnd::Param(0));
    let result = labels(&cursor, FlowEnd::Result);
    assert!(matches!(&cursor.params[0].1, DeclTy::Ref { mutable: false, target } if matches!(**target, DeclTy::Extern { .. })));
    assert_eq!(result, over);
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
