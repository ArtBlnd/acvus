//! `#[extern_fn]`: each function declared by the attribute is called through
//! its dispatch on `WordRuntime`, and its declaration read back.

use acvus_extern::Uniform;
use acvus_extern::callback::{Callable, CallsBack};
use acvus_extern::executor::Executor;
use acvus_extern::decl::{Contexts, DeclTy, Declaration, Reissue, SplitCost, Var, kind};
use acvus_extern::done::Done;
use acvus_extern::extern_fn;
use acvus_extern::mode::ByValue;
use acvus_extern::register::{Declared, Refused, Registers};
use acvus_extern::runtime::{Crosses, Runtime};
use acvus_extern::Loaned;
use acvus_extern_test::{Cursor, Frame, Pair, Pass, Stack, WordRuntime};
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
fn first_positive<'a, R: Runtime>(#[executor] cx: &Executor<'_, R>, xs: &'a [Uniform<R, i64>]) -> Option<&'a Uniform<R, i64>> {
    xs.iter().find(|x| x.get(cx) > 0)
}

#[extern_fn(effect = pure)]
fn tagged(#[capture] tag: &String, text: &str) -> String {
    format!("{tag}: {text}")
}

#[extern_fn(effect = E)]
async fn apply<'f, R: CallsBack, E: Var<kind::Effect>>(
    #[executor] cx: &Executor<'_, R>,
    f: Uniform<R, Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64 {
    f.call_await(cx, (n,)).await + 1
}

#[extern_fn(effect = E, sync = twice_now)]
async fn twice<'f, R: CallsBack, E: Var<kind::Effect>>(
    #[executor] cx: &Executor<'_, R>,
    f: &Uniform<R, Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64 {
    let once = f.call_await(cx, (n,)).await;
    f.call_await(cx, (once,)).await
}

/// `twice`'s returning body: the callback called synchronously.
fn twice_now<'f, R: CallsBack, E: Var<kind::Effect>>(
    cx: &Executor<'_, R>,
    f: &Uniform<R, Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64 {
    let once = f.call(cx, (n,));
    f.call(cx, (once,))
}

#[extern_fn(effect = idempotent)]
async fn measure(text: String) -> i64 {
    text.len() as i64
}

/// A variable holding `value`, made as a program makes one: a call's result.
fn produce<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call([], |entry| entry.open(|cx, [], []| cx.finish(value)))
}

/// A variable's value, given to a call that takes it by value.
fn read<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, variable: usize) -> T {
    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let value = __given0.open::<ByValue<T>>(); 
            read = Some(value);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

#[test]
fn a_callback_and_the_executor_are_passed() {
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

#[test]
fn a_pair_runs_either_body_on_one_signature() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let double = frame.function(Doubled);
    let n = produce(&mut frame, 5i64);
    let returned = frame.call_dispatch(&twice::new(), [Pass::Lend(double), Pass::Give(n)]);
    let m = produce(&mut frame, 5i64);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let awaited = executor.block_on(frame.call_dispatch_awaited(&twice::new(), [Pass::Lend(double), Pass::Give(m)]));

    assert_eq!(read::<i64>(&mut frame, returned), 20);
    assert_eq!(read::<i64>(&mut frame, awaited), 20);
}

#[test]
fn a_pair_has_an_awaited_body_and_does_not_suspend() {
    let (_, twice) = declaration_of(twice::new());
    assert!(twice.awaits);
    assert!(!twice.effect.suspends());
}

#[test]
fn a_pair_is_registered_once() {
    let mut runtime = WordRuntime::default();
    runtime.register_pair(twice::new()).expect("a free path");
    assert!(matches!(runtime.register_pair(twice::new()), Err(Refused::PathTaken(_))));
}

/// A function value that doubles its argument, as a program's lambda would.
struct Doubled;

// SAFETY: this stands in for the dispatch of a program's lambda: `call` opens
// its one argument, an `i64`, by value.
#[allow(unsafe_code)]
unsafe impl acvus_extern::seal::ExternFns for Doubled {}

impl<R: Runtime> acvus_extern::register::ExternFn<R, 1, 0> for Doubled {
    fn call(&self, runtime: &R, args: [R::Value; 1], site: &acvus_extern::SiteShape<R, 0>) -> Done<R> {
        acvus_extern::Entry::new(runtime, args, site).open(|cx, [__given0], []| { let n = __given0.open::<ByValue<i64>>(); cx.finish(n * 2) })
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
    // Unstated, its reissue is opaque; a returning body does not suspend.
    assert_eq!(
        first.effect,
        Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable, false)
    );
    assert!(!first.awaits);
}

#[test]
fn captures_and_the_executor_are_no_parameters() {
    let (names, tagged) = declaration_of(tagged::with(String::new()));
    assert_eq!(tagged.params, [(names.intern("text"), shared(DeclTy::Str))]);

    let (names, apply) = declaration_of(apply::new());
    // An awaited body with no returning one suspends.
    assert_eq!(apply.effect, Effect::new(Reissue::Pure, true, Contexts::new(), Contexts::new(), SplitCost::Unavailable, true));
    assert!(apply.awaits);
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
fn name_under<'a, R: Runtime>(#[executor] cx: &Executor<'a, R>, c: &Cursor<'a, R>) -> String {
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

/// A result borrowing from two arguments, each part the loan its lifetime
/// names.
#[extern_fn(effect = pure)]
fn both<'a, 'b>(a: &'a str, b: &'b str) -> (&'a str, &'b str) {
    (a, b)
}

/// The name of the pair a cursor keeps a loan on, borrowed through that loan:
/// the result holds the cursor's loan, not the borrow of the cursor.
#[extern_fn(effect = pure)]
fn name_through<'a, R: Runtime>(#[executor] cx: &Executor<'a, R>, c: &Cursor<'a, R>) -> &'a str {
    c.over.open(cx).name.as_str()
}

#[test]
fn a_result_borrows_from_each_argument_its_lifetimes_name() {
    let (_, both) = declaration_of(both::new());
    let a = labels(&both, FlowEnd::Param(0));
    let b = labels(&both, FlowEnd::Param(1));
    assert_eq!(labels(&both, FlowEnd::Result), [a[0], b[0]]);

    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let left = produce(&mut frame, String::from("left"));
    let right = produce(&mut frame, String::from("right"));
    let pick = frame.function(both::new());

    let pair = frame.call_function(pick, [Pass::Lend(left), Pass::Lend(right)]);

    assert_eq!(frame.pair_referents::<str>(pair), (String::from("left"), String::from("right")));
}

#[test]
fn a_result_borrows_through_a_kept_loan() {
    let (_, through) = declaration_of(name_through::new());
    let cursor = labels(&through, FlowEnd::Param(0));
    assert_eq!(labels(&through, FlowEnd::Result), [cursor[1]]);
    assert_ne!(cursor[0], cursor[1]);

    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let held = produce(&mut frame, pair("held"));
    let make = frame.function(cursor::new());
    let name = frame.function(name_through::new());

    let kept = frame.call_function(make, [Pass::Lend(held)]);
    let named = frame.call_function(name, [Pass::Lend(kept)]);

    assert_eq!(read_text(&mut frame, named), "held");
}

/// The text a reference value names, read by a call that takes `&str`.
fn read_text(frame: &mut Frame<'_>, variable: usize) -> String {
    let mut read = None;
    frame.call([Pass::Same(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let text = __given0.open::<acvus_extern::mode::ByStr>(); 
            read = Some(String::from(&*text));
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// A count stated to the compiler: how it ends and how its result relates to
/// its arguments.
#[extern_fn(effect = pure, ends = total, relation(result <= len(text) + n, old(len(text)) = 2 * len(text) - 1))]
fn bounded(text: &str, n: u64) -> u64 {
    text.len() as u64 + n
}

#[test]
fn the_facts_written_are_stated_as_the_compilers_forms() {
    use acvus_extern::decl::{Atom, Before, Comparison, Ends, Relation, Subject, Term};

    let (_, bounded) = declaration_of(bounded::new());
    assert_eq!(bounded.facts.ends, Ends::Total);
    let atom = |atom| Box::new(Term::Atom(atom));
    assert_eq!(
        *bounded.facts.relations,
        [
            Relation {
                left: Term::Atom(Atom::Result),
                comparison: Comparison::Le,
                right: Term::Add(atom(Atom::Len(Subject::Arg(0))), atom(Atom::Arg(1))),
            },
            Relation {
                left: Term::Atom(Atom::Old(Before::Len(0))),
                comparison: Comparison::Eq,
                right: Term::Sub(Box::new(Term::Scale(2, atom(Atom::Len(Subject::Arg(0))))), Box::new(Term::Const(1))),
            },
        ]
    );
    let (_, add) = declaration_of(add::new());
    assert_eq!(add.facts.ends, Ends::Unstated);
    assert!(add.facts.relations.is_empty());
}

/// How many elements a stack holds, read whole.
#[extern_fn(effect = pure)]
fn depth<T: Var<kind::Type>>(#[touch(read)] s: &Stack<T>) -> u64 {
    s.items.len() as u64
}

/// The element at `at` replaced by `x`, the one it held given back.
#[extern_fn(effect = pure)]
fn put<T: Var<kind::Type>>(
    #[touch(store(x), index = at)]
    #[gives(index = at)]
    s: &mut Stack<T>,
    at: u64,
    x: T,
) -> T {
    std::mem::replace(&mut s.items[at as usize], x)
}

/// A stack emptied, its depth before given back through `depth`.
#[extern_fn(effect = pure)]
fn clear<T: Var<kind::Type>>(#[touch(store(empty))] #[gives(through = depth)] s: &mut Stack<T>) -> u64 {
    let depth = s.items.len() as u64;
    s.items.clear();
    depth
}

#[test]
fn the_facts_of_a_place_are_stated_on_its_argument() {
    use acvus_extern::decl::{Access, Element, Mode, Place, Stored, Touch};

    let (_, depth_decl) = declaration_of(depth::new());
    let whole = Place { param: 0, element: None };
    assert_eq!(*depth_decl.facts.access, [Access::Touches(Box::new([Touch { place: whole, mode: Mode::Read }]))]);

    let (_, put) = declaration_of(put::new());
    let at = Place { param: 0, element: Some(Element::Index(1)) };
    assert_eq!(
        *put.facts.access,
        [
            Access::Touches(Box::new([Touch { place: at, mode: Mode::Store(Stored::Value(2)) }])),
            Access::Gives { place: at, through: None },
        ]
    );

    let (names, clear) = declaration_of(clear::new());
    let depth_path = <depth as Declared<WordRuntime>>::path(&names);
    assert_eq!(
        *clear.facts.access,
        [
            Access::Touches(Box::new([Touch { place: whole, mode: Mode::Store(Stored::Emptied) }])),
            Access::Gives { place: whole, through: Some(depth_path) },
        ]
    );

    let (_, add) = declaration_of(add::new());
    assert!(add.facts.access.is_empty());
}

#[extern_fn(effect = pure)]
fn zero() -> i64 {
    0
}

/// Addition into a counter: a combine, commutative, `zero` its identity.
#[extern_fn(effect = pure, combine(identity = zero, commutative))]
fn add_into(acc: &mut i64, by: i64) {
    *acc += by
}

/// A counter raised by `by`: its update is `add_into`'s.
#[extern_fn(effect = pure)]
fn bump(#[update(combine = add_into)] counter: &mut i64, by: i64) {
    *counter += by
}

/// A counter lowered by `by`: it undoes `bump`'s update.
#[extern_fn(effect = pure)]
fn unbump(#[inverse_of(bump)] counter: &mut i64, by: i64) {
    *counter -= by
}

/// Concatenation: a combine and no more stated.
#[extern_fn(effect = pure, combine)]
fn concat(a: String, b: String) -> String {
    a + &b
}

#[test]
fn the_actions_written_are_stated_as_the_compilers_forms() {
    use acvus_extern::decl::{Action, Combine, Place, Update};

    let (names, add_into) = declaration_of(add_into::new());
    let zero = <zero as Declared<WordRuntime>>::path(&names);
    assert_eq!(*add_into.facts.action, [Action::Combine(Combine { identity: Some(zero), commutative: true, absorbing: None })]);

    let whole = Place { param: 0, element: None };
    let (names, bump) = declaration_of(bump::new());
    let add_into = <add_into as Declared<WordRuntime>>::path(&names);
    assert_eq!(*bump.facts.action, [Action::Update(Update { place: whole, combine: add_into })]);

    let (names, unbump) = declaration_of(unbump::new());
    let bump = <bump as Declared<WordRuntime>>::path(&names);
    assert_eq!(*unbump.facts.action, [Action::InverseOf { place: whole, other: bump }]);

    let (_, concat) = declaration_of(concat::new());
    assert_eq!(*concat.facts.action, [Action::Combine(Combine { identity: None, commutative: false, absorbing: None })]);
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
