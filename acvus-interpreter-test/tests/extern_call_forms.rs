//! Each call form reaches its declaration's Rust body and brings back what
//! that body returned (RFC-0059 rule 4).
//!
//! These run the language rather than calling a handler directly: the
//! handler is the operation's type parameter, so the form is chosen where
//! the operation is built, and only a prepared body exercises that choice.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::listing::ops_of_anywhere;
use acvus_interpreter_test::listing::script_listing_with_externs;
use acvus_interpreter_test::{Context, run_script_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn form0() -> i64 {
    7
}

#[extern_fn(effect = pure)]
fn form1(a: i64) -> i64 {
    a * 10
}

#[extern_fn(effect = pure)]
fn form2(a: i64, b: i64) -> i64 {
    a * 10 + b
}

#[extern_fn(effect = pure)]
fn form3(a: i64, b: i64, c: i64) -> i64 {
    (a * 10 + b) * 10 + c
}

#[extern_fn(effect = pure)]
fn form4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    ((a * 10 + b) * 10 + c) * 10 + d
}

#[extern_fn(effect = pure)]
fn form_window(a: i64, b: i64, c: i64, d: i64, e: i64) -> i64 {
    (((a * 10 + b) * 10 + c) * 10 + d) * 10 + e
}

/// A `&str` parameter is two of the runtime's values, so these four name the
/// register forms by word count and not by parameter count: two, three, four
/// values, and then past them.
#[extern_fn(effect = pure)]
fn pair2(s: &str) -> i64 {
    s.len() as i64
}

#[extern_fn(effect = pure)]
fn pair3(s: &str, a: i64) -> i64 {
    s.len() as i64 * 10 + a
}

#[extern_fn(effect = pure)]
fn pair4(s: &str, t: &str) -> i64 {
    s.len() as i64 * 10 + t.len() as i64
}

#[extern_fn(effect = pure)]
fn pair6(s: &str, t: &str, u: &str) -> i64 {
    (s.len() as i64 * 10 + t.len() as i64) * 10 + u.len() as i64
}

#[extern_fn(effect = pure)]
fn form_string(s: String) -> String {
    s.to_uppercase()
}

/// A result two values wide, at one value of arguments and then at two,
/// three, four and past them. Each body hands back a run of bytes its own
/// argument lent, which is the only thing a view may be (RFC-0047 rule 3).
#[extern_fn(effect = pure)]
fn view1(s: &String) -> &str {
    &s[..]
}

#[extern_fn(effect = pure)]
fn view2(s: &str) -> &str {
    s.trim()
}

#[extern_fn(effect = pure)]
fn view3(s: &str, n: i64) -> &str {
    &s[..n as usize]
}

#[extern_fn(effect = pure)]
fn view4<'a>(s: &'a str, t: &'a str) -> &'a str {
    match s.len() >= t.len() {
        true => s,
        false => t,
    }
}

#[extern_fn(effect = pure)]
fn view_window<'a>(s: &'a str, t: &'a str, u: &'a str) -> &'a str {
    match (s.len() >= t.len(), s.len() >= u.len()) {
        (true, true) => s,
        (false, _) => t,
        (_, false) => u,
    }
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [
            form0,
            form1,
            form2,
            form3,
            form4,
            form_window,
            form_string,
            pair2,
            pair3,
            pair4,
            pair6,
            view1,
            view2,
            view3,
            view4,
            view_window,
            addr_of,
        ],
    }
}

async fn answer(source: &str) -> i64 {
    let interner = Interner::new();
    run_script_with_externs(
        &interner,
        source,
        Context::default(),
        vec![registry()],
        Ty::I64,
    )
    .await
    .value
    .as_int()
}

#[tokio::test]
async fn every_register_form_returns_what_its_body_returned() {
    assert_eq!(answer("form0()").await, 7);
    assert_eq!(answer("form1(4)").await, 40);
    assert_eq!(answer("form2(4, 2)").await, 42);
    assert_eq!(answer("form3(4, 2, 1)").await, 421);
    assert_eq!(answer("form4(4, 2, 1, 3)").await, 4213);
}

#[tokio::test]
async fn the_window_form_returns_what_its_body_returned() {
    assert_eq!(answer("form_window(4, 2, 1, 3, 5)").await, 42135);
}

#[tokio::test]
async fn a_pair_parameter_returns_what_its_body_returned() {
    assert_eq!(answer("pair2(\"abcd\")").await, 4);
    assert_eq!(answer("pair3(\"abcd\", 7)").await, 47);
    assert_eq!(answer("pair4(\"abcd\", \"xy\")").await, 42);
    assert_eq!(answer("pair6(\"abcd\", \"xy\", \"z\")").await, 421);
}

/// The value alone does not say which form ran — the window form answers the
/// same thing. This is the form itself, read off the prepared operations.
#[test]
fn a_pair_parameter_costs_two_registers_and_not_a_window() {
    let cases = [
        ("pair2(\"abcd\")", "CallExtern2"),
        ("pair3(\"abcd\", 7)", "CallExtern3"),
        ("pair4(\"abcd\", \"xy\")", "CallExtern4"),
        ("pair6(\"abcd\", \"xy\", \"z\")", "CallWindow"),
        ("form4(4, 2, 1, 3)", "CallExtern4"),
        ("form_window(4, 2, 1, 3, 5)", "CallWindow"),
    ];
    for (source, form) in cases {
        let interner = Interner::new();
        let blocks = script_listing_with_externs(
            &interner,
            source,
            Context::default(),
            vec![
                registry(),
                acvus_ext::conversion_registry(),
                acvus_ext::iterator_registry(),
                acvus_ext::string_registry(),
            ],
            Ty::I64,
        );
        let ops = ops_of_anywhere(&blocks);
        let called: Vec<&String> = ops
            .iter()
            .filter(|op| op.contains("__extern_fn_"))
            .collect();
        assert_eq!(
            called.len(),
            1,
            "{source} prepares one extern call, and these are {called:?}"
        );
        assert!(
            called[0].starts_with(form),
            "{source} takes {form}, and the operation prepared for it is {}",
            called[0]
        );
    }
}

/// A view's own length, read back through `len`, is what the caller sees of
/// the pair the call wrote.
async fn view_length(source: &str) -> i64 {
    let interner = Interner::new();
    run_script_with_externs(
        &interner,
        source,
        Context::default(),
        vec![
            registry(),
            acvus_ext::conversion_registry(),
            acvus_ext::iterator_registry(),
            acvus_ext::string_registry(),
        ],
        Ty::U64,
    )
    .await
    .value
    .as_int()
}

#[tokio::test]
async fn a_view_result_returns_what_its_body_returned() {
    for (source, bytes, _) in view_cases() {
        assert_eq!(view_length(&source).await, bytes as i64, "{source}");
    }
}

/// The length alone does not say which form ran. This is the form itself,
/// read off the prepared operations.
/// One script per pair form, with the byte length of the view each one
/// takes and the operation `prepare` builds for it.
fn view_cases() -> [(String, usize, &'static str); 5] {
    let owned = "let s = \"  ab \".to_string();";
    [
        ("view1(&s)", 5, "CallPair1"),
        ("view2(&s)", 2, "CallPair2"),
        ("view3(&s, 3)", 3, "CallPair3"),
        ("view4(&s, \"xyzxyz\")", 6, "CallPair4"),
        ("view_window(&s, \"xyzxyz\", \"pq\")", 6, "CallPairWindow"),
    ]
    .map(|(call, bytes, form)| (format!("{owned} let v = {call}; len(&v)"), bytes, form))
}

#[test]
fn a_view_result_takes_the_pair_form_of_its_argument_width() {
    for (source, _, form) in view_cases() {
        let interner = Interner::new();
        let blocks = script_listing_with_externs(
            &interner,
            &source,
            Context::default(),
            vec![
                registry(),
                acvus_ext::conversion_registry(),
                acvus_ext::iterator_registry(),
                acvus_ext::string_registry(),
            ],
            Ty::U64,
        );
        let ops = ops_of_anywhere(&blocks);
        let called: Vec<&String> = ops
            .iter()
            .filter(|op| op.contains("__extern_fn_view"))
            .collect();
        assert_eq!(
            called.len(),
            1,
            "{source} prepares one view-returning call, and these are {called:?}"
        );
        assert!(
            called[0].starts_with(form),
            "{source} takes {form}, and the operation prepared for it is {}",
            called[0]
        );
    }
}

/// The first byte of the view, as an address the script can subtract.
#[extern_fn(effect = pure)]
fn addr_of(s: &str) -> i64 {
    s.as_ptr() as i64
}

/// The counting allocator in `view_allocates_nothing.rs` bounds what the
/// run costs; this is the copy itself, or its absence: the bytes `trim`
/// hands back are two bytes into the argument's own buffer.
#[tokio::test]
async fn a_view_names_the_bytes_of_its_argument() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let s = \"  ab \".to_string(); let v = trim(&s); addr_of(&v) - addr_of(&s)",
        Context::default(),
        vec![
            registry(),
            acvus_ext::conversion_registry(),
            acvus_ext::iterator_registry(),
            acvus_ext::string_registry(),
        ],
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 2);
}

#[tokio::test]
async fn the_slice_form_reaches_the_elements() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let a = vec([1, 2, 3]); let i = 0; let n = len(&a); let total = 0; \
         while i < n { total = total + a[i]; i = i + n / n; } total",
        Context::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 6);
}

#[tokio::test]
async fn a_large_value_crosses_the_one_argument_form() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "form_string(\"hello\".to_string())",
        Context::default(),
        vec![
            registry(),
            acvus_ext::conversion_registry(),
            acvus_ext::iterator_registry(),
            acvus_ext::string_registry(),
        ],
        Ty::String,
    )
    .await;
    assert!(
        ran.value.is_string(),
        "expected a String, got {:?}",
        ran.value
    );
    // SAFETY: the assertion above is the witness.
    assert_eq!(unsafe { ran.value.as_str() }, "HELLO");
}

/// The fusion rule matches this run, and each of its nodes holds a handler
/// of its own type (RFC-0044 rule 7).
#[tokio::test]
async fn a_fused_run_returns_what_its_last_call_returned() {
    assert_eq!(answer("form1(form2(form1(1), 2))").await, 1020);
}

/// `count` is `acvus-ext`'s `async fn` declaration: the caller suspends and
/// the driver resumes it with the future's value.
#[tokio::test]
async fn an_awaited_declaration_resumes_with_its_value() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let a = vec([1, 2, 3, 4]); a | into_iter | filter(|x| -> x > 1) | count",
        Context::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 3);
}

#[tokio::test]
async fn the_forms_compose_in_one_body() {
    assert_eq!(
        answer("form_window(form0() - 6, form1(0), form2(0, 1), form3(0, 0, 2), pair2(\"abcd\"))")
            .await,
        10124
    );
}
