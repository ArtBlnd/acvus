use super::parse;
use super::record::{Failed, run};

/// Every form the text has, written as it is written out.
const EVERY_FORM: &str = "\
a <- b;
a <- Option<b>;
a <- {integer};
a <- {float};
a <- !;
a <- b as _;
a <- Option<{integer}> as _;
a.x <- u8;
b <- a.x.0.*.[].Some.Ok.Err.Error::V.E::V;
o <- {x: p, y: &mut q};
o <- {};
t <- ();
t <- (a,);
t <- (a, [b; 3], [c], String, &str);
r <- Result<ok, Error{A(a), B(Error<i32>)}>;
e <- E{V(a), W(())};
n <- Vec<Vec<a>>;
n <- ParseError;
s <- (i8, i16, i32, i64, u8, u16, u32, u64, usize, f32, f64, bool, char);
pass a to p;
first: pass a.x to p;
exit e.Err to w;
declare p = Option<t>;
declare q = &mut {x: u8};
row as_slice<T>: &Vec<T> -> &[T];
row wrap<T, U>: (T, U) -> Pair<T, U>;
row unit: () -> u8;
row push<T>: &mut Vec<T>, T -> ();
row now: -> u64;
struct Point {x, y};
struct Unit {};
f <- fn () -> r { };
f <- fn (x: &Vec<a>, y, z: u8) -> r { };
f.x <- fn [a, b] (x, y) -> r { a <- Option<x>; first: r <- y; g <- fn [x] () -> s { s <- x; }; call g() to r; };
call f(a, b.x) to w.0;
call f() to w;
call a.x::push(b) to w;
call v::len() to n;
assert a: Option<u8>;
assert a.x.*: &mut {x: (), y: Error{V(E)}};
assert accepted;
assert refused ..;
assert refused Mismatch, ..;
assert refused Mismatch, NoChange, EmptyDomain, Occurs, NotHeld, UndecidedClass;
assert refused Mismatch names;
assert refused Mismatch names first, second;
assert first takes;
assert first takes Deref, Lend, LendMut, Reborrow, Never, DropFields, AddVariants, Unsize, as_slice;
assert first at r.x.* takes Deref;
";

#[test]
fn what_is_written_out_is_read_back_as_the_same_program() {
    let program = parse(EVERY_FORM).expect("the text is a program");
    assert_eq!(program.statements.len() + program.assertions.len(), EVERY_FORM.lines().count());
    assert_eq!(program.to_string(), EVERY_FORM);
    assert_eq!(parse(&program.to_string()), Ok(program));
}

#[test]
fn white_space_and_comments_are_not_of_the_program() {
    let spaced = "// a line\n  a<-b ;// the rest\n\n\tpass   a\n to p;";
    assert_eq!(parse(spaced), parse("a <- b; pass a to p;"));
}

#[test]
fn a_text_that_is_no_program_is_refused_with_where() {
    let error = |text: &str| parse(text).expect_err("the text is no program");
    assert!(error("a <- b;\na <- ;").contains("2:6"), "{}", error("a <- b;\na <- ;"));
    assert!(error("a <- {x: b, x: c};").contains("`x` is written twice"));
    assert!(error("A <- b;").contains("1:1"));
    assert!(error("assert refused Wrong;").contains("`Wrong` is no cause of a refusal"));
    assert!(error("assert first takes Wrong;").contains("`Wrong` is no adjustment"));
    assert!(error("a <- [b; 99999999999999999999999];").contains("too large"));
}

/// Solves `text` in every order the tests of the law take, and holds it to what
/// it asserts.
fn holds(text: &str) {
    assert_eq!(run(text, 24), Ok(()), "\n{text}");
}

#[test]
fn a_part_is_written_through_a_reference() {
    // `let o = {x: return}; let r = &mut o; r.x = 2u8;`
    holds(
        "p <- !;
         o <- {x: p};
         r <- &mut o;
         write: r.x <- u8;
         assert o: {x: u8};
         assert write at r.x takes Deref;",
    );
}

#[test]
fn a_part_is_read_through_each_reference_above_it() {
    holds(
        "o <- {x: u8};
         r <- &&&o;
         read: v <- r.x;
         assert v: u8;
         assert read at r.x takes Deref, Deref, Deref;",
    );
}

#[test]
fn arguments_that_do_not_join_in_a_hole_are_refused_naming_each_passing() {
    // `h(a, b)` with `h(T, T)`.
    holds(
        "a <- u8;
         b <- u16;
         first: pass a to t;
         second: pass b to t;
         assert refused Mismatch, UndecidedClass;
         assert refused Mismatch names first, second;",
    );
}

#[test]
fn a_literal_as_underscore_takes_no_head_from_what_it_is_passed_to() {
    // `let x = 0; f(x);` and `let x = 0 as _; f(x);`, with `f(u8)`.
    holds(
        "declare p = u8;
         x <- {integer};
         pass x to p;
         assert x: u8;",
    );
    holds(
        "declare p = u8;
         x <- {integer} as _;
         pass x to p;
         assert x: i32;
         assert refused NoChange;",
    );
}

#[test]
fn exits_unite_the_variants_of_what_they_exit_with() {
    // `let v = parse()?; read()?; Ok(v)`
    holds(
        "parse <- Result<u8, ParseError>;
         v <- parse.Ok;
         body <- Result<any, w>;
         exit parse.Err to w;
         read <- Result<u16, IoError>;
         exit read.Err to w;
         body <- Result<v, other>;
         assert body: Result<u8, Error{IoError(IoError), ParseError(ParseError)}>;",
    );
}

#[test]
fn a_value_goes_through_a_row_and_what_it_takes_is_recorded() {
    // `f(&mut v)` with `f(&[u8])` and `as_slice: &Vec<T> -> &[T]`.
    holds(
        "row as_slice<T>: &Vec<T> -> &[T];
         declare p = &[u8];
         v <- Vec<u8>;
         r <- &mut v;
         passed: pass r to p;
         assert passed takes Reborrow, as_slice;",
    );
}

#[test]
fn a_class_only_parts_are_named_on_takes_the_head_they_make() {
    holds(
        "a.x <- u8;
         b <- a;
         b.y <- {integer};
         assert a: {x: u8};
         assert b: {x: u8, y: i32};",
    );
    holds("x <- a.E::V; assert a: E{V(!)};");
}

#[test]
fn a_part_a_declared_type_has_no_place_for_is_refused_where_it_is_written() {
    holds("declare p = u8; b <- p.x; assert refused NotHeld;");
}

#[test]
fn a_program_that_does_not_come_to_what_it_asserts_says_what_it_came_to() {
    assert_eq!(
        run("a <- u8; assert a: u16;", 2),
        Err(Failed::Differs("in order 0, `assert a: u16;` does not hold: `a` is `u8`".to_owned()))
    );
    assert_eq!(
        run("a <- u8; a <- u16;", 2),
        Err(Failed::Differs("in order 0, `assert accepted;` does not hold: refused by {Mismatch, UndecidedClass}".to_owned()))
    );
    assert_eq!(run("a <- u8; a <- u16; assert refused Mismatch, ..;", 2), Ok(()));
    assert_eq!(run("a <- u8; a <- u16; assert refused ..;", 2), Ok(()));
    assert_eq!(
        run("a <- u8; assert refused ..;", 2),
        Err(Failed::Differs("in order 0, `assert refused ..;` does not hold: accepted".to_owned()))
    );
}

#[test]
fn what_cannot_be_recorded_is_said_before_anything_is_solved() {
    assert_eq!(run("declare p = u8; declare p = u16;", 1), Err(Failed::Unread("`p` is declared twice".to_owned())));
    assert_eq!(run("declare p = Option<p>;", 1), Err(Failed::Unread("`p` is declared as a type that names it".to_owned())));
    assert_eq!(run("row f: a -> u8;", 1), Err(Failed::Unread("a row names the class `a`".to_owned())));
    assert_eq!(run("row f: {integer} -> u8;", 1), Err(Failed::Unread("a row holds a literal".to_owned())));
}

#[test]
fn a_shared_reference_read_through_is_not_passed_as_an_exclusive_one() {
    // `put(&mut r)` with `r: &u8`, to `put(&mut u8)` and to `put(&u8)`.
    let program = |given: &str, expected: &str| format!("x <- u8; r <- &x; m <- &mut r; declare p = {given}; passed: pass m to p; {expected}");
    holds(&program("&mut u8", "assert refused NoChange;"));
    holds(&program("&u8", "assert passed takes Deref, Reborrow;"));
}

#[test]
fn a_part_read_through_a_reference_that_refers_to_itself_lies_nowhere() {
    holds("r <- &r; v <- r.x; assert refused NotHeld, ..;");
}

#[test]
fn a_literal_made_one_type_with_a_head_outside_its_domain_is_refused() {
    holds("o <- {f: {integer}}; o <- {f: String}; assert refused EmptyDomain, ..;");
}

#[test]
fn variants_are_passed_to_more_variants_and_to_no_fewer() {
    holds("declare p = E{A(u8), B(u16)}; a <- E{A(u8)}; first: pass a to p; assert first takes AddVariants;");
    holds("declare p = E{A(u8)}; a <- E{A(u8)}; first: pass a to p; assert first takes;");
    holds("declare p = E{A(u8)}; a <- E{A(u8), B(u16)}; pass a to p; assert refused NoChange;");
    holds("declare p = Error{A(u8), B(u16)}; a <- Error{A(u8)}; first: pass a to p; assert first takes AddVariants;");
    holds("declare p = Error{A(u8)}; a <- Error{A(u8), B(u16)}; pass a to p; assert refused NoChange;");
}

#[test]
fn a_row_is_taken_only_where_the_argument_is_what_it_writes_at_every_depth() {
    let rows = "row f: Wrap<Option<u8>> -> u8; declare p = u8;";
    holds(&format!("{rows} a <- Wrap<Option<u8>>; first: pass a to p; assert first takes f;"));
    holds(&format!("{rows} a <- Wrap<Option<u16>>; pass a to p; assert refused NoChange;"));
}

// Lambdas and calls.

#[test]
fn what_a_body_says_of_a_capture_reaches_it_whether_the_lambda_is_called_or_not() {
    holds("a <- Option<nothing>; f <- fn [a] () -> r { a <- Option<u8>; }; assert a: Option<u8>;");
    holds("a <- Option<nothing>; f <- fn [a] (x) -> r { a <- Option<x>; }; assert a: Option<!>;");
    holds("a <- Option<nothing>; f <- fn [a] (x) -> r { a <- Option<x>; }; u <- u8; call f(u) to w; assert a: Option<u8>;");
}

#[test]
fn each_call_copies_the_template_anew() {
    holds("id <- fn (x) -> r { r <- x; }; p <- u8; q <- u16; call id(p) to m; call id(q) to n; assert m: u8; assert n: u16;");
}

#[test]
fn a_call_is_refused_of_what_is_no_function_and_of_another_count_of_arguments() {
    holds("f <- u8; u <- u8; call f(u) to w; assert refused NoChange, ..;");
    holds("f <- fn (x) -> r { r <- x; }; u <- u8; call f(u, u) to w; assert refused Mismatch, ..;");
    holds("f <- fn (x) -> r { r <- x; }; call f() to w; assert refused Mismatch, ..;");
}

#[test]
fn a_body_is_refused_where_it_is_written_and_the_same_however_often_it_is_called() {
    let body = "f <- fn (x) -> r { z <- u8; z <- char; r <- x; };";
    holds(&format!("{body} assert refused Mismatch, UndecidedClass;"));
    holds(&format!("{body} u <- u8; call f(u) to v; call f(u) to w; assert refused Mismatch, UndecidedClass; assert v: u8;"));
}

#[test]
fn the_results_of_two_lambdas_that_met_are_one_type() {
    let met = "f <- fn (x) -> r { r <- x; }; f <- fn (x) -> r { r <- u8; };";
    holds(&format!("{met} q <- u8; call f(q) to m; assert m: u8;"));
    holds(&format!("{met} q <- u16; call f(q) to m; assert refused Mismatch, ..;"));
}

#[test]
fn each_copy_defines_a_closure_over_its_own_classes() {
    let made = "make <- fn (x) -> g { g <- fn [x] (y) -> r { r <- Option<x>; }; };
        a <- u8; b <- u16; c <- char;
        call make(a) to p; call make(b) to q;";
    holds(&format!("{made} call p(c) to m; call q(c) to n; assert m: Option<u8>; assert n: Option<u16>;"));
    holds(&format!("{made} both <- p; both <- q; call both(c) to m; assert refused Mismatch, ..;"));
}

#[test]
fn a_lambda_in_a_body_captures_what_the_body_captures() {
    holds(
        "a <- Option<nothing>;
         outer <- fn [a] () -> inner { inner <- fn [a] (x) -> r { a <- Option<x>; }; };
         call outer() to f; u <- u8; call f(u) to w;
         assert a: Option<u8>;",
    );
}

#[test]
fn a_lambda_that_names_one_class_twice_or_declares_is_no_program() {
    let unread = |text: &str| match run(text, 1) {
        Err(Failed::Unread(why)) => why,
        other => panic!("`{text}` is read: {other:?}"),
    };
    assert!(unread("f <- fn [a] (a) -> r { };").contains("`a` is named twice"));
    assert!(unread("f <- fn (x) -> x { };").contains("`x` is named twice"));
    assert!(unread("f <- fn () -> r { declare p = u8; };").contains("a declaration is written where a statement is"));
}

#[test]
fn each_capture_is_the_class_of_its_own_name() {
    holds("f <- fn [a, b] () -> r { a <- u8; b <- u16; }; assert a: u8; assert b: u16;");
}

// A call of a closure within a copy of it.

#[test]
fn a_lambda_calls_itself_passed_to_itself() {
    // `let f = |g, n| if .. { 0u8 } else { g(g, n) }; f(f, 1u8)`
    holds("f <- fn (g, n) -> r { r <- u8; call g(g, n) to s; r <- s; }; x <- u8; call f(f, x) to w; assert w: u8;");
    // What is never given by any branch never returns.
    holds("f <- fn (g) -> r { call g(g) to r; }; call f(f) to w; assert w: !;");
}

#[test]
fn a_lambda_calls_itself_captured() {
    holds("f <- fn [f] (n) -> r { r <- u8; call f(n) to s; r <- s; }; x <- u8; call f(x) to w; assert w: u8;");
}

#[test]
fn two_lambdas_call_one_another() {
    holds(
        "even <- fn [odd] (n) -> r { r <- bool; call odd(n) to s; r <- s; };
         odd <- fn [even] (n) -> r { call even(n) to r; };
         x <- u8; call even(x) to w; assert w: bool;",
    );
}

#[test]
fn the_arguments_of_a_call_and_of_the_call_within_it_enter_one_parameter() {
    let body = |inner: &str| format!("f <- fn (g, x) -> r {{ y <- {inner}; call g(g, y) to s; r <- x; }};");
    holds(&format!("{} x <- Option<nothing>; call f(f, x) to w; assert w: Option<u8>;", body("Option<u8>")));
    holds(&format!("{} x <- u8; call f(f, x) to w; assert refused Mismatch, UndecidedClass;", body("u16")));
    holds(&format!("{} x <- Option<u8>; call f(f, x) to w; assert refused Mismatch;", body("Option<x>")));
}

#[test]
fn the_arguments_of_two_calls_meet_in_a_capture() {
    holds(
        "a <- Option<nothing>; f <- fn [a] (x) -> r { a <- Option<x>; };
         u <- u8; v <- u16; call f(u) to p; call f(v) to q;
         assert refused Mismatch, UndecidedClass;",
    );
}

// Declared functions of several parameters.

#[test]
fn a_value_is_passed_through_a_row_of_one_parameter_and_of_no_other_count() {
    holds("row one: char -> u8; declare p = u8; a <- char; first: pass a to p; assert first takes one;");
    holds("row two: char, char -> u8; declare p = u8; a <- char; pass a to p; assert refused NoChange;");
    holds("row none: -> u8; declare p = u8; a <- char; pass a to p; assert refused NoChange;");
    holds("row two: char, bool -> i8; row one: i8 -> u8; row first: char -> i8; declare p = u8; a <- char; way: pass a to p; assert way takes first, one;");
}

// A call by name.

const NAMED: &str = "row len<T>: &Vec<T> -> usize; row push<T>: &mut Vec<T>, T -> ();
    row into_inner<T>: Vec<T> -> T; row len: &String -> usize;";

#[test]
fn a_receiver_goes_to_the_first_parameter_as_an_argument_is_passed() {
    let called = |program: &str| holds(&format!("{NAMED} v <- Vec<u8>; {program}"));
    called("r <- &v; c: call r::len() to n; assert n: usize; assert c takes len;");
    called("r <- &mut v; c: call r::len() to n; assert c takes Reborrow, len;");
    called("r <- &v; rr <- &r; c: call rr::len() to n; assert c takes Deref, len;");
    called("c: call v::into_inner() to n; assert n: u8; assert c takes into_inner;");
    called("r <- &v; x <- u8; call r::push(x) to n; assert refused NoChange;");
    called("r <- &v; call r::into_inner() to n; assert refused NoChange;");
}

#[test]
fn a_receiver_that_is_no_reference_is_lent_where_the_parameter_is_one() {
    let called = |program: &str| holds(&format!("{NAMED} v <- Vec<u8>; {program}"));
    called("c: call v::len() to n; assert n: usize; assert c takes Lend, len;");
    called("x <- u8; c: call v::push(x) to n; assert n: (); assert c takes LendMut, push;");
}

#[test]
fn the_function_of_a_name_is_the_one_the_receiver_goes_to() {
    holds(&format!("{NAMED} s <- String; c: call s::len() to n; assert n: usize; assert c takes Lend, len;"));
    holds(&format!("{NAMED} v <- Vec<char>; call v::into_inner() to n; assert n: char;"));
    holds(&format!("{NAMED} v <- Vec<u8>; call v::nope() to n; assert refused NoChange;"));
    holds(&format!("{NAMED} v <- u8; call v::len() to n; assert refused NoChange;"));
    holds("row twice<T>: T -> u8; row twice: char -> u16; v <- char; call v::twice() to n; assert refused NoChange;");
    holds("row twice<T>: T -> u8; row twice: char -> u16; v <- bool; call v::twice() to n; assert n: u8;");
}

#[test]
fn the_arguments_after_the_receiver_are_passed_to_their_parameters() {
    holds(&format!("{NAMED} v <- Vec<u8>; call v::push() to n; assert refused Mismatch, ..;"));
    holds(&format!("{NAMED} v <- Vec<u8>; x <- char; call v::push(x) to n; assert refused ..;"));
    holds(&format!("{NAMED} v <- Vec<u8>; y <- u8; x <- &mut y; c: call v::push(x) to n; assert refused ..;"));
    holds("row put: &mut Vec<u8>, &u8 -> (); v <- Vec<u8>; y <- u8; x <- &mut y; call v::put(x) to n; assert n: ();");
}

#[test]
fn a_call_by_name_in_a_body_is_made_in_each_copy() {
    holds(&format!("{NAMED} f <- fn (x) -> r {{ call x::into_inner() to r; }}; a <- Vec<u8>; b <- Vec<char>; call f(a) to m; call f(b) to n; assert m: u8; assert n: char;"));
}

#[test]
fn a_function_of_a_name_is_told_from_another_below_the_head_of_the_receiver() {
    let rows = "row first: &Vec<u8> -> u8; row first: &Vec<char> -> char;";
    holds(&format!("{rows} v <- Vec<char>; call v::first() to n; assert n: char;"));
    holds(&format!("{rows} v <- Vec<u8>; call v::first() to n; assert n: u8;"));
    holds(&format!("{rows} v <- Vec<bool>; call v::first() to n; assert refused NoChange;"));
}

#[test]
fn an_argument_is_passed_to_a_parameter_that_writes_a_type_and_does_not_enter_it() {
    // `&&u8` is passed to `&u8`, read through; it does not enter it.
    holds("row put: &mut Vec<u8>, &u8 -> (); v <- Vec<u8>; y <- u8; r <- &y; x <- &r; c: call v::put(x) to n; assert n: ();");
}

// A parameter that writes a type.

#[test]
fn what_is_passed_to_a_parameter_that_writes_a_type_is_passed_as_to_a_declared_type() {
    let f = "f <- fn (x: &u8) -> r { r <- x; };";
    holds(&format!("{f} v <- u8; a <- &mut v; call f(a) to m; assert m: &u8;"));
    // Read through, which what enters is not.
    holds(&format!("{f} v <- u8; a <- &v; b <- &a; call f(b) to m; assert m: &u8;"));
    holds(&format!("{f} a <- char; call f(a) to m; assert refused NoChange, ..;"));
    // Lent to no parameter but a receiver's.
    holds(&format!("{f} a <- u8; call f(a) to m; assert refused NoChange, ..;"));
    holds("f <- fn (x: &mut u8) -> r { r <- u8; }; v <- u8; a <- &v; call f(a) to m; assert refused NoChange, ..;");
    holds("f <- fn (x: u8) -> r { r <- x; }; a <- {integer}; call f(a) to m; assert m: u8; assert a: u8;");
}

#[test]
fn the_type_a_parameter_writes_names_classes_of_the_lambda() {
    holds("f <- fn [t] (x: Option<t>) -> r { r <- x; }; t <- u8; a <- Option<u8>; call f(a) to m; assert m: Option<u8>;");
    let f = "f <- fn (y, x: Option<y>) -> r { r <- x; };";
    holds(&format!("{f} b <- u8; a <- Option<u8>; call f(b, a) to m; assert m: Option<u8>;"));
    holds(&format!("{f} b <- char; a <- Option<u8>; call f(b, a) to m; assert refused ..;"));
}

#[test]
fn a_parameter_that_writes_a_type_is_read_in_the_body_as_a_value_of_it() {
    holds("f <- fn (s: &{x: u8}) -> r { r <- s.x; }; o <- {x: u8}; a <- &o; call f(a) to m; assert m: u8;");
}

// The fields of a struct.

#[test]
fn a_field_of_a_struct_is_its_type_parameter_at_the_place_the_struct_declares() {
    let point = |program: &str| holds(&format!("struct Point {{x, y}}; {program}"));
    point("p <- Point<u8, char>; a <- p.x; b <- p.y; assert a: u8; assert b: char;");
    point("v <- p.x; p <- Point<u8, char>; assert v: u8;");
    point("p <- Point<u8, char>; r <- &p; rr <- &r; c: b <- rr.y; assert b: char; assert c at rr.y takes Deref, Deref;");
    point("p <- Point<t, char>; p.x <- u8; assert p: Point<u8, char>;");
    point("p <- Point<u8, char>; p.x <- bool; assert refused ..;");
    point("p <- Point<u8, char>; a <- p.z; assert refused NotHeld;");
    point("p <- Point<u8, char>; a <- p.0; assert refused NotHeld;");
    point("p <- Other<u8>; a <- p.x; assert refused NotHeld;");
    point("f <- fn (s: &Point<a, b>) -> r { r <- s.x; }; p <- Point<u8, char>; q <- &p; call f(q) to m; assert m: u8;");
}

#[test]
fn a_struct_declared_twice_or_with_one_field_twice_is_no_program() {
    let unread = |text: &str| match run(text, 1) {
        Err(Failed::Unread(why)) => why,
        other => panic!("`{text}` is read: {other:?}"),
    };
    assert!(unread("struct P {x}; struct P {y};").contains("the struct `P` is declared twice"));
    assert!(unread("struct P {x, x};").contains("the struct `P` has the field `x` twice"));
    assert!(unread("f <- fn () -> r { struct P {x}; };").contains("a declaration is written where a statement is"));
}

#[test]
fn a_field_of_a_struct_found_through_a_reference_decided_already_takes_the_way_there() {
    holds(
        "struct Point {x, y}; o <- {f: a}; o <- {f: &Point<u8, char>};
         c: b <- a.y; assert b: char; assert c at a.y takes Deref;",
    );
}
