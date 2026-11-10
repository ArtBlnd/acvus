use super::parse;
use super::record::run;

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
assert a: Option<u8>;
assert a.x.*: &mut {x: (), y: Error{V(E)}};
assert accepted;
assert refused ..;
assert refused Mismatch, ..;
assert refused Mismatch, NoChange, EmptyDomain, Occurs, NotHeld, UndecidedClass;
assert refused Mismatch names;
assert refused Mismatch names first, second;
assert first takes;
assert first takes Deref, Reborrow, Never, DropFields, AddVariants, Unsize, as_slice;
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
         call: pass r to p;
         assert call takes Reborrow, as_slice;",
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
        Err("in order 0, `assert a: u16;` does not hold: `a` is `u8`".to_owned())
    );
    assert_eq!(
        run("a <- u8; a <- u16;", 2),
        Err("in order 0, `assert accepted;` does not hold: refused by {Mismatch, UndecidedClass}".to_owned())
    );
    assert_eq!(run("a <- u8; a <- u16; assert refused Mismatch, ..;", 2), Ok(()));
    assert_eq!(run("a <- u8; a <- u16; assert refused ..;", 2), Ok(()));
    assert_eq!(
        run("a <- u8; assert refused ..;", 2),
        Err("in order 0, `assert refused ..;` does not hold: accepted".to_owned())
    );
}

#[test]
fn what_cannot_be_recorded_is_said_before_anything_is_solved() {
    assert_eq!(run("declare p = u8; declare p = u16;", 1), Err("`p` is declared twice".to_owned()));
    assert_eq!(run("declare p = Option<p>;", 1), Err("`p` is declared as a type that names it".to_owned()));
    assert_eq!(run("row f: a -> u8;", 1), Err("a row names the class `a`".to_owned()));
    assert_eq!(run("row f: {integer} -> u8;", 1), Err("a row holds a literal".to_owned()));
}

#[test]
fn a_shared_reference_read_through_is_not_passed_as_an_exclusive_one() {
    // `put(&mut r)` with `r: &u8`, to `put(&mut u8)` and to `put(&u8)`.
    let program = |given: &str, expected: &str| format!("x <- u8; r <- &x; m <- &mut r; declare p = {given}; call: pass m to p; {expected}");
    holds(&program("&mut u8", "assert refused NoChange;"));
    holds(&program("&u8", "assert call takes Deref, Reborrow;"));
}
