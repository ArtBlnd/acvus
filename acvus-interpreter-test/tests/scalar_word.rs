//! An inline scalar has one word wherever it came from (RFC-0037 rule 6,
//! `acvus_extern::repr::Word`): a value an extern returned or wrote through a
//! `&mut` compares equal, inside every structural shape and as a map key, to
//! the same value written as a literal. The `&mut` is every one a Rust body
//! can be lent in place (RFC-0039 rule 2): a parameter, a field of a derived
//! struct's projection, the payload of a derived enum's arm, the payload of a
//! host's lent `Result` and `Option`, and an instance's receiver. Each
//! program runs at both optimization levels.

use std::ops::Deref;
use std::sync::Arc;

use acvus_extern::{
    Ctx, Erased, Instance, Registry, Runtime, TyArg, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::{AcvusRuntime, Host, MemoryStorage, SequentialExecutor, Source, Value};
use acvus_interpreter_test::{check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

macro_rules! narrow_externs {
    ($($t:ident: $low:ident = $low_v:expr, $high:ident = $high_v:expr, $set:ident;)*) => {
        $(
            #[extern_fn(effect = pure)]
            fn $low() -> $t {
                $low_v
            }

            #[extern_fn(effect = pure)]
            fn $high() -> $t {
                $high_v
            }

            /// Writes `value` into the lent place (RFC-0018).
            #[extern_fn(effect = pure)]
            fn $set(place: &mut $t, value: $t) {
                *place = value;
            }
        )*

        fn narrow_registry() -> Registry<AcvusRuntime> {
            extern_registry! {
                ns: "t",
                signatures: [sig::overwrite],
                fns: [
                    $($low, $high, $set,)* set_first_i8, set_first_i16, set_first_i32,
                    set_narrow_a, set_cell, overwrite_i8, overwritten,
                ],
            }
        }
    };
}

narrow_externs! {
    i8: low_i8 = -1, high_i8 = i8::MIN, set_i8;
    i16: low_i16 = -1, high_i16 = i16::MIN, set_i16;
    i32: low_i32 = -1, high_i32 = i32::MIN, set_i32;
    i64: low_i64 = -1, high_i64 = i64::MIN, set_i64;
    u8: low_u8 = 1, high_u8 = u8::MAX, set_u8;
    u16: low_u16 = 1, high_u16 = u16::MAX, set_u16;
    u32: low_u32 = 1, high_u32 = u32::MAX, set_u32;
    u64: low_u64 = 1, high_u64 = u64::MAX, set_u64;
}

/// Writes `value` into the first element of the lent container, through
/// `Erased::get_mut`.
#[extern_fn(effect = opaque)]
fn set_first_i8<Rt>(dst: &mut [Erased<Rt, i8>], value: i8)
where
    Rt: Runtime,
{
    *dst[0].get_mut() = value;
}

#[extern_fn(effect = opaque)]
fn set_first_i16<Rt>(dst: &mut [Erased<Rt, i16>], value: i16)
where
    Rt: Runtime,
{
    *dst[0].get_mut() = value;
}

#[extern_fn(effect = opaque)]
fn set_first_i32<Rt>(dst: &mut [Erased<Rt, i32>], value: i32)
where
    Rt: Runtime,
{
    *dst[0].get_mut() = value;
}

/// A struct of narrow scalars, lent through its exclusive projection, whose
/// `a` is a `&mut i8` over the field's own word.
#[derive(TyArg)]
#[projection]
pub struct Narrow {
    a: i8,
    b: i16,
}

/// Writes `value` into the lent struct's `a` through its projection.
#[extern_fn(effect = pure)]
fn set_narrow_a(n: NarrowMut<'_>, value: i8) {
    *n.a = value;
}

/// An enum whose one payload is a narrow scalar.
#[derive(TyArg)]
#[projection]
pub enum Cell {
    Empty,
    Full(i8),
}

/// Writes `value` into the payload of the lent enum's `Full` arm.
#[extern_fn(effect = pure)]
fn set_cell<Rt>(c: CellMut<'_, Rt>, value: i8)
where
    Rt: Runtime,
{
    let mut c = c;
    if let CellArms::Full(x) = c.arms() {
        *x = value;
    }
}

mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "t",
        fn overwrite<T>(place: &mut T)
        where
            T: Var<kind::Type>;
    }
}

/// The `i8` instance of `overwrite`: its receiver is lent in place through
/// the mono glue.
#[extern_fn(instance_of = sig::overwrite, effect = pure)]
fn overwrite_i8(place: &mut i8) {
    *place = -1;
}

/// Calls the `overwrite` of its `T` on its own copy of `a`, and returns it.
#[extern_fn(effect = pure)]
fn overwritten<T, Rt>(ctx: &mut Ctx<'_, Rt>, a: T, overwrite: Instance<'_, sig::overwrite<T, Rt>, T, Rt>) -> T
where
    T: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut a = a;
    overwrite.call(ctx, &mut a, ());
    a
}

fn run(source: &str, ret: Ty, opt: Opt) -> Value {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("main parses"));
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(narrow_registry());
    let cr = check_graph(&i, parsed, &[], &FxHashMap::default(), registries, ret, opt, |_| {})
        .unwrap_or_else(|r| panic!("at {opt:?}, refused:\n  {}\n{source}", r.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime
        .block_on(interp.execute())
        .expect("the program reads no context")
}

/// `source` is `true` at both optimization levels.
fn holds(source: &str) {
    for opt in [Opt::None, Opt::Full] {
        assert!(run(source, Ty::Bool, opt).as_bool(), "at {opt:?}: {source}");
    }
}

/// One integer width: the externs that return a value of it and write one,
/// and the literals that spell the same values.
struct Width {
    name: &'static str,
    /// Returns `-1` at a signed width and `1` at an unsigned one.
    low: &'static str,
    /// Returns the value whose high bit alone is set at a signed width, and
    /// every bit at an unsigned one.
    high: &'static str,
    high_literal: &'static str,
    set: &'static str,
}

impl Width {
    fn signed(&self) -> bool {
        self.name.starts_with('i')
    }

    fn low_literal(&self) -> String {
        match self.signed() {
            true => format!("-1{}", self.name),
            false => format!("1{}", self.name),
        }
    }

    /// Each extern call beside the literal that spells its result.
    fn results(&self) -> [(&'static str, String); 2] {
        [(self.low, self.low_literal()), (self.high, self.high_literal.to_owned())]
    }
}

const WIDTHS: [Width; 8] = [
    Width { name: "i8", low: "low_i8()", high: "high_i8()", high_literal: "-128i8", set: "set_i8" },
    Width { name: "i16", low: "low_i16()", high: "high_i16()", high_literal: "-32768i16", set: "set_i16" },
    Width {
        name: "i32",
        low: "low_i32()",
        high: "high_i32()",
        high_literal: "-2147483648i32",
        set: "set_i32",
    },
    Width {
        name: "i64",
        low: "low_i64()",
        high: "high_i64()",
        high_literal: "-9223372036854775808i64",
        set: "set_i64",
    },
    Width { name: "u8", low: "low_u8()", high: "high_u8()", high_literal: "255u8", set: "set_u8" },
    Width { name: "u16", low: "low_u16()", high: "high_u16()", high_literal: "65535u16", set: "set_u16" },
    Width {
        name: "u32",
        low: "low_u32()",
        high: "high_u32()",
        high_literal: "4294967295u32",
        set: "set_u32",
    },
    Width {
        name: "u64",
        low: "low_u64()",
        high: "high_u64()",
        high_literal: "18446744073709551615u64",
        set: "set_u64",
    },
];

#[test]
fn an_extern_result_equals_its_literal_in_every_structural_shape() {
    for width in &WIDTHS {
        for (got, lit) in width.results() {
            holds(&format!("{got} == {lit}"));
            holds(&format!("({got}, 0) == ({lit}, 0)"));
            holds(&format!("[{got}] == [{lit}]"));
            holds(&format!("{{ x: {got}, }} == {{ x: {lit}, }}"));
            holds(&format!("A::X({got}) == A::X({lit})"));
            holds(&format!("Some(({got}, 0)) == Some(({lit}, 0))"));
            holds(&format!("{{ a: [({got}, 1)], }} == {{ a: [({lit}, 1)], }}"));
        }
    }
}

/// A key of these types has no `core::hash` instance, so the map keys by
/// the lambdas `hash_map_by` takes: one that hashes the integer by its
/// value, and one that hashes every key alike, so the key's own `==`
/// decides.
#[test]
fn a_map_keyed_by_an_extern_result_finds_its_literal() {
    for width in &WIDTHS {
        for (got, lit) in width.results() {
            holds(&format!(
                "let m = hash_map_by(|k| -> *k as u64, |a, b| -> a == b); \
                 insert(&mut m, {got}, 10); let q = {lit}; contains_key(&m, &q)"
            ));
            holds(&format!(
                "let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); \
                 insert(&mut m, ({got}, 1), 10); let q = ({lit}, 1); contains_key(&m, &q)"
            ));
            holds(&format!(
                "let m = hash_map_by(|k| -> k.k as u64, |a, b| -> a == b); \
                 insert(&mut m, {{ k: {got}, }}, 10); let q = {{ k: {lit}, }}; contains_key(&m, &q)"
            ));
            holds(&format!(
                "let s = hash_set_by(|k| -> 0u64, |a, b| -> a == b); \
                 insert(&mut s, ({lit}, 1)); insert(&mut s, ({got}, 1)); len(&s) == 1u64"
            ));
        }
    }
}

/// A `&mut` integer the extern writes crosses the sign in both directions
/// at a signed width: to a negative value from zero, and back to a positive
/// one from a negative one, whose word must lose the sign bits it had.
#[test]
fn a_place_an_extern_writes_through_a_reference_equals_its_literal() {
    for width in &WIDTHS {
        let name = width.name;
        let set = width.set;
        let start = format!("0{name}");
        let written = match width.signed() {
            true => format!("-1{name}"),
            false => format!("7{name}"),
        };
        holds(&format!("let x = {start}; {set}(&mut x, {written}); (x, 0) == ({written}, 0)"));
        holds(&format!(
            "let o = {{ x: {start}, }}; {set}(&mut o.x, {written}); o == {{ x: {written}, }}"
        ));
        holds(&format!("let x = {written}; {set}(&mut x, 1{name}); (x, 0) == (1{name}, 0)"));
        holds(&format!(
            "let x = {start}; {set}(&mut x, {written}); \
             let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); insert(&mut m, (x, 1), 10); \
             let q = ({written}, 1); contains_key(&m, &q)"
        ));
    }
}

/// `Erased::get_mut` lends an element of a container in place; the loan
/// ends with the borrow.
#[test]
fn an_element_an_extern_writes_in_place_equals_its_literal() {
    for width in WIDTHS.iter().filter(|w| matches!(w.name, "i8" | "i16" | "i32")) {
        let name = width.name;
        let set_first = format!("set_first_{name}");
        holds(&format!(
            "let a = vec([0{name}, 5{name}]); {set_first}(&mut a, -1{name}); \
             let e = a[0u64]; (e, 0) == (-1{name}, 0)"
        ));
        holds(&format!(
            "let a = vec([-1{name}, 5{name}]); {set_first}(&mut a, 1{name}); \
             let e = a[0u64]; (e, 0) == (1{name}, 0)"
        ));
    }
}

/// A `&mut i8` field of a derived struct's projection, written across the
/// sign in both directions; the struct is then compared whole, in a tuple,
/// and as a map key, with one built from literals.
#[test]
fn a_field_an_extern_writes_through_a_struct_projection_equals_its_literal() {
    for (start, written) in [("0i8", "-1i8"), ("-1i8", "1i8")] {
        let lent = format!("let s = {{ a: {start}, b: -2i16, }}; set_narrow_a(&mut s, {written});");
        let literal = format!("{{ a: {written}, b: -2i16, }}");
        holds(&format!("{lent} s == {literal}"));
        holds(&format!("{lent} (s, 0) == ({literal}, 0)"));
        holds(&format!(
            "{lent} let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); insert(&mut m, s, 10); \
             let q = {literal}; contains_key(&m, &q)"
        ));
    }
}

/// The payload of a derived enum's arm, lent through `CellMut::arms`.
#[test]
fn a_payload_an_extern_writes_through_an_enum_arm_equals_its_literal() {
    for (start, written) in [("0i8", "-1i8"), ("-1i8", "1i8")] {
        let lent = format!(
            "let c = if 0 < 1 {{ Cell::Full({start}) }} else {{ Cell::Empty }}; set_cell(&mut c, {written});"
        );
        let literal = format!("Cell::Full({written})");
        holds(&format!("{lent} c == {literal}"));
        holds(&format!("{lent} (c, 0) == ({literal}, 0)"));
        holds(&format!(
            "{lent} let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); insert(&mut m, c, 10); \
             let q = {literal}; contains_key(&m, &q)"
        ));
    }
}

/// An instance's `&mut` receiver, lent in place by its mono glue.
#[test]
fn a_receiver_an_instance_writes_equals_its_literal() {
    holds("(overwritten(0i8), 0) == (-1i8, 0)");
    holds(
        "let x = overwritten(0i8); let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); \
         insert(&mut m, (x, 1), 10); let q = (-1i8, 1); contains_key(&m, &q)",
    );
}

/// `@key` is seeded by `seed`, edited in place by `edit` through the page's
/// `with_mut`, and then `check` must answer `true`, at both optimization
/// levels.
async fn edited_context_holds<Q, F>(seed: &str, check: &str, edit: F)
where
    F: acvus_extern::Borrows<AcvusRuntime, Q, ()> + Clone,
{
    for opt in [Opt::None, Opt::Full] {
        let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
        registries.push(narrow_registry());
        let host = Host::new(registries)
            .opt(opt)
            .entry::<(), ()>("seed", Source::Script(seed))
            .entry::<(), bool>("check", Source::Script(check));
        let program = match host.compile(SequentialExecutor) {
            Ok(program) => program,
            Err(error) => panic!("at {opt:?}, refused: {error:?}"),
        };
        let edit = edit.clone();
        program
            .scope(async |s| {
                let mut storage = MemoryStorage::new();
                let mut page = s.open(&mut storage);
                let seeded = s.entry::<(), ()>("seed").expect("`seed` returns `()`");
                seeded.run(&mut page, ()).await.expect("`seed` reads no context");
                page.with_mut("r", edit).await.expect("`@r` is held at the closure's type");
                let checked = s.entry::<(), bool>("check").expect("`check` returns `bool`");
                let output = checked.run(&mut page, ()).await.expect("`@r` is held");
                assert!(output.with(|b: &bool| *b).expect("a `bool`"), "at {opt:?}: {check}");
            })
            .await;
    }
}

/// A host's `Result<&mut i8, _>`: the payload of the `Ok` arm, lent in place.
#[tokio::test]
async fn a_result_payload_a_host_writes_equals_its_literal() {
    let seed = r#"@r = if 0 < 1 { Ok(0i8) } else { Err("e".to_string()) };"#;
    let set = |r: Result<&mut i8, &mut String>| {
        *r.expect("`seed` wrote `Ok`") = -1;
    };
    let literal = r#"if 0 < 1 { Ok(-1i8) } else { Err("e".to_string()) }"#;
    // A `Result` holding a `String` is moved out of `@r` by the read, and a
    // run leaves every context it moved assigned, so each check writes
    // `@r` back.
    let back = format!("@r = {literal};");
    edited_context_holds(seed, &format!("let ok = @r == {literal}; {back} ok"), set).await;
    edited_context_holds(seed, &format!("let ok = (@r, 0) == ({literal}, 0); {back} ok"), set).await;
    edited_context_holds(
        seed,
        &format!(
            "let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); insert(&mut m, @r, 10); \
             let q = {literal}; let ok = contains_key(&m, &q); {back} ok"
        ),
        set,
    )
    .await;
}

/// A host's `Option<&mut i16>`: the payload of a `Some`, lent in place.
#[tokio::test]
async fn an_option_payload_a_host_writes_equals_its_literal() {
    let seed = "@r = if 0 < 1 { Some(0i16) } else { None };";
    let set = |o: Option<&mut i16>| {
        *o.expect("`seed` wrote `Some`") = -1;
    };
    let literal = "if 0 < 1 { Some(-1i16) } else { None }";
    edited_context_holds(seed, &format!("@r == {literal}"), set).await;
    edited_context_holds(seed, &format!("(@r, 0) == ({literal}, 0)"), set).await;
    edited_context_holds(
        seed,
        &format!(
            "let m = hash_map_by(|k| -> 0u64, |a, b| -> a == b); insert(&mut m, @r, 10); \
             let q = {literal}; contains_key(&m, &q)"
        ),
        set,
    )
    .await;
}
