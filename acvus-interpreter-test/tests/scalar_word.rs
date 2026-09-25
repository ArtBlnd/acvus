//! An inline scalar has one word wherever it came from (RFC-0037 rule 6,
//! `acvus_extern::repr::Word`): a value an extern returned or wrote through a
//! `&mut` compares equal, inside every structural shape and as a map key, to
//! the same value written as a literal. Each program runs at both
//! optimization levels.

use std::sync::Arc;

use acvus_extern::{Erased, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
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
                fns: [$($low, $high, $set,)* set_first_i8, set_first_i16, set_first_i32],
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
