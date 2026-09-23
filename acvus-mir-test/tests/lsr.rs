//! The listing `optimize::lsr` leaves (RFC-0056), and the shapes it
//! declines. The pass reduces only strong loops (`analysis::carried`), so
//! every loop here that is meant to be reduced carries a recurrence:
//! `acc * 2 + …` reads the accumulator outside a merge.
//!
//! `acvus-interpreter-test/tests/strength_reduction.rs` runs the same two
//! loops and reads their values; this file is what says the two loops are
//! not the same loop — that the first lost its multiplication and the
//! second kept it — so that the value test is not comparing a program with
//! itself. Both must move together.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const ENTRY: &str = "ENTRY";

struct BlockBody {
    label: String,
    insts: Vec<String>,
}

impl BlockBody {
    fn multiplications(&self) -> usize {
        self.insts
            .iter()
            .filter(|inst| inst.contains(" * "))
            .count()
    }

    fn multiplications_by(&self, factor: &str) -> usize {
        self.insts
            .iter()
            .filter_map(|inst| inst.split_once(" = "))
            .filter_map(|(_, product)| product.split_once(" * "))
            .filter(|(left, right)| {
                [left, right]
                    .iter()
                    .any(|operand| operand.split(' ').next() == Some(factor))
            })
            .count()
    }
}

fn blocks(listing: &str) -> Vec<BlockBody> {
    let mut found = vec![BlockBody {
        label: ENTRY.to_string(),
        insts: Vec::new(),
    }];
    for line in listing.lines() {
        let Some((_, rest)) = line.split_once('|') else {
            continue;
        };
        let rest = rest.trim_end();
        match rest.strip_prefix(' ').filter(|r| !r.starts_with(' ')) {
            Some(header) => found.push(BlockBody {
                label: header
                    .split('(')
                    .next()
                    .unwrap_or(header)
                    .trim_end_matches(':')
                    .to_string(),
                insts: Vec::new(),
            }),
            None => {
                let inst = rest.trim();
                if !inst.is_empty() {
                    found
                        .last_mut()
                        .expect("a listing opens with the entry block")
                        .insts
                        .push(inst.to_string());
                }
            }
        }
    }
    found
}

fn where_it_multiplies(listing: &str) -> Vec<String> {
    blocks(listing)
        .iter()
        .filter(|block| block.multiplications() > 0)
        .map(|block| format!("{}: {}", block.label, block.multiplications()))
        .collect()
}

fn where_it_multiplies_by_context(listing: &str, context: &str) -> Vec<String> {
    let fetch = format!(" = fetch {context}");
    let Some(factor) = listing.lines().find_map(|line| {
        let (_, inst) = line.split_once('|')?;
        inst.trim().strip_suffix(fetch.as_str())
    }) else {
        panic!("`{context}` is fetched:\n{listing}");
    };
    blocks(listing)
        .iter()
        .filter(|block| block.multiplications_by(factor) > 0)
        .map(|block| format!("{}: {}", block.label, block.multiplications_by(factor)))
        .collect()
}

fn multiplications(listing: &str) -> usize {
    blocks(listing).iter().map(BlockBody::multiplications).sum()
}

fn ctx(i: &Interner, names: &[&str]) -> FxHashMap<Astr, Ty> {
    names.iter().map(|n| (i.intern(n), Ty::I64)).collect()
}

const BOTH_FORMS: &str = "\
let reduced = 0; \
let i = 0; \
while i < @n { \
    reduced = reduced * 2 + (i * @k + @x); \
    i = i + 1; \
} \
let unreduced = 0; \
let j = 0; \
while j < @n { \
    if j + 1 > j { unreduced = unreduced * 2 + (j * @k + @x); }; \
    j = j + 1; \
} \
reduced - unreduced";

#[test]
fn the_reduced_body_multiplies_and_the_unreduced_one_still_does() {
    let i = Interner::new();
    let listing =
        compile_script_optimized(&i, BOTH_FORMS, &ctx(&i, &["n", "k", "x"])).expect("it compiles");
    insta::assert_snapshot!("both_forms", listing);

    assert_eq!(
        where_it_multiplies(&listing),
        [
            format!("{ENTRY}: 2"),
            "L1: 1".to_string(),
            "L6: 2".to_string()
        ],
        "the reduction's start and step stand above the first loop, whose body \
         keeps only `reduced * 2`; under the `if`, `j * @k` stays beside \
         `unreduced * 2`:\n{listing}"
    );
}

/// Two bare products: the first read by a sum that is not loop-invariant,
/// the second read twice. Neither has the one invariant sum that makes the
/// operation count fall, so both multiplications stay in the body.
const BARE_PRODUCTS: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc * 2 + i * @k; \
    i = i + 1; \
} \
let shared = 0; \
let j = 0; \
while j < @n { \
    let p = j * @k; \
    shared = shared + p + (p + @x); \
    j = j + 1; \
} \
acc - shared";

#[test]
fn a_bare_product_keeps_its_multiplication() {
    let i = Interner::new();
    let listing = compile_script_optimized(&i, BARE_PRODUCTS, &ctx(&i, &["n", "k", "x"]))
        .expect("it compiles");
    insta::assert_snapshot!("bare_products", listing);
    assert_eq!(
        where_it_multiplies(&listing),
        ["L1: 2".to_string(), "L4: 1".to_string()],
        "both bodies keep the multiplication they were written with, beside \
         the first body's `acc * 2`, and no start or step stands above either \
         header: a reduction that does not lower the operation count is not \
         applied (RFC-0056):\n{listing}"
    );
}

const FLOAT_PRODUCT: &str = "\
let acc = 0.0; \
let x = 0.0; \
let i = 0; \
while i < @n { \
    acc = acc + x * 2.0; \
    x = x + 1.0; \
    i = i + 1; \
} \
acc";

#[test]
fn a_float_product_keeps_its_multiplication() {
    let i = Interner::new();
    let listing =
        compile_script_optimized(&i, FLOAT_PRODUCT, &ctx(&i, &["n"])).expect("it compiles");
    insta::assert_snapshot!("float_product", listing);
    assert_eq!(
        multiplications(&listing),
        1,
        "`x * 2.0` stays where it is: accumulating it would round differently:\n{listing}"
    );
}

const WEAK: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc + (i * @k + @x); \
    i = i + 1; \
} \
acc";

const STRONG: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc * 2 + (i * @k + @x); \
    i = i + 1; \
} \
acc";

#[test]
fn a_weak_loop_is_left_as_written_and_a_strong_one_is_reduced() {
    let i = Interner::new();
    let names = ctx(&i, &["n", "k", "x"]);
    let weak = compile_script_optimized(&i, WEAK, &names).expect("it compiles");
    let strong = compile_script_optimized(&i, STRONG, &names).expect("it compiles");
    assert_eq!(
        where_it_multiplies_by_context(&weak, "@k"),
        ["L1: 1".to_string()],
        "the accumulator is a merge and `i` an induction variable, so the loop \
         is weak and keeps `i * @k` in its body, with no start or step above \
         it (RFC-0056); its `i` is IV canonicalization's (RFC-0066 rule 7):\n{weak}"
    );
    assert_eq!(
        where_it_multiplies(&strong),
        [format!("{ENTRY}: 2"), "L1: 1".to_string()],
        "`acc * 2` is a recurrence, so the loop is strong: the reduction's \
         start and step stand above it and the body keeps only `acc * 2`:\n{strong}"
    );
}

/// `max` declares itself associative and commutative (RFC-0082 rule 2), so
/// the loop merging through it is weak; `saturating_add` over `i64`
/// declares no law, since it is not associative, so the loop merging
/// through it carries a recurrence and is strong.
const MERGED_BY_LAW: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = max(acc, i * @k + @x); \
    i = i + 1; \
} \
acc";

const MERGED_WITHOUT_LAW: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = saturating_add(acc, i * @k + @x); \
    i = i + 1; \
} \
acc";

#[test]
fn a_loop_merging_through_a_lawful_extern_is_left_as_written() {
    let i = Interner::new();
    let names = ctx(&i, &["n", "k", "x"]);
    let lawful = compile_script_optimized(&i, MERGED_BY_LAW, &names).expect("it compiles");
    let lawless = compile_script_optimized(&i, MERGED_WITHOUT_LAW, &names).expect("it compiles");
    assert_eq!(
        where_it_multiplies(&lawful),
        ["L1: 1".to_string()],
        "`max` is a declared merge and `i` an induction variable, so the loop is \
         weak and keeps `i * @k` in its body:\n{lawful}"
    );
    assert_eq!(
        where_it_multiplies(&lawless),
        [format!("{ENTRY}: 2")],
        "`saturating_add` declares no law, so the loop is strong and reduced:\n{lawless}"
    );
}
