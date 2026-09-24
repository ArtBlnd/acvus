//! The listing `optimize::lsr` leaves (RFC-0056), and the shapes it
//! declines. The pass reduces only a counter expression that one `InOrder`
//! join reads (RFC-0066 rule 7), so every loop here that is meant to be
//! reduced carries a recurrence: `acc * 2 + …` reads the accumulator outside
//! a merge, and its join, which reads the expression, is `InOrder`.
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

/// The listing's blocks, with a loop's stage chain read as the one body it
/// is (RFC-0089 rule 1): a block without parameters that the block before
/// it enters by a bare jump continues that block.
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
            Some(header) => {
                let label = header
                    .split('(')
                    .next()
                    .unwrap_or(header)
                    .trim_end_matches(':')
                    .to_string();
                let before = found
                    .last_mut()
                    .expect("a listing opens with the entry block");
                let stepped =
                    before.insts.last() == Some(&format!("jump {label}")) && !header.contains('(');
                match stepped {
                    true => {
                        before.insts.pop();
                    }
                    false => found.push(BlockBody {
                        label,
                        insts: Vec::new(),
                    }),
                }
            }
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
        ["L1: 1".to_string(), "L6: 2".to_string()],
        "the first loop's body keeps only `reduced * 2`, and the reduction \
         steps by `@k` and starts at `@x`, the start `0 * @k + @x` written \
         folded, so nothing multiplies above the header; under the `if`, \
         `j * @k` stays beside `unreduced * 2`:\n{listing}"
    );
}

fn arithmetic_operands(block: &BlockBody) -> Vec<(String, [String; 2])> {
    block
        .insts
        .iter()
        .filter_map(|inst| inst.split_once(" = "))
        .filter_map(|(_, computed)| {
            [" * ", " + "].into_iter().find_map(|op| {
                let (left, right) = computed.split_once(op)?;
                let operand = |written: &str| written.split(' ').next().unwrap_or("").to_string();
                Some((op.trim().to_string(), [operand(left), operand(right)]))
            })
        })
        .collect()
}

/// A range from 0 by one: the reduced counter's start is `0 * @k + @x`, and
/// `lsr` is the last pass to touch it, since no value numbering follows the
/// stages (RFC-0056). What stands above the header is the folded start.
#[test]
fn a_reduced_counter_s_setup_multiplies_by_no_zero_and_adds_no_zero() {
    let i = Interner::new();
    let listing =
        compile_script_optimized(&i, RECURRENCE, &ctx(&i, &["n", "k", "x"])).expect("it compiles");
    let preheader = blocks(&listing)
        .into_iter()
        .next()
        .expect("a listing opens with the entry block");
    let on_zero: Vec<(String, [String; 2])> = arithmetic_operands(&preheader)
        .into_iter()
        .filter(|(_, operands)| operands.iter().any(|operand| operand == "0"))
        .collect();
    assert_eq!(
        on_zero,
        Vec::new(),
        "the preheader computes no `0 * k` and no `+ 0`:\n{listing}"
    );
    let x = listing
        .lines()
        .find_map(|line| line.split_once('|')?.1.trim().strip_suffix(" = fetch @x"))
        .unwrap_or_else(|| panic!("`@x` is fetched:\n{listing}"));
    let entered_with = preheader
        .insts
        .last()
        .and_then(|jump| jump.strip_prefix("jump L0("))
        .and_then(|args| args.strip_suffix(')'))
        .unwrap_or_else(|| panic!("the preheader jumps to the header:\n{listing}"));
    assert_eq!(
        entered_with.rsplit(", ").next(),
        Some(x),
        "the derived counter enters the loop at `@x` itself:\n{listing}"
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

const MERGED: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc + (i * @k + @x); \
    i = i + 1; \
} \
acc";

const RECURRENCE: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc * 2 + (i * @k + @x); \
    i = i + 1; \
} \
acc";

#[test]
fn a_merge_is_left_as_written_and_a_recurrence_is_reduced() {
    let i = Interner::new();
    let names = ctx(&i, &["n", "k", "x"]);
    let merged = compile_script_optimized(&i, MERGED, &names).expect("it compiles");
    let recurrence = compile_script_optimized(&i, RECURRENCE, &names).expect("it compiles");
    assert_eq!(
        where_it_multiplies_by_context(&merged, "@k"),
        ["L1: 1".to_string()],
        "the accumulator is a merge, joined `AnyOrder`, so the loop keeps \
         `i * @k` in its body, computed from the counter IV canonicalization \
         puts in `i`'s place, with no start or step above it (RFC-0066 \
         rule 7):\n{merged}"
    );
    assert_eq!(
        where_it_multiplies(&recurrence),
        ["L1: 1".to_string()],
        "`acc * 2` is a recurrence, joined `InOrder`, and the join reads \
         `i * @k + @x`, so the body keeps only `acc * 2`. The counter steps \
         by 1, so the reduction steps by `@k`, and its start `0 * @k + @x` \
         is written folded as `@x`, so nothing multiplies above the \
         header:\n{recurrence}"
    );
}

/// `max` declares itself associative and commutative (RFC-0082 rule 2), so
/// the loop merging through it joins `AnyOrder`; `saturating_add` over
/// `i64` declares no law, since it is not associative, so the loop merging
/// through it carries a recurrence, joined `InOrder`.
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
        where_it_multiplies_by_context(&lawful, "@k"),
        ["L1: 1".to_string()],
        "`max` is a declared merge, joined `AnyOrder`, so the loop keeps \
         `i * @k` in its body:\n{lawful}"
    );
    assert_eq!(
        where_it_multiplies(&lawless),
        Vec::<String>::new(),
        "`saturating_add` declares no law, so its join is `InOrder` and the \
         expression it reads is reduced: the body keeps no `i * @k`, and the \
         reduction steps by `@k` from `@x`, the start `0 * @k + @x` written \
         folded:\n{lawless}"
    );
}
