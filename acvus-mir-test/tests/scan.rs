//! A cycle whose law's partials are read outside it is a scan (RFC-0093
//! rule 8): what `analysis::loop_deps` judges of such a cycle after the full
//! pipeline, and what the facts print beside its law.

use acvus_mir::analysis::loop_deps::{
    CycleLaw, Law, LawOp, LoopDeps, Member, Order, Storage, Token,
};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loops::natural_loops_innermost_first;
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{BinOp, InstKind};
use acvus_mir::printer::dump_with_facts;
use acvus_mir_test::compile_script_at;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct JudgedCycle {
    tokens: Vec<Token>,
    holds: Vec<&'static str>,
    order: Order,
    law: Option<CycleLaw>,
}

struct Scanned {
    listing: String,
    cycles: Vec<JudgedCycle>,
}

impl Scanned {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
        let cfg = promote(compiled.module.main.clone());
        let header = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
            .iter()
            .filter(|loop_| matches!(cfg.blocks[loop_.header.0].terminator, Terminator::For { .. }))
            .max_by_key(|loop_| loop_.block_count())
            .map(|loop_| loop_.header)
            .unwrap_or_else(|| panic!("a `for`:\n{listing}"));
        let deps = LoopDeps::of(&cfg, &compiled.laws, header)
            .unwrap_or_else(|fault| panic!("{}:\n{listing}", fault.shown()));
        let judged = deps.judge(&cfg, &compiled.laws);
        let cycles = deps
            .cycles
            .iter()
            .zip(judged)
            .map(|(cycle, judged)| JudgedCycle {
                tokens: cycle.tokens.clone(),
                holds: cycle
                    .members
                    .iter()
                    .filter_map(|member| match *member {
                        Member::Inst(at) => Some(kind_name(&cfg, at.block, at.at)),
                        Member::Term(_) => None,
                    })
                    .collect(),
                order: judged.order,
                law: judged.law,
            })
            .collect();
        Self { listing, cycles }
    }

    fn cycle_where(&self, holds: impl Fn(&[Token]) -> bool) -> &JudgedCycle {
        let found: Vec<&JudgedCycle> = self
            .cycles
            .iter()
            .filter(|cycle| holds(&cycle.tokens))
            .collect();
        match found[..] {
            [one] => one,
            _ => panic!("one such cycle:\n{}", self.listing),
        }
    }

    fn carried_cycle(&self) -> &JudgedCycle {
        self.cycle_where(|tokens| matches!(tokens, [Token::Carried(_)]))
    }
}

fn kind_name(cfg: &CfgBody, block: BlockIdx, at: usize) -> &'static str {
    match &cfg.blocks[block.0].insts[at].kind {
        InstKind::BinOp { op, .. } => match op {
            BinOp::Add(_) => "+",
            BinOp::Sub(_) => "-",
            BinOp::Mul(_) => "*",
            BinOp::Gt => ">",
            _ => "binop",
        },
        InstKind::Index { .. } => "index",
        InstKind::IndexSet { .. } => "index_set",
        InstKind::Const { .. } => "const",
        _ => "other",
    }
}

fn law_of(cycle: &JudgedCycle) -> Option<(&Law, bool)> {
    cycle
        .law
        .as_ref()
        .map(|law| (&law.accumulator.law, law.scan))
}

/// Corpus row C01; the order is RFC-0092 rule 5's.
#[test]
fn a_partial_read_by_a_later_stage_marks_the_law_a_scan_in_order() {
    let c = Scanned::of(
        "let v = [5, 3, 8, 1]; let s = 0; let out = vec::new(); \
         for x in &v { s = s + *x; vec::push(&mut out, s); } vec::len(&out)",
    );
    let held = c.carried_cycle();
    assert_eq!(law_of(held), Some((&Law::Op(LawOp::Add), true)), "{}", c.listing);
    assert_eq!(held.order, Order::InOrder, "{}", c.listing);
    assert!(
        c.listing
            .contains("in_order law(Op(Add) exact commutative) scan {+}"),
        "the facts say `scan` beside the law:\n{}",
        c.listing
    );
}

/// Corpus row Z07.
#[test]
fn a_cycle_without_a_law_is_no_scan_though_its_value_is_read() {
    let c = Scanned::of(
        "let v = [1, 2, 3]; let s = 2; let out = vec::new(); \
         for x in &v { s = s * s + *x; vec::push(&mut out, s); } vec::len(&out)",
    );
    let held = c.carried_cycle();
    assert_eq!(law_of(held), None, "{}", c.listing);
    assert_eq!(held.order, Order::InOrder, "{}", c.listing);
    assert!(!c.listing.contains(" scan "), "{}", c.listing);
}

/// RFC-0093 rule 8: "a reader that writes the token is in the cycle, not a
/// reader". `t0 > 3` reads the total as the outer iteration received it and
/// decides how much of the row the inner loop adds to it.
#[test]
fn a_reader_that_writes_the_token_is_in_the_cycle() {
    let c = Scanned::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { let t0 = total; \
         for x in &row { if t0 > 3 { break; }; total = total + *x; } } total",
    );
    let held = c.carried_cycle();
    assert!(held.holds.contains(&">"), "{:?}\n{}", held.holds, c.listing);
    assert_eq!(law_of(held), None, "{}", c.listing);
    assert_eq!(held.order, Order::InOrder, "{}", c.listing);
}

/// Corpus rows C03 and Z03.
#[test]
fn a_read_of_the_previous_iterations_place_is_the_previous_partial() {
    let storage = |c: &Scanned| {
        law_of(c.cycle_where(|tokens| matches!(tokens, [Token::Storage(Storage::Slot(_))])))
            .map(|(law, scan)| (law.clone(), scan))
    };
    let previous = Scanned::of(
        "let v = vec([5, 3, 8, 1]); for i in 1u64..v.len() { v[i] = v[i] + v[i - 1u64]; } v[3u64]",
    );
    assert_eq!(
        storage(&previous),
        Some((Law::Op(LawOp::Add), true)),
        "{}",
        previous.listing
    );
    let two_back = Scanned::of(
        "let v = vec([5, 3, 8, 1]); for i in 2u64..v.len() { v[i] = v[i] + v[i - 2u64]; } v[3u64]",
    );
    assert_eq!(storage(&two_back), None, "{}", two_back.listing);
    let squared = Scanned::of(
        "let v = vec([3, 0, 0]); for i in 1u64..v.len() { v[i] = v[i - 1u64] * v[i - 1u64] + 1; } v[2u64]",
    );
    assert_eq!(storage(&squared), None, "{}", squared.listing);
}

/// Corpus rows C08, P07 and Z16.
#[test]
fn an_affine_update_has_the_affine_map_law_at_an_integer_width_only() {
    let pushed = Scanned::of(
        "let v = [1, 2, 3, 4]; let y = 0; let out = vec::new(); \
         for x in &v { y = 2 * y + *x; vec::push(&mut out, y); } vec::len(&out)",
    );
    assert_eq!(law_of(pushed.carried_cycle()), Some((&Law::AffineMap, true)), "{}", pushed.listing);
    assert!(pushed.listing.contains("law(AffineMap exact) scan"), "{}", pushed.listing);
    let alone = Scanned::of("let v = [9, 0, 2]; let n = 0; for d in &v { n = n * 10 + *d; } n");
    let held = alone.carried_cycle();
    assert_eq!(law_of(held), Some((&Law::AffineMap, false)), "{}", alone.listing);
    assert_eq!(held.order, Order::InOrder, "{}", alone.listing);
    let float = Scanned::of(
        "let v = [2.0, 4.0, 3.0]; let m = 0.0; let out = vec::new(); \
         for x in &v { m = m * 0.5 + *x; vec::push(&mut out, m); } vec::len(&out)",
    );
    let held = float.carried_cycle();
    assert_eq!(law_of(held), None, "{}", float.listing);
    assert_eq!(held.order, Order::InOrder, "{}", float.listing);
}

/// Corpus rows C06 (exclusive) and C01 (inclusive).
#[test]
fn an_exclusive_and_an_inclusive_read_are_both_scans() {
    let exclusive = Scanned::of(
        "let v = [3, 4, 1]; let total = 0; let out = vec::new(); \
         for x in &v { vec::push(&mut out, total); total = total + *x; } vec::len(&out)",
    );
    assert_eq!(
        law_of(exclusive.carried_cycle()),
        Some((&Law::Op(LawOp::Add), true)),
        "{}",
        exclusive.listing
    );
    let inclusive = Scanned::of(
        "let v = [3, 4, 1]; let total = 0; let out = vec::new(); \
         for x in &v { total = total + *x; vec::push(&mut out, total); } vec::len(&out)",
    );
    assert_eq!(
        law_of(inclusive.carried_cycle()),
        Some((&Law::Op(LawOp::Add), true)),
        "{}",
        inclusive.listing
    );
    let between = Scanned::of(
        "let v = [3, 4, 1]; let total = 0; let out = vec::new(); \
         for x in &v { let t = total + *x; vec::push(&mut out, t); total = t * 3; } vec::len(&out)",
    );
    assert_eq!(law_of(between.carried_cycle()), None, "{}", between.listing);
}
