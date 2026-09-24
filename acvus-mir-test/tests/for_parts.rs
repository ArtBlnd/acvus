//! What `optimize::for_parts` leaves (RFC-0089): the terminator each loop
//! ends in after the full pipeline, its parts in order, each part's kind and
//! each accumulator's law.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::Function;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{InstKind, Label, Law, LawOp, Part, PartKind};
use acvus_mir::printer::dump_with;
use acvus_mir_test::{compile_script_at, optimized_script_module};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

struct Compiled {
    listing: String,
    cfg: CfgBody,
}

impl Compiled {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from_module(&interner, compiled.module)
    }

    fn with_io(source: &str) -> Self {
        let interner = Interner::new();
        let module = optimized_script_module(&interner, source, &io_functions(&interner))
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from_module(&interner, module)
    }

    fn from_module(interner: &Interner, module: acvus_mir::ir::MirModule) -> Self {
        let listing = dump_with(interner, &module);
        Self {
            listing,
            cfg: promote(module.main),
        }
    }

    fn only_loop(&self) -> Looped<'_> {
        let mut found: Vec<Looped<'_>> = self
            .cfg
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(at, block)| match &block.terminator {
                Terminator::For { .. } => Some(Looped::For),
                Terminator::ForParts { parts, .. } => Some(Looped::Parts {
                    header: BlockIdx(at),
                    parts,
                }),
                _ => None,
            })
            .collect();
        match found.len() {
            0 => Looped::Neither,
            1 => found.remove(0),
            n => panic!("{n} loops where one was written:\n{}", self.listing),
        }
    }

    fn parts(&self) -> (BlockIdx, &[Part]) {
        match self.only_loop() {
            Looped::Parts { header, parts } => (header, parts),
            Looped::For => panic!("the loop stayed a `For`:\n{}", self.listing),
            Looped::Neither => panic!("no `for` is left:\n{}", self.listing),
        }
    }

    /// The blocks of part `index`: what its entry reaches before the next
    /// part's entry or the header.
    fn blocks_of(&self, header: BlockIdx, parts: &[Part], index: usize) -> Vec<BlockIdx> {
        let entry = |at: usize| self.cfg.label_to_block[&parts[at].entry];
        let stop: Vec<BlockIdx> = (0..parts.len()).map(entry).chain([header]).collect();
        let mut seen: FxHashSet<BlockIdx> = FxHashSet::from_iter([entry(index)]);
        let mut work = vec![entry(index)];
        while let Some(block) = work.pop() {
            for succ in self.cfg.successors(block) {
                if !stop.contains(&succ) && seen.insert(succ) {
                    work.push(succ);
                }
            }
        }
        let mut blocks: Vec<BlockIdx> = seen.into_iter().collect();
        blocks.sort();
        blocks
    }
}

enum Looped<'a> {
    For,
    Parts { header: BlockIdx, parts: &'a [Part] },
    Neither,
}

/// `io::print`, the effectful extern `acvus-ext` holds outside the standard
/// registries.
fn io_functions(interner: &Interner) -> Vec<Function> {
    let core: FxHashSet<_> = Externs::<TypesOnly>::combine(vec![], interner)
        .expect("core combines")
        .functions
        .into_iter()
        .map(|function| function.qref)
        .collect();
    Externs::combine(vec![acvus_ext::io_registry::<TypesOnly>()], interner)
        .expect("the io registry combines")
        .functions
        .into_iter()
        .filter(|function| !core.contains(&function.qref))
        .collect()
}

fn laws(part: &Part) -> Vec<&Law> {
    match &part.kind {
        PartKind::Law(accs) => accs.iter().map(|acc| &acc.law).collect(),
        PartKind::Sequential => panic!("part at {:?} is `Sequential`", part.entry),
    }
}

fn is_sequential(part: &Part) -> bool {
    matches!(part.kind, PartKind::Sequential)
}

#[test]
fn a_sum_and_a_product_are_two_exact_op_laws() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { s = s + *x; p = p * *x; } s + p",
    );
    let (_, parts) = c.parts();
    assert_eq!(parts.len(), 2, "{}", c.listing);
    assert_eq!(laws(&parts[0]), [&Law::Op(LawOp::Add)], "{}", c.listing);
    assert_eq!(laws(&parts[1]), [&Law::Op(LawOp::Mul)], "{}", c.listing);
    for part in parts {
        let PartKind::Law(accs) = &part.kind else {
            unreachable!("checked above")
        };
        assert!(
            accs.iter().all(|acc| acc.exact && acc.commutative),
            "{}",
            c.listing
        );
    }
}

#[test]
fn three_independent_recurrences_are_three_sequential_parts() {
    let c = Compiled::of(
        "let a = 0; let b = 0; let c = 0; \
         for i in 0..10 { a = a * a + i; b = b * b + 1; c = c * c + 2; } a + b + c",
    );
    let (_, parts) = c.parts();
    assert_eq!(parts.len(), 3, "{}", c.listing);
    assert!(parts.iter().all(is_sequential), "{}", c.listing);
    assert!(
        parts.iter().all(|part| part.carried.len() == 1),
        "{}",
        c.listing
    );
}

#[test]
fn a_recurrence_and_a_sum_are_a_sequential_part_and_a_law_part() {
    let c = Compiled::of("let a = 1; let s = 0; for i in 0..5 { a = a * a + i; s = s + i; } a + s");
    let (_, parts) = c.parts();
    assert_eq!(parts.len(), 2, "{}", c.listing);
    assert!(is_sequential(&parts[0]), "{}", c.listing);
    assert_eq!(laws(&parts[1]), [&Law::Op(LawOp::Add)], "{}", c.listing);
}

#[test]
fn two_sums_of_one_computation_are_one_law_part() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let t = 0; \
         for x in &v { let y = *x * *x + 3; s = s + y; t = t + y; } s + t",
    );
    let (_, parts) = c.parts();
    assert_eq!(parts.len(), 1, "{}", c.listing);
    assert_eq!(
        laws(&parts[0]),
        [&Law::Op(LawOp::Add), &Law::Op(LawOp::Add)],
        "{}",
        c.listing
    );
}

#[test]
fn a_float_sum_is_an_inexact_op_law() {
    let c = Compiled::of("let v = [1.5, 2.5, 3.0]; let s = 0.0; for x in &v { s = s + *x; } s");
    let (_, parts) = c.parts();
    let [part] = parts else {
        panic!("one part:\n{}", c.listing)
    };
    let PartKind::Law(accs) = &part.kind else {
        panic!("a law:\n{}", c.listing)
    };
    let [acc] = &accs[..] else {
        panic!("one accumulator:\n{}", c.listing)
    };
    assert_eq!(acc.law, Law::Op(LawOp::Add), "{}", c.listing);
    assert!(!acc.exact, "{}", c.listing);
}

#[test]
fn a_min_is_a_commutative_call_law() {
    let c = Compiled::of("let v = [5, 2, 3, 4]; let m = 100; for x in &v { m = min(m, *x); } m");
    let (_, parts) = c.parts();
    let [part] = parts else {
        panic!("one part:\n{}", c.listing)
    };
    let PartKind::Law(accs) = &part.kind else {
        panic!("a law:\n{}", c.listing)
    };
    let [acc] = &accs[..] else {
        panic!("one accumulator:\n{}", c.listing)
    };
    assert!(matches!(acc.law, Law::Call(_)), "{}", c.listing);
    assert!(acc.commutative, "{}", c.listing);
}

#[test]
fn a_push_with_a_fold_law_is_a_fold_law() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let out = vec::new(); \
         for x in &v { vec::push(&mut out, *x); } vec::len(&out)",
    );
    let (_, parts) = c.parts();
    let [part] = parts else {
        panic!("one part:\n{}", c.listing)
    };
    assert!(matches!(laws(part)[..], [Law::Fold(_)]), "{}", c.listing);
}

#[test]
fn an_effectful_call_inside_anyorder_is_an_order_law() {
    let c =
        Compiled::with_io("let v = [1, 2, 3]; anyorder { for x in &v { io::print(\"a\"); } } 0");
    let (_, parts) = c.parts();
    let [part] = parts else {
        panic!("one part:\n{}", c.listing)
    };
    assert_eq!(laws(part), [&Law::Order], "{}", c.listing);
}

#[test]
fn a_loop_with_break_stays_a_for() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { if *x > 2 { break; } s = s + *x; p = p * *x; } s + p",
    );
    assert!(matches!(c.only_loop(), Looped::For), "{}", c.listing);
}

#[test]
fn one_recurrence_alone_stays_a_for() {
    let c = Compiled::of("let a = 1; for i in 0..5 { a = a * a + i; } a");
    assert!(matches!(c.only_loop(), Looped::For), "{}", c.listing);
}

#[test]
fn a_while_is_not_rewritten() {
    let c = Compiled::of("let i = 0; let s = 0; while i < 10 { s = s + i; i = i + 2; } s");
    assert!(matches!(c.only_loop(), Looped::Neither), "{}", c.listing);
    assert!(!c.listing.contains(" parts ["), "{}", c.listing);
}

#[test]
fn an_if_in_the_body_and_the_sum_it_guards_are_one_part() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { if *x > 2 { s = s + *x; } p = p * *x; } s + p",
    );
    let (header, parts) = c.parts();
    assert_eq!(parts.len(), 2, "{}", c.listing);
    let branches: Vec<usize> = (0..parts.len())
        .filter(|&index| {
            c.blocks_of(header, parts, index).iter().any(|block| {
                matches!(
                    c.cfg.blocks[block.0].terminator,
                    Terminator::Diamond { .. } | Terminator::JumpIf { .. }
                )
            })
        })
        .collect();
    let [branching] = branches[..] else {
        panic!("one part holds the `if`:\n{}", c.listing)
    };
    let s = parts[branching].carried[0];
    let adds_s = c
        .blocks_of(header, parts, branching)
        .iter()
        .flat_map(|block| &c.cfg.blocks[block.0].insts)
        .any(|inst| {
            matches!(&inst.kind, InstKind::BinOp { left, right, .. }
                if *left == s || *right == s)
        });
    assert!(
        adds_s,
        "the sum the `if` guards is in its part:\n{}",
        c.listing
    );
    assert_eq!(parts[branching].carried.len(), 1, "{}", c.listing);
    let other = 1 - branching;
    assert_eq!(laws(&parts[other]), [&Law::Op(LawOp::Mul)], "{}", c.listing);
}

#[test]
fn every_part_entry_is_a_block_and_the_first_is_the_body() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { s = s + *x; p = p * *x; } s + p",
    );
    let (header, parts) = c.parts();
    let Terminator::ForParts { source, body, .. } = &c.cfg.blocks[header.0].terminator else {
        unreachable!("`parts` found a `for_parts` here")
    };
    assert_eq!(parts[0].entry, *body);
    assert_eq!(
        c.cfg.blocks[c.cfg.label_to_block[body].0].params.len(),
        source.supplied_params(),
        "the body block takes the element and the counter, and no carried value \
         (RFC-0089 rule 1):\n{}",
        c.listing
    );
    let entries: Vec<Label> = parts.iter().map(|part| part.entry).collect();
    assert!(
        entries
            .iter()
            .all(|entry| c.cfg.label_to_block.contains_key(entry))
    );
}
