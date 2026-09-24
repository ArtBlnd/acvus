//! What `optimize::stages` leaves (RFC-0089): each loop's stages after the
//! full pipeline, in order, with each join's targets, order and law.

use acvus_mir::cfg::{CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{Law, LawOp, Order, Stage, Stages, Target, Targets};
use acvus_mir::printer::dump_with;
use acvus_mir_test::compile_script_at;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct Compiled {
    listing: String,
    cfg: CfgBody,
}

impl Compiled {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self {
            listing: dump_with(&interner, &compiled.module),
            cfg: promote(compiled.module.main),
        }
    }

    fn only_loop(&self) -> &Stages {
        let found: Vec<&Stages> = self
            .cfg
            .blocks
            .iter()
            .filter_map(|block| match &block.terminator {
                Terminator::For { stages, .. } => Some(stages),
                _ => None,
            })
            .collect();
        match found[..] {
            [stages] => stages,
            _ => panic!(
                "{} loops where one was written:\n{}",
                found.len(),
                self.listing
            ),
        }
    }

    fn for_line(&self) -> &str {
        self.listing
            .lines()
            .find(|line| line.contains(" for "))
            .unwrap_or_else(|| panic!("no `for` is printed:\n{}", self.listing))
    }

    fn shapes(&self) -> Vec<Shape> {
        self.only_loop().iter().map(Shape::of).collect()
    }
}

#[derive(Debug, PartialEq)]
enum Shape {
    Pure,
    Join {
        targets: Vec<TargetKind>,
        order: Order,
        law: Option<LawShape>,
    },
}

#[derive(Debug, PartialEq)]
enum TargetKind {
    Carried,
    Storage,
    Element,
}

#[derive(Debug, PartialEq)]
struct LawShape {
    law: LawKind,
    exact: bool,
}

#[derive(Debug, PartialEq)]
enum LawKind {
    Op(LawOp),
    Call,
    Fold,
}

impl Shape {
    fn of(stage: &Stage) -> Shape {
        let Stage::Join {
            targets,
            order,
            law,
            ..
        } = stage
        else {
            return Shape::Pure;
        };
        let Targets::Listed(targets) = targets else {
            panic!("the stage pass lists every join's targets: {stage:?}")
        };
        Shape::Join {
            targets: targets
                .iter()
                .map(|target| match target {
                    Target::Carried(_) => TargetKind::Carried,
                    Target::Storage(_) => TargetKind::Storage,
                    Target::Element => TargetKind::Element,
                    Target::Context(_) => panic!("no test here commits a context"),
                })
                .collect(),
            order: *order,
            law: law.as_ref().map(|acc| LawShape {
                law: match &acc.law {
                    Law::Op(op) => LawKind::Op(*op),
                    Law::Call(_) => LawKind::Call,
                    Law::Fold(_) => LawKind::Fold,
                    Law::Order => panic!("no test here runs in an `anyorder` region"),
                },
                exact: acc.exact,
            }),
        }
    }
}

fn join(targets: Vec<TargetKind>, order: Order, law: Option<LawShape>) -> Shape {
    Shape::Join {
        targets,
        order,
        law,
    }
}

fn exact(law: LawKind) -> Option<LawShape> {
    Some(LawShape { law, exact: true })
}

#[test]
fn a_range_sum_is_one_any_order_join_with_an_add_law() {
    let c = Compiled::of("let s = 0; for x in 0..10 { s = s + x; } s");
    assert_eq!(
        c.shapes(),
        [join(
            vec![TargetKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        )],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_slice_sum_reads_its_element_in_a_pure_stage_before_the_join() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; let s = 0; for x in &v { s = s + *x; } s");
    assert_eq!(
        c.shapes(),
        [
            Shape::Pure,
            join(
                vec![TargetKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            )
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_float_sum_is_in_order_with_its_inexact_law() {
    let c = Compiled::of("let v = [1.5, 2.5, 3.0]; let s = 0.0; for x in &v { s = s + *x; } s");
    assert_eq!(
        c.shapes(),
        [
            Shape::Pure,
            join(
                vec![TargetKind::Carried],
                Order::InOrder,
                Some(LawShape {
                    law: LawKind::Op(LawOp::Add),
                    exact: false
                })
            )
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_pure_call_is_a_pure_stage_before_the_push_it_feeds() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let out = vec::new(); \
         for x in &v { let y = min(*x, 3); vec::push(&mut out, y); } vec::len(&out)",
    );
    assert_eq!(
        c.shapes(),
        [
            Shape::Pure,
            join(
                vec![TargetKind::Storage],
                Order::InOrder,
                exact(LawKind::Fold)
            )
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_store_through_a_mutable_element_is_a_disjoint_join_after_its_pure_load() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; for x in &mut v { *x = min(*x, 2); } v[3]");
    assert_eq!(
        c.shapes(),
        [
            Shape::Pure,
            join(vec![TargetKind::Element], Order::Disjoint, None)
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_pop_a_pure_computation_and_a_push_on_one_vec_are_one_in_order_join() {
    let c = Compiled::of(
        "let q = vec::new(); vec::push(&mut q, 5); \
         for i in 0..3 { let t = unwrap_or(vec::pop(&mut q), 0); let y = min(t * 2, 50); \
         vec::push(&mut q, y); } vec::len(&q)",
    );
    let joins: Vec<Shape> = c
        .shapes()
        .into_iter()
        .filter(|shape| *shape != Shape::Pure)
        .collect();
    assert_eq!(
        joins,
        [join(vec![TargetKind::Storage], Order::InOrder, None)],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_recurrence_with_no_law_is_one_in_order_join() {
    let c =
        Compiled::of("let v = [1, 2, 3]; let acc = 1; for x in &v { acc = acc * acc + *x; } acc");
    assert_eq!(
        c.shapes(),
        [
            Shape::Pure,
            join(vec![TargetKind::Carried], Order::InOrder, None)
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn three_independent_accumulators_are_three_joins() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let a = 0; let b = 1; let c = 100; \
         for x in &v { a = a + *x; b = b * *x; c = min(c, *x); } a + b + c",
    );
    let joins: Vec<Shape> = c
        .shapes()
        .into_iter()
        .filter(|shape| *shape != Shape::Pure)
        .collect();
    assert_eq!(
        joins,
        [
            join(
                vec![TargetKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
            join(
                vec![TargetKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Mul))
            ),
            join(
                vec![TargetKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Call)
            ),
        ],
        "{}",
        c.for_line()
    );
}

/// RFC-0089 rule 7: the `break` leaves from the body, so the pure
/// computation it tests sits in the one `InOrder` join the loop is.
#[test]
fn a_break_in_a_pure_computation_sits_in_an_in_order_join() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; \
         for x in &v { let y = *x * 3; if y > 6 { break; } s = s + y; } s",
    );
    assert_eq!(
        c.shapes(),
        [join(vec![TargetKind::Carried], Order::InOrder, None)],
        "{}",
        c.for_line()
    );
}

/// RFC-0089 rule 4: the body needs the id `pop` hands it before it can
/// compute, so the join over `ids` is a producer ahead of the pure stage
/// that reads it.
#[test]
fn a_counter_the_body_reads_is_a_producer_join_first() {
    let c = Compiled::of(
        "let ids = vec::new(); vec::push(&mut ids, 7); vec::push(&mut ids, 8); \
         let v = [1, 2]; let s = 0; \
         for x in &v { let id = unwrap_or(vec::pop(&mut ids), 0); s = s + id * *x; } s",
    );
    assert_eq!(
        c.shapes(),
        [
            join(vec![TargetKind::Storage], Order::InOrder, None),
            Shape::Pure,
            join(
                vec![TargetKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
        ],
        "{}",
        c.for_line()
    );
}

#[test]
fn a_while_is_not_rewritten() {
    let c = Compiled::of("let i = 0; let s = 0; while i < 10 { s = s + i; i = i + 2; } s");
    assert!(!c.listing.contains(" stages ["), "{}", c.listing);
}

#[test]
fn every_stage_entry_is_a_block_and_the_first_is_the_body() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { s = s + *x; p = p * *x; } s + p",
    );
    let (source, stages) = c
        .cfg
        .blocks
        .iter()
        .find_map(|block| match &block.terminator {
            Terminator::For { source, stages, .. } => Some((source, stages)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("one `for`:\n{}", c.listing));
    assert!(stages.len() > 1, "{}", c.listing);
    assert_eq!(
        c.cfg.blocks[c.cfg.label_to_block[&stages.body()].0]
            .params
            .len(),
        source.supplied_params(),
        "the body block takes the element and the counter, and no carried value \
         (RFC-0089 rule 1):\n{}",
        c.listing
    );
    assert!(
        stages
            .entries()
            .all(|entry| c.cfg.label_to_block.contains_key(&entry))
    );
}
