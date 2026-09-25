//! Where `optimize::stages` cuts each loop after the full pipeline, and what
//! `analysis::loop_deps` computes of each stage (RFC-0089): free, or its
//! cycles, each with its tokens, order and law; and the loop's control.

use acvus_mir::analysis::affine::{AffineValues, Derivation};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loans::Loans;
use acvus_mir::analysis::loop_deps::{
    Control, Law, LawOp, LoopDeps, Member, Order, StageMembership, Storage, Token,
};
use acvus_mir::analysis::loops::{Invariants, LoopNest};
use acvus_mir::analysis::targets::{effect, slots_lent_mutably};
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ir::{BinOp, InstKind, Overflow, ValueId};
use acvus_mir::laws::LawTable;
use acvus_mir::printer::dump_with_facts;
use acvus_mir::ty::{Effect, ParamTerm, Poly, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::{LoweredScript, compile_script_at, optimized_script};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct Compiled {
    listing: String,
    cfg: CfgBody,
    laws: LawTable,
}

impl Compiled {
    fn of(source: &str) -> Self {
        Self::reading(source, &[])
    }

    /// `source` reading each of `contexts` as an `i64` context.
    fn reading(source: &str, contexts: &[&str]) -> Self {
        let interner = Interner::new();
        let context: FxHashMap<_, _> = contexts
            .iter()
            .map(|name| (interner.intern(name), Ty::I64))
            .collect();
        let compiled = compile_script_at(&interner, source, &context, Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from(&interner, compiled)
    }

    fn with_externs(source: &str, externs: impl FnOnce(&Interner) -> Vec<Function>) -> Self {
        let interner = Interner::new();
        let compiled = optimized_script(&interner, source, &externs(&interner), vec![])
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from(&interner, compiled)
    }

    /// `source` with `io` beside the standard registries.
    fn with_io(source: &str) -> Self {
        let interner = Interner::new();
        let own = vec![acvus_ext::io_registry::<acvus_extern::TypesOnly>()];
        let compiled = optimized_script(&interner, source, &[], own)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from(&interner, compiled)
    }

    fn from(interner: &Interner, compiled: LoweredScript) -> Self {
        Self {
            listing: dump_with_facts(interner, &compiled.module, &compiled.laws),
            cfg: promote(compiled.module.main.clone()),
            laws: compiled.laws,
        }
    }

    fn only_header(&self) -> BlockIdx {
        let found: Vec<BlockIdx> = (0..self.cfg.blocks.len())
            .map(BlockIdx)
            .filter(|at| matches!(self.cfg.blocks[at.0].terminator, Terminator::For { .. }))
            .collect();
        match found[..] {
            [header] => header,
            _ => panic!("one `for`:\n{}", self.listing),
        }
    }

    fn deps(&self) -> LoopDeps {
        LoopDeps::of(&self.cfg, &self.laws, self.only_header())
            .unwrap_or_else(|fault| panic!("{}:\n{}", fault.shown(), self.listing))
    }

    fn membership(&self) -> StageMembership {
        self.deps().membership
    }

    /// The header parameters `analysis::affine` derives as carried
    /// induction variables.
    fn carried_ivs(&self) -> Vec<ValueId> {
        let header = self.only_header();
        let invariants = Invariants::of(&self.cfg);
        let nest = LoopNest::of(&self.cfg, &DomTree::build(&self.cfg), &invariants);
        let loop_ = nest.get(nest.by_header(header).expect("the header heads a loop"));
        let affine = AffineValues::of(&self.cfg, loop_, &invariants, &self.laws);
        self.cfg.blocks[header.0]
            .params
            .iter()
            .copied()
            .filter(|param| {
                matches!(
                    affine.get(*param).map(|found| &found.derivation),
                    Some(Derivation::Carried { .. })
                )
            })
            .collect()
    }

    fn stage_blocks(&self) -> Vec<Vec<BlockIdx>> {
        self.membership()
            .stages()
            .iter()
            .map(|stage| stage.blocks.clone())
            .collect()
    }

    fn stages_where(&self, holds: impl Fn(&CfgBody, BlockIdx) -> bool) -> Vec<usize> {
        let membership = self.membership();
        let mut stages: Vec<usize> = (0..self.cfg.blocks.len())
            .map(BlockIdx)
            .filter(|block| holds(&self.cfg, *block))
            .filter_map(|block| membership.stage_of(block))
            .collect();
        stages.sort_unstable();
        stages.dedup();
        stages
    }

    /// The stages whose instructions multiply by the word `factor`.
    fn stages_multiplying_by(&self, factor: i128) -> Vec<usize> {
        let words: Vec<ValueId> = self
            .cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.kind {
                InstKind::Const {
                    dst,
                    value: acvus_ast::Literal::Int(word),
                } if *word == factor => Some(*dst),
                _ => None,
            })
            .collect();
        self.stage_blocks()
            .iter()
            .enumerate()
            .filter(|(_, blocks)| {
                blocks.iter().any(|block| {
                    self.cfg.blocks[block.0].insts.iter().any(|inst| {
                        matches!(&inst.kind, InstKind::BinOp { op: BinOp::Mul(_), left, right, .. }
                            if words.contains(left) || words.contains(right))
                    })
                })
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// The `for` line and the facts printed under it.
    fn for_lines(&self) -> String {
        let mut lines = self
            .listing
            .lines()
            .skip_while(|line| !line.contains(" for "));
        let head = lines
            .next()
            .unwrap_or_else(|| panic!("no `for` is printed:\n{}", self.listing));
        std::iter::once(head)
            .chain(lines.take_while(|line| line.contains("// ")))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Each stage as `loop_deps` computes it.
    fn shapes(&self) -> Vec<Shape> {
        let deps = self.deps();
        let judged = deps.judge(&self.cfg, &self.laws);
        assert_eq!(
            deps.crossing().count(),
            0,
            "no cycle crosses a boundary:\n{}",
            self.listing
        );
        (0..deps.membership.stages().len())
            .map(|stage| {
                let cycles: Vec<CycleShape> = deps
                    .cycles
                    .iter()
                    .zip(&judged)
                    .filter(|(cycle, _)| cycle.stage() == Some(stage))
                    .map(|(cycle, judged)| CycleShape {
                        tokens: cycle.tokens.iter().map(TokenKind::of).collect(),
                        order: judged.order,
                        law: judged.law.as_ref().map(|acc| LawShape {
                            law: match &acc.law {
                                Law::Op(op) => LawKind::Op(*op),
                                Law::Call(_) => LawKind::Call,
                                Law::Fold(_) => LawKind::Fold,
                                Law::Order => LawKind::Order,
                            },
                            exact: acc.exact,
                        }),
                    })
                    .collect();
                match cycles.is_empty() {
                    true => Shape::Free,
                    false => Shape::Cycles(cycles),
                }
            })
            .collect()
    }

    /// The instruction kinds a stage's cycles hold, by name.
    fn held_by_cycles_of(&self, stage: usize) -> Vec<&'static str> {
        let deps = self.deps();
        let mut held: Vec<&'static str> = deps
            .cycles_in(stage)
            .flat_map(|cycle| &cycle.members)
            .filter_map(|member| match *member {
                Member::Inst(at) => Some(kind_name(&self.cfg.blocks[at.block.0].insts[at.at].kind)),
                Member::Term(_) => None,
            })
            .collect();
        held.sort_unstable();
        held
    }
}

fn kind_name(kind: &InstKind) -> &'static str {
    match kind {
        InstKind::Spawn { .. } => "spawn",
        InstKind::Eval { .. } => "eval",
        InstKind::Merge { .. } => "merge",
        InstKind::FunctionCall { .. } => "call",
        _ => "other",
    }
}

#[derive(Debug, PartialEq)]
struct FreeEffect {
    kind: &'static str,
    control: Control,
}

#[derive(Debug, PartialEq)]
enum Shape {
    Free,
    Cycles(Vec<CycleShape>),
}

#[derive(Debug, PartialEq)]
struct CycleShape {
    tokens: Vec<TokenKind>,
    order: Order,
    law: Option<LawShape>,
}

#[derive(Debug, PartialEq)]
enum TokenKind {
    Order,
    Carried,
    Storage,
    Element,
    Context,
    Control,
}

impl TokenKind {
    fn of(token: &Token) -> TokenKind {
        match token {
            Token::Order(_) => TokenKind::Order,
            Token::Carried(_) => TokenKind::Carried,
            Token::Storage(Storage::Slot(_)) => TokenKind::Storage,
            Token::Storage(Storage::Element) => TokenKind::Element,
            Token::Storage(Storage::Context(_)) => TokenKind::Context,
            Token::Control => TokenKind::Control,
        }
    }
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
    Order,
}

fn cycle(tokens: Vec<TokenKind>, order: Order, law: Option<LawShape>) -> CycleShape {
    CycleShape { tokens, order, law }
}

fn one(tokens: Vec<TokenKind>, order: Order, law: Option<LawShape>) -> Shape {
    Shape::Cycles(vec![cycle(tokens, order, law)])
}

fn exact(law: LawKind) -> Option<LawShape> {
    Some(LawShape { law, exact: true })
}

fn cycle_stages(shapes: &[Shape]) -> Vec<&Shape> {
    shapes
        .iter()
        .filter(|shape| **shape != Shape::Free)
        .collect()
}

#[test]
fn a_range_sum_is_one_any_order_cycle_with_an_add_law() {
    let c = Compiled::of("let s = 0; for x in 0..10 { s = s + x; } s");
    assert_eq!(
        c.shapes(),
        [one(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        )],
        "{}",
        c.for_lines()
    );
}

/// Table row `s = s + x` over `i64`: `[free, cycle s any_order Op(+)]`.
#[test]
fn a_slice_sum_reads_its_element_in_a_free_stage_before_the_cycle() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; let s = 0; for x in &v { s = s + *x; } s");
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            )
        ],
        "{}",
        c.for_lines()
    );
}

/// Table row `s = s + x` over `f64`: the cycle is `in_order`.
#[test]
fn a_float_sum_is_in_order_with_its_inexact_law() {
    let c = Compiled::of("let v = [1.5, 2.5, 3.0]; let s = 0.0; for x in &v { s = s + *x; } s");
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(
                vec![TokenKind::Carried],
                Order::InOrder,
                Some(LawShape {
                    law: LawKind::Op(LawOp::Add),
                    exact: false
                })
            )
        ],
        "{}",
        c.for_lines()
    );
}

/// Table row `out.push(f(x))`: `[free(f), cycle Storage(out) in_order]`.
#[test]
fn a_pure_call_is_a_free_stage_before_the_push_it_feeds() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let out = vec::new(); \
         for x in &v { let y = min(*x, 3); vec::push(&mut out, y); } vec::len(&out)",
    );
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(
                vec![TokenKind::Storage],
                Order::InOrder,
                exact(LawKind::Fold)
            )
        ],
        "{}",
        c.for_lines()
    );
}

/// Table row `for x in &mut xs { *x = g(*x) }`:
/// `[free(load, g), cycle Storage(element) disjoint]`.
#[test]
fn a_store_through_a_mutable_element_is_a_disjoint_cycle_after_its_free_load() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; for x in &mut v { *x = min(*x, 2); } v[3]");
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(vec![TokenKind::Element], Order::Disjoint, None)
        ],
        "{}",
        c.for_lines()
    );
}

/// Table row `pop → pure → push` on one vec: one cycle, `in_order`.
#[test]
fn a_pop_a_pure_computation_and_a_push_on_one_vec_are_one_in_order_cycle() {
    let c = Compiled::of(
        "let q = vec::new(); vec::push(&mut q, 5); \
         for i in 0..3 { let t = unwrap_or(vec::pop(&mut q), 0); let y = min(t * 2, 50); \
         vec::push(&mut q, y); } vec::len(&q)",
    );
    assert_eq!(
        cycle_stages(&c.shapes()),
        [&one(vec![TokenKind::Storage], Order::InOrder, None)],
        "{}",
        c.for_lines()
    );
}

#[test]
fn a_recurrence_with_no_law_is_one_in_order_cycle() {
    let c =
        Compiled::of("let v = [1, 2, 3]; let acc = 1; for x in &v { acc = acc * acc + *x; } acc");
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(vec![TokenKind::Carried], Order::InOrder, None)
        ],
        "{}",
        c.for_lines()
    );
}

#[test]
fn three_independent_accumulators_are_three_cycles_in_three_stages() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let a = 0; let b = 1; let c = 100; \
         for x in &v { a = a + *x; b = b * *x; c = min(c, *x); } a + b + c",
    );
    assert_eq!(
        cycle_stages(&c.shapes()),
        [
            &one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
            &one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Mul))
            ),
            &one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Call)
            ),
        ],
        "{}",
        c.for_lines()
    );
}

/// Table row, a loop with `break`: its control is chained. The `break`
/// leaves from the body, so the loop is one stage, whose cycles are one
/// `InOrder` join with the exit (RFC-0089 rule 5, RFC-0066 rule 10).
#[test]
fn a_break_chains_the_control_token_through_its_one_in_order_stage() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; \
         for x in &v { let y = *x * 3; if y > 6 { break; } s = s + y; } s",
    );
    assert_eq!(
        c.shapes(),
        [one(
            vec![TokenKind::Carried, TokenKind::Control],
            Order::InOrder,
            None
        )],
        "{}",
        c.for_lines()
    );
    assert!(
        matches!(c.deps().control, Control::Chained { .. }),
        "{}",
        c.for_lines()
    );
}

/// RFC-0089 rule 6: the body needs the id `pop` hands it before it can
/// compute, so the cycle over `ids` is a producer ahead of the free stage
/// that reads it.
#[test]
fn a_counter_the_body_reads_is_a_producer_cycle_first() {
    let c = Compiled::of(
        "let ids = vec::new(); vec::push(&mut ids, 7); vec::push(&mut ids, 8); \
         let v = [1, 2]; let s = 0; \
         for x in &v { let id = unwrap_or(vec::pop(&mut ids), 0); s = s + id * *x; } s",
    );
    assert_eq!(
        c.shapes(),
        [
            one(vec![TokenKind::Storage], Order::InOrder, None),
            Shape::Free,
            one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
        ],
        "{}",
        c.for_lines()
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

// -- Induction variables and strength reduction (RFC-0066 rule 7) -------

/// `i` is read by `*x * i`, a pure computation, so IV canonicalization
/// computes it from the counter: the header carries `s` alone, and no
/// cycle is over `i`. The `Check` that keeps `i`'s step's trap reads the
/// counter alone, so it is free work, cut after `s`'s cycle.
#[test]
fn a_counter_a_pure_computation_reads_is_not_carried_and_has_no_cycle() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let i = 0; let s = 0; \
         for x in &v { let y = *x * i; s = s + y; i = i + 1; } s + i",
    );
    assert_eq!(c.carried_ivs(), [], "{}", c.listing);
    assert_eq!(
        c.cfg.blocks[c.only_header().0].params.len(),
        1,
        "the header carries `s` alone:\n{}",
        c.listing
    );
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
            Shape::Free
        ],
        "{}",
        c.for_lines()
    );
}

/// `i * 4 + 7` is read only by `acc`'s update, whose stage is `InOrder`, so
/// strength reduction carries it as a counter of its own, whose cycle lies
/// in that stage beside `acc`'s (RFC-0056).
#[test]
fn a_counter_expression_an_in_order_stage_alone_reads_is_reduced_inside_that_stage() {
    let c = Compiled::of(
        "let i = 0; let acc = 1; \
         for k in 0..10 { acc = acc * acc + (i * 4 + 7); i = i + 1; } acc",
    );
    let shapes = c.shapes();
    let Some(Shape::Cycles(last)) = shapes.last() else {
        panic!("the chain ends in `acc`'s stage: {}", c.for_lines());
    };
    assert_eq!(
        last[..],
        [
            cycle(vec![TokenKind::Carried], Order::InOrder, None),
            cycle(vec![TokenKind::Carried], Order::InOrder, None)
        ],
        "the stage holds `acc`'s cycle and the reduced counter's:\n{}",
        c.listing
    );
    assert!(
        shapes[..shapes.len() - 1]
            .iter()
            .all(|shape| *shape == Shape::Free),
        "{}",
        c.for_lines()
    );
    assert_eq!(
        c.stages_multiplying_by(4),
        Vec::<usize>::new(),
        "nothing in the body multiplies by 4:\n{}",
        c.listing
    );
    let [derived] = c.carried_ivs()[..] else {
        panic!(
            "one carried induction variable, the reduced counter:\n{}",
            c.listing
        );
    };
    let stage = c.stage_blocks().len() - 1;
    let advanced_in_stage = c.stage_blocks()[stage].iter().any(|block| {
        c.cfg.blocks[block.0].insts.iter().any(|inst| {
            matches!(&inst.kind, InstKind::BinOp { op: BinOp::Add(Overflow::Wrap), left, right, .. }
                if *left == derived || *right == derived)
        })
    });
    assert!(
        advanced_in_stage,
        "its step is in the stage:\n{}",
        c.listing
    );
}

/// The same expression read by `min`, a pure call, stays `i * 4 + 7`
/// computed from the canonical `i` in a free stage.
#[test]
fn a_counter_expression_a_free_stage_reads_is_computed_from_the_counter() {
    let c = Compiled::of(
        "let i = 0; let acc = 1; \
         for k in 0..10 { acc = acc * acc + min(i * 4 + 7, 50); i = i + 1; } acc",
    );
    assert_eq!(c.carried_ivs(), [], "{}", c.listing);
    let shapes = c.shapes();
    let multiplies = c.stages_multiplying_by(4);
    let [stage] = multiplies[..] else {
        panic!("one stage multiplies by 4:\n{}", c.listing);
    };
    assert_eq!(shapes[stage], Shape::Free, "{}", c.listing);
    assert_eq!(
        shapes.last(),
        Some(&one(vec![TokenKind::Carried], Order::InOrder, None)),
        "{}",
        c.for_lines()
    );
}

// -- What orders a loop (RFC-0089 rules 2, 4 and 5) -----------------------

fn emit(i: &Interner) -> Vec<Function> {
    vec![Function {
        qref: QualifiedRef::root(i.intern("emit")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }]
}

/// An opaque call carries an order, so it sits in the `InOrder` cycle of
/// the loop's `Order` and in no free stage.
#[test]
fn an_ordered_effect_sits_in_the_in_order_cycle_of_the_order() {
    let c = Compiled::with_externs("let n = 3; for k in 0..n { emit(k); } 0", emit);
    let calls = c.stages_where(|cfg, block| {
        cfg.blocks[block.0].insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::FunctionCall { .. } | InstKind::Spawn { .. }
            )
        })
    });
    let [stage] = calls[..] else {
        panic!("one stage calls `emit`:\n{}", c.listing);
    };
    assert_eq!(
        c.shapes()[stage],
        one(vec![TokenKind::Order], Order::InOrder, None),
        "{}",
        c.for_lines()
    );
}

/// Table row, `io::print` in a loop without `anyorder`: the cycle of the
/// `Order` is `in_order` and holds the spawn and the eval.
#[test]
fn a_print_outside_anyorder_is_an_in_order_cycle_holding_its_spawn_and_eval() {
    let c = Compiled::with_io("let v = [1, 2, 3]; for x in &v { io::print(\"a\"); } 0");
    let shapes = c.shapes();
    let ordered: Vec<usize> = (0..shapes.len())
        .filter(|stage| shapes[*stage] != Shape::Free)
        .collect();
    let [stage] = ordered[..] else {
        panic!("one stage holds a cycle:\n{}", c.for_lines());
    };
    assert_eq!(
        shapes[stage],
        one(vec![TokenKind::Order], Order::InOrder, None),
        "{}",
        c.for_lines()
    );
    let held = c.held_by_cycles_of(stage);
    assert!(
        held.contains(&"spawn") && held.contains(&"eval"),
        "{held:?}\n{}",
        c.for_lines()
    );
}

/// Table row, `anyorder { … io::print … }`: the spawn, which takes the
/// block's entry `Order`, and the eval, whose `Order` the merge reads, sit
/// in a free stage before the cycle (rule 6), and the cycle of the loop's
/// `Order` holds the merge alone, `any_order` by the law of `merge`. Rule 5
/// lets both run once their iteration holds the control token, which it
/// holds from the start.
#[test]
fn an_anyorder_print_spawns_and_evaluates_in_a_free_stage_and_joins_by_merge() {
    let c =
        Compiled::with_io("let v = [1, 2, 3]; anyorder { for x in &v { io::print(\"a\"); } } 0");
    let shapes = c.shapes();
    let Some(Shape::Cycles(last)) = shapes.last() else {
        panic!("the chain ends in the `Order`'s stage: {}", c.for_lines());
    };
    assert_eq!(
        last[..],
        [cycle(
            vec![TokenKind::Order],
            Order::AnyOrder,
            exact(LawKind::Order)
        )],
        "{}",
        c.for_lines()
    );
    assert!(
        shapes[..shapes.len() - 1]
            .iter()
            .all(|shape| *shape == Shape::Free),
        "{}",
        c.for_lines()
    );
    assert_eq!(
        c.held_by_cycles_of(shapes.len() - 1),
        ["merge"],
        "{}",
        c.for_lines()
    );
    let deps = c.deps();
    let mut effects: Vec<FreeEffect> = deps
        .effects_in_free_stages()
        .iter()
        .filter_map(|wait| match wait.member {
            Member::Inst(at) => Some(FreeEffect {
                kind: kind_name(&c.cfg.blocks[at.block.0].insts[at.at].kind),
                control: wait.control,
            }),
            Member::Term(_) => None,
        })
        .collect();
    effects.sort_unstable_by_key(|effect| effect.kind);
    assert_eq!(
        effects,
        [
            FreeEffect {
                kind: "eval",
                control: Control::Upfront
            },
            FreeEffect {
                kind: "spawn",
                control: Control::Upfront
            }
        ],
        "{}",
        c.for_lines()
    );
}

/// The same print in a loop that can `break`: the loop leaves from its
/// body, so the control token is chained, and the spawn and the eval lie in
/// the stage the loop leaves from, the join of that token (rule 5). No
/// effect sits in a free stage.
#[test]
fn an_anyorder_print_in_a_loop_that_breaks_keeps_its_effects_in_the_exiting_stage() {
    let c = Compiled::with_io(
        "let v = [1, 2, 3]; anyorder { for x in &v { if *x == 2 { break; }; io::print(\"a\"); } } 0",
    );
    let deps = c.deps();
    assert!(
        matches!(deps.control, Control::Chained { .. }),
        "{}",
        c.for_lines()
    );
    assert_eq!(deps.effects_in_free_stages(), [], "{}", c.for_lines());
    let Control::Chained { cycle } = deps.control else {
        panic!("{}", c.for_lines())
    };
    let exiting = deps.cycles[cycle]
        .stage()
        .expect("no cycle crosses a boundary");
    let effects = c.stages_where(|cfg, block| {
        cfg.blocks[block.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Spawn { .. } | InstKind::Eval { .. }))
    });
    assert_eq!(effects, [exiting], "{}", c.for_lines());
}

/// A write through the element is the element's `Disjoint` cycle; one
/// through another `&mut` is a storage's, `InOrder`.
#[test]
fn a_write_elsewhere_than_the_element_is_an_in_order_cycle_over_its_storage() {
    let c = Compiled::of(
        "let v = vec([1, 2, 3]); let t = 0; let r = &mut t; for x in &mut v { *r = *x; } t",
    );
    assert_eq!(
        cycle_stages(&c.shapes()),
        [&one(vec![TokenKind::Storage], Order::InOrder, None)],
        "{}",
        c.for_lines()
    );
}

// -- One membership, the readers of each cycle (RFC-0089 rule 1) ---------

const BREAK: &str = "let j = 0; let s = 0; \
     for i in 0..@n { if i == 4 { break; }; s = s + j; j = j + 5; } s * 1000 + j";

/// A loop that leaves early is one stage and keeps its carried induction
/// variable, whose readers lie in the stage its cycle lies in.
#[test]
fn a_carried_iv_is_read_in_the_stage_its_cycle_lies_in() {
    let c = Compiled::reading(BREAK, &["n"]);
    let [iv] = c.carried_ivs()[..] else {
        panic!("one carried induction variable:\n{}", c.listing);
    };
    let reading = c.stages_where(|cfg, block| {
        let block = &cfg.blocks[block.0];
        block
            .insts
            .iter()
            .any(|inst| acvus_mir::analysis::inst_info::uses(&inst.kind).contains(&iv))
            || acvus_mir::analysis::inst_info::terminator_uses(&block.terminator).contains(&iv)
    });
    let deps = c.deps();
    let cycle = deps
        .cycles
        .iter()
        .find(|cycle| cycle.tokens.contains(&Token::Carried(iv)))
        .unwrap_or_else(|| panic!("{iv:?} has a cycle:\n{}", c.for_lines()));
    assert_eq!(
        cycle.stage().into_iter().collect::<Vec<_>>(),
        reading,
        "{}",
        c.for_lines()
    );
    assert!(deps.early_reads().is_empty(), "{}", c.for_lines());
}

/// `lsr` advances the reduced counter at the end of the `InOrder` stage
/// that reads it, which is the block membership ends that stage with.
#[test]
fn a_reduced_counter_advances_at_the_end_membership_gives_its_stage() {
    let c = Compiled::of(
        "let i = 0; let acc = 1; for k in 0..10 { acc = acc * acc + (i * 4 + 7); i = i + 1; } acc",
    );
    let [derived] = c.carried_ivs()[..] else {
        panic!(
            "one carried induction variable, the reduced counter:\n{}",
            c.listing
        );
    };
    let advances = |cfg: &CfgBody, block: BlockIdx| {
        cfg.blocks[block.0].insts.iter().any(|inst| {
            matches!(&inst.kind, InstKind::BinOp { op: BinOp::Add(Overflow::Wrap), left, .. } if *left == derived)
        })
    };
    let stage = c.shapes().len() - 1;
    let Shape::Cycles(held) = &c.shapes()[stage] else {
        panic!("the last stage holds cycles: {}", c.for_lines());
    };
    assert!(
        held.iter()
            .all(|cycle| cycle.order == Order::InOrder && cycle.law.is_none()),
        "{}",
        c.for_lines()
    );
    assert_eq!(c.stages_where(advances), [stage], "{}", c.listing);
    let end = c.membership().stages()[stage]
        .sole_end()
        .unwrap_or_else(|| panic!("the stage ends in one block:\n{}", c.listing));
    assert!(advances(&c.cfg, end), "{}", c.listing);
}

/// A carried value no stage reads dies on the body edge, and
/// `drop_insertion` releases it at the entry of the stage its cycle lies
/// in, the entry membership gives that stage.
#[test]
fn an_overwritten_carried_value_is_dropped_at_the_entry_of_its_cycle_s_stage() {
    let c = Compiled::reading(
        "let last = \"\".to_string(); for x in 0..@n { last = x.to_string(); } last",
        &["n"],
    );
    let header = c.only_header();
    let deps = c.deps();
    let carried: Vec<(usize, ValueId)> = deps
        .cycles
        .iter()
        .flat_map(|cycle| {
            cycle.tokens.iter().filter_map(|token| match token {
                Token::Carried(param) => cycle.stage().map(|stage| (stage, *param)),
                _ => None,
            })
        })
        .collect();
    let [(stage, last)] = carried[..] else {
        panic!("one cycle carries one value:\n{}", c.for_lines());
    };
    assert!(
        c.cfg.blocks[header.0].params.contains(&last),
        "{}",
        c.listing
    );
    let dropped_at: Vec<BlockIdx> = (0..c.cfg.blocks.len())
        .map(BlockIdx)
        .filter(|block| {
            c.cfg.blocks[block.0]
                .insts
                .iter()
                .any(|inst| matches!(&inst.kind, InstKind::Drop { src } if *src == last))
        })
        .collect();
    assert_eq!(
        dropped_at,
        [c.membership().stages()[stage].blocks[0]],
        "{}",
        c.listing
    );
}

/// The `AnyOrder` cycle's update lies in the stage membership puts it in.
#[test]
fn an_any_order_cycle_s_update_lies_where_membership_puts_it() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; let s = 0; for x in &v { s = s + *x; } s");
    let header = c.only_header();
    let [s] = c.cfg.blocks[header.0].params[..] else {
        panic!("the header carries `s` alone:\n{}", c.listing);
    };
    let updating = c.stages_where(|cfg, block| {
        cfg.blocks[block.0].insts.iter().any(|inst| {
            matches!(&inst.kind, InstKind::BinOp { op: BinOp::Add(Overflow::Trap), left, right, .. }
                if *left == s || *right == s)
        })
    });
    assert_eq!(updating, [c.shapes().len() - 1], "{}", c.listing);
    assert_eq!(
        c.shapes()[updating[0]],
        one(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        ),
        "{}",
        c.for_lines()
    );
}

/// The `Disjoint` cycle's store through the element lies in the stage
/// membership puts it in.
#[test]
fn a_disjoint_cycle_s_store_lies_where_membership_puts_it() {
    let c = Compiled::of("let v = [1, 2, 3, 4]; for x in &mut v { *x = min(*x, 2); } v[3]");
    let loans = Loans::build(&c.cfg);
    let writing = c.stages_where(|cfg, block| {
        cfg.blocks[block.0].insts.iter().any(|inst| {
            !effect(&loans, &inst.kind).writes.is_empty()
                || !slots_lent_mutably(&loans, &inst.kind).is_empty()
        })
    });
    assert_eq!(writing, [c.shapes().len() - 1], "{}", c.listing);
    assert_eq!(
        c.shapes()[writing[0]],
        one(vec![TokenKind::Element], Order::Disjoint, None),
        "{}",
        c.for_lines()
    );
}

/// Table row, the DSE property: `f`'s only effectful call sits in an arm
/// the fold removes once `f` is inlined, so the loop's `Order` is handed on
/// unchanged, no token is left, and the loop is one free stage. The
/// terminator names no token, so nothing keeps the `Order`'s phi alive.
#[test]
fn a_loop_whose_only_effect_folds_away_after_inlining_is_one_free_stage() {
    let i = Interner::new();
    let listing = acvus_mir_test::compile_multi_fn_optimized_with_facts(
        &i,
        ("main", "for x in 1..@n { f(x); } 0"),
        &[(
            "f",
            "if false { emit($x); }; 10 / $x",
            vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
        )],
        &[("n", Ty::I64)],
        &emit(&i),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut lines = listing.lines().skip_while(|line| !line.contains(" for "));
    let head = lines
        .next()
        .unwrap_or_else(|| panic!("the loop stands:\n{listing}"));
    let facts: Vec<&str> = lines
        .take_while(|line| line.contains("// "))
        .map(|line| {
            line.split_once("// ")
                .expect("`take_while` kept only the lines holding the marker")
                .1
        })
        .collect();
    assert!(head.contains("stages [L"), "{listing}");
    assert_eq!(facts.len(), 2, "one stage and the control:\n{listing}");
    assert!(facts[0].contains(": free {"), "{listing}");
    assert_eq!(facts[1], "control upfront", "{listing}");
    assert!(!listing.contains("spawn"), "{listing}");
}

// -- A law read from what a cycle computes (RFC-0089 rule 4) ------------

/// The one cycle of `source`'s loop, as `loop_deps` judges it, and the
/// loop's printed facts.
fn the_cycle(source: &str) -> (CycleShape, String) {
    let c = Compiled::of(source);
    let lines = c.for_lines();
    let mut cycles: Vec<CycleShape> = c
        .shapes()
        .into_iter()
        .flat_map(|shape| match shape {
            Shape::Free => Vec::new(),
            Shape::Cycles(cycles) => cycles,
        })
        .collect();
    let (Some(held), None) = (cycles.pop(), cycles.pop()) else {
        panic!("the loop holds one cycle:\n{lines}")
    };
    (held, lines)
}

#[test]
fn a_sum_under_a_branch_that_reads_no_state_is_an_add_law() {
    let (held, lines) =
        the_cycle("let v = [5, 3, 8]; let c = 0; for x in &v { if *x > 4 { c = c + *x; }; } c");
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        ),
        "{lines}"
    );
}

#[test]
fn a_sum_under_a_branch_that_reads_the_state_has_no_law() {
    let (held, lines) =
        the_cycle("let v = [5, 3, 8]; let c = 0; for x in &v { if c < 7 { c = c + *x; }; } c");
    assert_eq!(
        held,
        cycle(vec![TokenKind::Carried], Order::InOrder, None),
        "{lines}"
    );
}

#[test]
fn a_compare_and_select_of_the_compared_value_is_a_min_law() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 8]; let m = i64::MAX(); for x in &v { if *x < m { m = *x; }; } m",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Min))
        ),
        "{lines}"
    );
}

#[test]
fn a_compare_and_select_of_the_compared_value_on_the_else_side_is_a_max_law() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 8]; let m = i64::MIN(); for x in &v { m = if *x < m { m } else { *x }; } m",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Max))
        ),
        "{lines}"
    );
}

#[test]
fn a_select_of_a_value_other_than_the_compared_one_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 8]; let m = i64::MAX(); for x in &v { if *x < m { m = *x + 1; }; } m",
    );
    assert_eq!(
        held,
        cycle(vec![TokenKind::Carried], Order::InOrder, None),
        "{lines}"
    );
}

#[test]
fn a_difference_from_the_state_is_an_add_law_and_one_to_the_state_has_none() {
    let (held, lines) = the_cycle("let v = [5, 3, 8]; let s = 0; for x in &v { s = s - *x; } s");
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        ),
        "{lines}"
    );
    let (held, lines) = the_cycle("let v = [5, 3, 8]; let s = 0; for x in &v { s = *x - s; } s");
    assert_eq!(
        held,
        cycle(vec![TokenKind::Carried], Order::InOrder, None),
        "{lines}"
    );
}

#[test]
fn a_float_difference_keeps_its_order_with_an_inexact_law() {
    let (held, lines) = the_cycle("let v = [1.5, 2.5]; let s = 0.0; for x in &v { s = s - *x; } s");
    let inexact = Some(LawShape {
        law: LawKind::Op(LawOp::Add),
        exact: false,
    });
    assert_eq!(
        held,
        cycle(vec![TokenKind::Carried], Order::InOrder, inexact),
        "{lines}"
    );
}

#[test]
fn an_appended_text_is_an_in_order_concat_law_and_a_prepended_one_has_none() {
    let (held, lines) = the_cycle(
        "let v = vec([\"a\".to_string(), \"b\".to_string()]); let s = \"\".to_string(); \
         for x in &v { s = s + x; } s.len()",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::InOrder,
            exact(LawKind::Op(LawOp::Concat))
        ),
        "{lines}"
    );
    let (held, lines) = the_cycle(
        "let v = vec([\"a\".to_string(), \"b\".to_string()]); let s = \"\".to_string(); \
         for x in &v { s = x.to_string() + s; } s.len()",
    );
    assert_eq!(
        held,
        cycle(vec![TokenKind::Storage], Order::InOrder, None),
        "{lines}"
    );
}

#[test]
fn a_storage_read_again_after_its_store_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = [5, 3]; let s = 0; let t = 0; for x in &v { s = s + *x; t = t * 0 + s; } s + t",
    );
    assert_eq!(held.law, None, "{lines}");
}
