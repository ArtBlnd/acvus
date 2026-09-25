//! Where `optimize::stages` cuts each loop after the full pipeline, and what
//! `analysis::loop_deps` computes of each stage (RFC-0089): free, or its
//! cycles, each with its tokens, order and law; and the loop's control.

use acvus_mir::analysis::affine::{AffineValues, Derivation};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loans::Loans;
use acvus_mir::analysis::loop_deps::{
    Control, Law, LawOp, LoopDeps, Member, Order, StageMembership, Storage, Token,
};
use acvus_mir::analysis::loops::{Invariants, LoopNest, natural_loops_innermost_first};
use acvus_mir::analysis::targets::{effect, slots_lent_mutably};
use acvus_mir::analysis::cost::{CostTable, Costs, InPlace, LoopCost};
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ir::{BinOp, Callee, InstKind, Overflow, ValueId};
use acvus_mir::laws::{LawTable, Returns};
use acvus_mir::printer::dump_with_facts;
use acvus_mir::ty::{Effect, GenericSig, ParamTerm, Poly, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::{LoweredScript, compile_script_at, optimized_script};
use acvus_extern::{Registry, TypesOnly};
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

    /// The `for` whose loop holds every other `for`.
    fn outer_header(&self) -> BlockIdx {
        let loops = natural_loops_innermost_first(&self.cfg, &DomTree::build(&self.cfg));
        loops
            .iter()
            .filter(|loop_| matches!(self.cfg.blocks[loop_.header.0].terminator, Terminator::For { .. }))
            .max_by_key(|loop_| loop_.block_count())
            .map(|loop_| loop_.header)
            .unwrap_or_else(|| panic!("a `for`:\n{}", self.listing))
    }

    fn deps(&self) -> LoopDeps {
        self.deps_of(self.only_header())
    }

    fn deps_of(&self, header: BlockIdx) -> LoopDeps {
        LoopDeps::of(&self.cfg, &self.laws, header)
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
        self.shapes_of(self.only_header())
    }

    fn shapes_of(&self, header: BlockIdx) -> Vec<Shape> {
        let deps = self.deps_of(header);
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
                        law: judged.law.as_ref().map(|law| LawShape {
                            law: LawKind::of(&law.accumulator.law),
                            exact: law.accumulator.exact,
                            scan: law.scan,
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
    /// RFC-0093 rule 8.
    scan: bool,
}

#[derive(Debug, PartialEq)]
enum LawKind {
    Op(LawOp),
    Call,
    Fold,
    Order,
    Last,
    Extremum(LawOp),
    OptionLifted(Box<LawKind>),
    Product(Vec<LawKind>),
    Ordered(LawOp),
    First,
    Reset(Box<LawKind>),
    AffineMap,
}

impl LawKind {
    fn of(law: &Law) -> LawKind {
        match law {
            Law::Op(op) => LawKind::Op(*op),
            Law::Call(_) => LawKind::Call,
            Law::Fold(_) => LawKind::Fold,
            Law::Order => LawKind::Order,
            Law::Last => LawKind::Last,
            Law::Extremum { op, .. } => LawKind::Extremum(*op),
            Law::OptionLifted(inner) => LawKind::OptionLifted(Box::new(LawKind::of(inner))),
            Law::Product(parts) => {
                LawKind::Product(
                    parts.iter().map(|(_, part)| LawKind::of(&part.accumulator.law)).collect(),
                )
            }
            Law::Ordered { op, .. } => LawKind::Ordered(*op),
            Law::First { .. } => LawKind::First,
            Law::Reset(inner) => LawKind::Reset(Box::new(LawKind::of(inner))),
            Law::AffineMap => LawKind::AffineMap,
        }
    }
}

fn cycle(tokens: Vec<TokenKind>, order: Order, law: Option<LawShape>) -> CycleShape {
    CycleShape { tokens, order, law }
}

fn one(tokens: Vec<TokenKind>, order: Order, law: Option<LawShape>) -> Shape {
    Shape::Cycles(vec![cycle(tokens, order, law)])
}

/// The exact `+` law of a cycle whose partials are read outside it
/// (RFC-0093 rule 8).
fn scanned_add() -> Option<LawShape> {
    Some(LawShape {
        law: LawKind::Op(LawOp::Add),
        exact: true,
        scan: true,
    })
}

fn exact(law: LawKind) -> Option<LawShape> {
    Some(LawShape {
        law,
        exact: true,
        scan: false,
    })
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
                    exact: false,
                    scan: false,
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

/// Table row, a loop with `break`: its control is chained through the
/// stage it leaves from. `y` and the test run ahead in a free stage, the
/// exit is the control token's cycle in order, and the sum after the exit
/// is a cycle of its own with its `+` law (RFC-0089 rules 5 and 6).
#[test]
fn a_break_chains_the_control_token_through_the_stage_it_leaves_from() {
    let c = Compiled::of(
        "let v = [1, 2, 3, 4]; let s = 0; \
         for x in &v { let y = *x * 3; if y > 6 { break; } s = s + y; } s",
    );
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(vec![TokenKind::Control], Order::InOrder, None),
            one(
                vec![TokenKind::Carried],
                Order::AnyOrder,
                exact(LawKind::Op(LawOp::Add))
            ),
        ],
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
    let c = Compiled::of("let i = 0; let s = 0; while i != 10 { s = s + i; i = i + 2; } s");
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
/// in that stage beside `acc`'s (RFC-0056). `acc`'s update reads the
/// counter's partial, so the counter's `+` law is a scan (RFC-0093 rule 8).
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
            cycle(vec![TokenKind::Carried], Order::InOrder, scanned_add())
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

/// The same print after a `break`: the loop leaves from its body, so the
/// control token is chained, and the spawn and the eval follow the stage
/// the loop leaves from, where the iteration has held the control token
/// (rule 5). No effect lies in a stage before it.
#[test]
fn an_anyorder_print_after_a_break_issues_its_effects_after_the_exiting_stage() {
    let c = Compiled::with_io(
        "let v = [1, 2, 3]; anyorder { for x in &v { if *x == 2 { break; }; io::print(\"a\"); } } 0",
    );
    let deps = c.deps();
    let exiting = c.exiting_stage();
    assert_eq!(deps.ahead_of_exit(), [], "{}", c.for_lines());
    assert!(
        deps.effects_in_free_stages()
            .iter()
            .all(|wait| wait.stage > exiting),
        "{}",
        c.for_lines()
    );
    let effects = c.stages_running(|kind| {
        matches!(kind, InstKind::Spawn { .. } | InstKind::Eval { .. })
    });
    assert!(
        !effects.is_empty() && effects.iter().all(|stage| *stage > exiting),
        "{effects:?}\n{}",
        c.for_lines()
    );
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

/// A loop that leaves early keeps its carried induction variable, which no
/// stage reads before the one its cycle lies in.
#[test]
fn a_carried_iv_is_read_no_earlier_than_the_stage_its_cycle_lies_in() {
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
    let stage = cycle
        .stage()
        .unwrap_or_else(|| panic!("no cycle crosses a boundary:\n{}", c.for_lines()));
    assert!(
        reading.contains(&stage) && reading.iter().all(|at| *at >= stage),
        "{reading:?}\n{}",
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
    assert_eq!(
        held[..],
        [
            cycle(vec![TokenKind::Carried], Order::InOrder, None),
            cycle(vec![TokenKind::Carried], Order::InOrder, scanned_add())
        ],
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
        scan: false,
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

// -- A loop the body leaves (RFC-0089 rules 5 and 6) ---------------------

impl Compiled {
    fn with_registries(source: &str, own: Vec<Registry<TypesOnly>>) -> Self {
        let interner = Interner::new();
        let compiled = optimized_script(&interner, source, &[], own)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self::from(&interner, compiled)
    }

    fn exiting_stage(&self) -> usize {
        let deps = self.deps();
        let Control::Chained { cycle } = deps.control else {
            panic!("the loop leaves from its body:\n{}", self.for_lines())
        };
        deps.cycles[cycle]
            .stage()
            .unwrap_or_else(|| panic!("no cycle crosses a boundary:\n{}", self.for_lines()))
    }

    fn stages_running(&self, wanted: impl Fn(&InstKind) -> bool) -> Vec<usize> {
        self.stages_where(|cfg, block| cfg.blocks[block.0].insts.iter().any(|inst| wanted(&inst.kind)))
    }
}

fn is_call(kind: &InstKind) -> bool {
    matches!(kind, InstKind::FunctionCall { .. })
}

/// `probe` at `f64`, registered first, states nothing; at `i64` it states
/// `returns`. Both are one name.
mod probe_fx {
    use acvus_extern::{Registry, TypesOnly, extern_fn, extern_registry, extern_signature};

    extern_signature! {
        ns: "probe",
        fn probe<T>(x: T) -> bool
        where
            T: Var<kind::Type>;
    }

    #[extern_fn(instance_of = probe, effect = pure)]
    fn probe_float(x: f64) -> bool {
        let _ = x;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(instance_of = probe, effect = pure, returns)]
    fn probe_int(x: i64) -> bool {
        let _ = x;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "probe",
            signatures: [probe],
            fns: [probe_float, probe_int],
        }
    }
}

/// Table row S01: the predicate finishes and has no effect, so it runs
/// ahead in a free stage, and the exit is the control token's cycle, in
/// order, in the stage after it.
#[test]
fn a_body_exit_runs_its_predicate_ahead_in_a_free_stage() {
    let c = Compiled::of(
        "let xs = vec([5, 3, 9, 1]); let at = 99u64; \
         for i in 0u64..xs.len() { if xs[i] == 9 { at = i; break; }; } at",
    );
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(vec![TokenKind::Control], Order::InOrder, None)
        ],
        "{}",
        c.for_lines()
    );
    assert_eq!(c.exiting_stage(), 1, "{}", c.for_lines());
}

/// `starts_with` states no `returns`: nothing says a call ends, so it waits
/// for the control token in the exiting stage.
#[test]
fn a_predicate_calling_an_extern_that_states_no_returns_stays_in_the_exiting_stage() {
    let c = Compiled::of(
        "let xs = vec([\"alpha\".to_string(), \"beta\".to_string()]); let hit = false; \
         for s in &xs { if s.starts_with(\"be\") { hit = true; break; }; } hit",
    );
    assert_eq!(c.stages_running(is_call), [c.exiting_stage()], "{}", c.for_lines());
}

/// A `while` in the predicate is not known to finish (RFC-0089 rule 5). It
/// compares `c * 1`, which no counted form of RFC-0094 reads, so it stays a
/// `while`.
#[test]
fn a_predicate_holding_a_while_stays_in_the_exiting_stage() {
    let c = Compiled::of(
        "let v = vec([1, 5, 9]); let at = 99u64; \
         for i in 0u64..v.len() { let n = v[i]; let c = 0; while c * 1 < n { c = c + 2; } \
         if c > 4 { at = i; break; }; } at",
    );
    let stepping = |kind: &InstKind| {
        matches!(kind, InstKind::BinOp { op: BinOp::Add(_), .. })
    };
    assert_eq!(c.stages_running(stepping), [c.exiting_stage()], "{}", c.for_lines());
}

/// A call of a local function is not known to finish (RFC-0089 rule 5).
#[test]
fn a_predicate_calling_a_local_function_stays_in_the_exiting_stage() {
    let c = Compiled::of(
        "let f = |x| -> { if x > 2 { x * 3 } else { 0 } }; let v = vec([1, 5, 9]); \
         let at = 99u64; for i in 0u64..v.len() { if f(v[i]) > 4 { at = i; break; }; } at",
    );
    let local_call = |kind: &InstKind| {
        matches!(
            kind,
            InstKind::FunctionCall {
                callee: Callee::Direct(_) | Callee::Indirect(_),
                ..
            }
        )
    };
    let calls = c.stages_running(local_call);
    assert!(!calls.is_empty(), "the call of `f` stays a call:\n{}", c.listing);
    assert_eq!(calls, [c.exiting_stage()], "{}", c.for_lines());
}

/// A print before the exit is issued once its iteration holds the control
/// token: its spawn and eval lie in the exiting stage, and no effect lies in
/// a stage before it. Under `anyorder` they are free units the exit does not
/// read; outside it they are the `Order`'s cycle, which the exiting stage
/// then holds.
#[test]
fn an_effect_before_the_exit_stays_in_the_exiting_stage() {
    for source in [
        "let v = [1, 2, 3]; anyorder { for x in &v { io::print(\"a\"); if *x == 2 { break; }; } } 0",
        "let v = [1, 2, 3]; for x in &v { io::print(\"a\"); if *x == 2 { break; }; } 0",
    ] {
        let c = Compiled::with_io(source);
        let effects = c.stages_running(|kind| {
            matches!(kind, InstKind::Spawn { .. } | InstKind::Eval { .. })
        });
        assert_eq!(effects, [c.exiting_stage()], "{}", c.for_lines());
        assert_eq!(c.deps().ahead_of_exit(), [], "{}", c.for_lines());
    }
}

/// `emit` again, stating `returns`: it finishes, and only its effect holds
/// it behind the exit.
fn emit_returning(i: &Interner) -> Vec<Function> {
    let mut functions = emit(i);
    let FnKind::Extern { instances, .. } = &mut functions[0].kind else {
        unreachable!("`emit` is an extern")
    };
    instances.generic = Some(GenericSig {
        returns: Returns::Stated,
        ..GenericSig::default()
    });
    functions
}

/// An effect that finishes is still held back: under `anyorder` the spawn
/// of `emit_returning` is a free unit before the exit, and it lies in the
/// exiting stage.
#[test]
fn an_effect_that_finishes_before_the_exit_stays_in_the_exiting_stage() {
    let c = Compiled::with_externs(
        "let n = 5; anyorder { for k in 0..n { emit(k); if k == 2 { break; }; } } 0",
        emit_returning,
    );
    let spawns = c.stages_running(|kind| matches!(kind, InstKind::Spawn { .. }));
    assert_eq!(spawns, [c.exiting_stage()], "{}", c.for_lines());
}

/// `returns` is read off the instance a call names: `probe` at `i64`
/// states it and runs ahead; at `f64`, one name, it states nothing and
/// waits in the exiting stage.
#[test]
fn returns_is_read_by_the_instance_a_call_names() {
    let loop_over = |values: &str| {
        Compiled::with_registries(
            &format!(
                "let v = vec([{values}]); let hit = false; \
                 for x in &v {{ if probe::probe(*x) {{ hit = true; break; }}; }} hit"
            ),
            vec![probe_fx::registry()],
        )
    };
    let int = loop_over("1, 2");
    assert_eq!(int.stages_running(is_call), [0], "{}", int.for_lines());
    assert_eq!(int.exiting_stage(), 1, "{}", int.for_lines());
    let float = loop_over("1.5, 2.5");
    assert_eq!(float.stages_running(is_call), [float.exiting_stage()], "{}", float.for_lines());
}

/// RFC-0089 rule 4, through a nested loop: `total` enters the inner loop as
/// its header parameter's entry value and the inner loop's exit hands it
/// back, the inner cycle on it is `+` and nothing else there reads it, so
/// the outer cycle on `total` is `+` too.
#[test]
fn a_total_an_inner_loop_sums_into_has_the_add_law_in_the_outer_loop() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { for x in &row { total = total + *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        )],
        "{}",
        c.listing
    );
}

/// The inner loop reads `total` besides its `+`: `total % 2` reads a
/// partial, so neither loop has a law.
#[test]
fn an_inner_loop_that_also_reads_the_total_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 1; \
         for row in &m { for x in &row { total = total + *x + total % 2; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(vec![TokenKind::Carried], Order::InOrder, None)],
        "{}",
        c.listing
    );
}

/// The inner loop multiplies `total` and the outer loop adds to it: one
/// iteration of the outer loop maps `total` to `(total + 1)·p`, `p` the
/// row's product, which is the affine map law over `i64` (RFC-0093 rule 8).
#[test]
fn an_inner_product_under_an_outer_sum_gives_the_outer_loop_the_affine_map_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 1; \
         for row in &m { total = total + 1; for x in &row { total = total * *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried],
            Order::InOrder,
            exact(LawKind::AffineMap)
        )],
        "{}",
        c.listing
    );
}

/// A float sum through an inner loop keeps its inexact law, and is joined
/// in order: a regrouping changes the rounding.
#[test]
fn a_float_total_an_inner_loop_sums_into_is_in_order_in_the_outer_loop() {
    let c = Compiled::of(
        "let m = vec([vec([1.5, 2.5]), vec([3.0, 0.25])]); let total = 0.0; \
         for row in &m { for x in &row { total = total + *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried],
            Order::InOrder,
            Some(LawShape {
                law: LawKind::Op(LawOp::Add),
                exact: false,
                scan: false,
            })
        )],
        "{}",
        c.listing
    );
}

/// An inner loop that leaves early by a `break` decided by its element hands
/// back `total` at the iteration it leaves in, which is still the entry value
/// plus a run of the inner loop's elements, so the outer cycle on `total` is
/// `+`.
#[test]
fn an_inner_loop_leaving_early_by_its_element_hands_the_outer_loop_the_add_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { for x in &row { if *x == 5 { break; }; total = total + *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Add))
        )],
        "{}",
        c.listing
    );
}

/// The inner loop leaves when `total` passes 3: how much of a row joins
/// `total` depends on `total`, so the outer loop has no law.
#[test]
fn an_inner_loop_leaving_by_the_total_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { for x in &row { if total > 3 { break; }; total = total + *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(vec![TokenKind::Carried], Order::InOrder, None)],
        "{}",
        c.listing
    );
}

/// The inner loop leaves by `t0`, the total the outer iteration entered it
/// with. The inner loop's own cycle on `total` is still `+`, but how much of
/// a row joins `total` depends on `total`, so the outer loop has no law.
#[test]
fn an_inner_loop_leaving_by_the_entry_total_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { let t0 = total; \
         for x in &row { if t0 > 3 { break; }; total = total + *x; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(vec![TokenKind::Carried], Order::InOrder, None)],
        "{}",
        c.listing
    );
}

/// The `break` hands back `100`, not a sum, so the outer loop has no law.
#[test]
fn an_inner_loop_leaving_with_another_value_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { for x in &row { total = total + *x; \
         if *x == 5 { total = 100; break; }; } } total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(vec![TokenKind::Carried], Order::InOrder, None)],
        "{}",
        c.listing
    );
}

/// A `return` of `total` from the inner loop leaves the outer loop too: the
/// exit is the control token's cycle, `total`'s cycle joins it, and a cycle
/// of two tokens has no law (RFC-0089 rule 5).
#[test]
fn a_return_of_the_total_from_an_inner_loop_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]); let total = 0; \
         for row in &m { for x in &row { if *x == 5 { return total; }; total = total + *x; } } \
         total",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried, TokenKind::Control],
            Order::InOrder,
            None
        )],
        "{}",
        c.listing
    );
}

/// A `?` in the inner loop stays where it is, since its `Err` holds a
/// `String`, and leaves the outer loop from inside the inner one: the outer
/// loop is one stage, the control token's cycle with `total` in it, with no
/// law.
#[test]
fn a_try_in_an_inner_loop_gives_the_outer_loop_no_law() {
    let c = Compiled::of(
        "let rows = vec([vec([\"1\".to_string()]), vec([\"30\".to_string()])]); let total = 0; \
         for row in &rows { for s in &row { total = total + i64::from_str(s)?; } } Ok(total)",
    );
    assert_eq!(
        cycle_stages(&c.shapes_of(c.outer_header())),
        [&one(
            vec![TokenKind::Carried, TokenKind::Control],
            Order::InOrder,
            None
        )],
        "{}",
        c.listing
    );
}

/// RFC-0066 rule 1: the `return` inside the inner loop moves into the inner
/// loop's own exit, so the outer loop leaves from its own body. The inner
/// search finishes, so it runs ahead in a free stage, and the exit is the
/// control token's cycle after it (RFC-0089 rule 5).
#[test]
fn a_return_from_a_nested_loop_lets_the_outer_loop_run_the_inner_search_ahead() {
    let c = Compiled::of(
        "let m = vec([vec([1, 2]), vec([3, 7])]); \
         for i in 0u64..m.len() { for j in 0u64..m[i].len() { \
         if m[i][j] == 7 { return i * 10 + j; }; } } 99u64",
    );
    let outer = c.outer_header();
    assert_eq!(
        c.shapes_of(outer),
        [
            Shape::Free,
            one(vec![TokenKind::Control], Order::InOrder, None)
        ],
        "{}",
        c.listing
    );
    let deps = c.deps_of(outer);
    let inner_header_stage = (0..c.cfg.blocks.len())
        .map(BlockIdx)
        .filter(|block| *block != outer)
        .find(|block| matches!(c.cfg.blocks[block.0].terminator, Terminator::For { .. }))
        .and_then(|block| deps.membership.stage_of(block));
    assert_eq!(inner_header_stage, Some(0), "{}", c.listing);
    assert_eq!(deps.ahead_of_exit(), [], "{}", c.listing);
}

/// A nested loop over an `Array` keeps its `return`: the array's release on a
/// moved edge would be a drop block of its own, which runs the loop as joints
/// where the `return` leaves it one region (RFC-0057 rules 6 and 7).
#[test]
fn a_return_from_a_nested_loop_over_an_array_stays_where_it_is() {
    let c = Compiled::of(
        "let m = vec([1, 2]); \
         for i in 0u64..m.len() { for t in [\"x\".to_string(), \"yz\".to_string()] { \
         if len(&t) == 2 { return i; }; } } 99u64",
    );
    assert_eq!(
        c.shapes_of(c.outer_header()).len(),
        1,
        "the exit is not moved, and the outer loop is one stage:\n{}",
        c.listing
    );
}

// -- Laws through switches, chains, Bool, several tokens, Option, `last`
// -- and a left-biased extremum (RFC-0089 rules 2 and 4) -----------------

fn law_of(held: &CycleShape) -> Option<&LawKind> {
    held.law.as_ref().map(|shape| &shape.law)
}

/// Every token of every cycle of `source`'s one loop.
fn all_tokens(source: &str) -> (Vec<TokenKind>, String) {
    let c = Compiled::of(source);
    let deps = c.deps();
    let tokens = deps
        .cycles
        .iter()
        .flat_map(|cycle| cycle.tokens.iter().map(TokenKind::of))
        .collect();
    (tokens, c.for_lines())
}

#[test]
fn a_short_circuit_or_on_the_token_is_the_or_law() {
    let (held, lines) =
        the_cycle("let v = [5, 3, 8]; let f = false; for x in &v { f = f || *x > 4; } f");
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Or))
        ),
        "{lines}"
    );
}

#[test]
fn a_short_circuit_and_on_the_token_is_the_and_law() {
    let (held, lines) =
        the_cycle("let v = [5, 3, 8]; let f = true; for x in &v { f = f && *x > 2; } f");
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::And))
        ),
        "{lines}"
    );
}

/// A chunk combined from `false` computes the right operand where the
/// program's own run skips it, so an operand that can trap has no law.
#[test]
fn a_short_circuit_whose_skipped_operand_can_raise_has_no_law() {
    let (held, lines) =
        the_cycle("let v = [5, 0, 8]; let f = true; for x in &v { f = f || 10 / *x > 1; } f");
    assert_eq!(law_of(&held), None, "{lines}");
    assert_eq!(held.order, Order::InOrder, "{lines}");
}

#[test]
fn an_arm_fixing_a_bool_token_reads_as_the_law_that_constant_absorbs() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 9]; let f = false; for x in &v { if *x == 9 { f = true; }; } f",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Op(LawOp::Or)), "{lines}");
    assert_eq!(held.order, Order::AnyOrder, "{lines}");
    let (held, lines) = the_cycle(
        "let v = [5, 3, 9]; let f = true; for x in &v { if *x == 9 { f = false; }; } f",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Op(LawOp::And)), "{lines}");
    assert_eq!(held.order, Order::AnyOrder, "{lines}");
}

#[test]
fn a_negation_under_a_branch_is_the_xor_law() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 9, 9]; let on = false; for x in &v { if *x == 9 { on = !on; }; } on",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Op(LawOp::Xor))
        ),
        "{lines}"
    );
}

#[test]
fn two_tokens_whose_steps_read_no_other_have_the_product_of_their_laws() {
    let (held, lines) = the_cycle(
        "let v = [5, -3, 8]; let s = 0; let n = 0; \
         for x in &v { if *x >= 0 { s = s + *x; n = n + 1; }; } s / n",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried, TokenKind::Carried],
            Order::AnyOrder,
            exact(LawKind::Product(vec![
                LawKind::Op(LawOp::Add),
                LawKind::Op(LawOp::Add)
            ]))
        ),
        "{lines}"
    );
}

/// RFC-0093 rule 8: `n`'s steps read no other token and have `+`, and `s`'s
/// step reads it, so `n` is a scan and `s` adds over its partials; the
/// product is a scan, in order.
#[test]
fn two_tokens_one_of_whose_steps_reads_the_other_scan_the_one_read() {
    let (held, lines) = the_cycle(
        "let v = [5, -3, 8]; let s = 0; let n = 0; \
         for x in &v { if *x >= 0 { s = s + n; n = n + 1; }; } s",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried, TokenKind::Carried],
            Order::InOrder,
            Some(LawShape {
                law: LawKind::Product(vec![LawKind::Op(LawOp::Add), LawKind::Op(LawOp::Add)]),
                exact: true,
                scan: true,
            })
        ),
        "{lines}"
    );
    assert!(
        lines.contains("Carried(r11): Op(Add) exact commutative, Carried(r12): Op(Add) exact commutative scan"),
        "the part `s` reads is the scan: {lines}"
    );
}

#[test]
fn a_switch_on_an_option_token_combining_its_payload_is_the_law_lifted_over_option() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 3, 8]); let best = None; \
         for x in &v { best = match best { None => Some(*x), Some(b) => Some(max(b, *x)), }; } \
         best.unwrap()",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::AnyOrder,
            exact(LawKind::OptionLifted(Box::new(LawKind::Call)))
        ),
        "{lines}"
    );
}

#[test]
fn a_switch_on_an_option_token_combining_another_value_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 3, 8]); let best = None; \
         for x in &v { best = match best { None => Some(*x), Some(b) => Some(max(b, *x + 1)), }; } \
         best.unwrap()",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// Both values are computed before the switch, so only their being one
/// value tells the lifted law from another.
#[test]
fn a_switch_on_an_option_token_combining_a_value_other_than_the_one_it_starts_from_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 3, 8]); let best = None; \
         for x in &v { let y = *x; let z = *x + 1; \
         best = match best { None => Some(y), Some(b) => Some(max(b, z)), }; } \
         best.unwrap()",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

#[test]
fn an_arm_setting_a_value_reading_none_of_the_token_is_last() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let at = 99u64; \
         for i in 0u64..v.len() { if v[i] == 9 { at = i; }; } at",
    );
    assert_eq!(
        held,
        cycle(vec![TokenKind::Carried], Order::InOrder, exact(LawKind::Last)),
        "{lines}"
    );
}

#[test]
fn an_arm_setting_the_token_under_a_branch_that_reads_it_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let at = 99u64; \
         for i in 0u64..v.len() { if at == 99u64 && v[i] == 9 { at = i; }; } at",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

#[test]
fn a_strict_compare_and_select_carrying_another_token_is_a_left_biased_extremum() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let best = i64::MIN(); let at = 0u64; \
         for i in 0u64..v.len() { if v[i] > best { best = v[i]; at = i; }; } at",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried, TokenKind::Carried],
            Order::InOrder,
            exact(LawKind::Extremum(LawOp::Max))
        ),
        "{lines}"
    );
}

/// With `>=` a tie takes the later index: no left-biased extremum.
#[test]
fn a_compare_and_select_by_a_non_strict_order_carrying_a_token_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let best = i64::MIN(); let at = 0u64; \
         for i in 0u64..v.len() { if v[i] >= best { best = v[i]; at = i; }; } at",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

#[test]
fn a_compare_and_select_carrying_a_value_that_reads_a_token_has_no_law() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let best = i64::MIN(); let at = 0u64; \
         for i in 0u64..v.len() { if v[i] > best { best = v[i]; at = at + 1u64; }; } at",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

#[test]
fn a_header_parameter_every_back_edge_sends_one_constant_is_no_token() {
    let source = "let v = [5, 3, 8]; let first = true; let s = 0; \
         for x in &v { if first { s = s + 100; }; first = false; s = s + *x; } s";
    let (tokens, lines) = all_tokens(source);
    assert_eq!(tokens, [TokenKind::Carried], "{lines}");
    let (held, lines) = the_cycle(source);
    assert_eq!(law_of(&held), Some(&LawKind::Op(LawOp::Add)), "{lines}");
}

#[test]
fn a_header_parameter_kept_where_a_branch_on_it_fixed_it_is_no_token() {
    let source = "let v = [5, 3, 8]; let first = true; let s = 0; \
         for x in &v { if first { s = s + 100; first = false; } else { s = s + *x; }; } s";
    let (tokens, lines) = all_tokens(source);
    assert_eq!(tokens, [TokenKind::Carried], "{lines}");
}

#[test]
fn a_header_parameter_sent_its_own_negation_stays_a_token() {
    let source = "let v = [5, 3, 8]; let odd = true; let s = 0; \
         for x in &v { if odd { s = s + *x; }; odd = !odd; } s";
    let (tokens, lines) = all_tokens(source);
    assert_eq!(tokens.len(), 2, "{lines}");
}

#[test]
fn two_assignments_of_one_storage_in_an_iteration_are_read_as_their_chain() {
    let (held, lines) = the_cycle(
        "let out = \"\".to_string(); \
         for i in 0u64..3u64 { if i > 0u64 { out = out + \",\"; }; out = out + i.to_string(); } \
         out",
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
}

#[test]
fn a_chain_of_assignments_under_a_branch_that_reads_the_storage_has_no_law() {
    let (held, lines) = the_cycle(
        "let out = \"\".to_string(); \
         for i in 0u64..3u64 { if len(&out) > 0u64 { out = out + \",\"; }; out = out + i.to_string(); } \
         out",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// The skip arm keeps `flag` under a branch on another value, so past the
/// first iteration it is not one constant.
#[test]
fn a_header_parameter_kept_under_a_branch_on_another_value_stays_a_token() {
    let source = "let v = [5, 3, 8]; let flag = true; \
         for x in &v { if *x > 4 { flag = false; }; } flag";
    let (tokens, lines) = all_tokens(source);
    assert_eq!(tokens, [TokenKind::Carried], "{lines}");
}

// -- `first`, the first-iteration reset, and a total order (RFC-0089 rule 4,
// RFC-0082 rule 10) ------------------------------------------------------

const FIRST_INDEX: &str = "let v = vec([5, 9, 1, 9]); let at = 99u64; let found = false; \
     for i in 0u64..v.len() { if !found && v[i] == 9 { at = i; found = true; }; } at";

#[test]
fn a_flag_its_arm_sets_guarding_that_arm_is_first() {
    let (held, lines) = the_cycle(FIRST_INDEX);
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried, TokenKind::Carried],
            Order::InOrder,
            exact(LawKind::First)
        ),
        "{lines}"
    );
}

/// The arm runs where `found` already holds, so a later hit replaces the
/// earlier one.
#[test]
fn an_arm_the_flag_it_sets_does_not_guard_is_not_first() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let at = 99u64; let found = false; \
         for i in 0u64..v.len() { if found || v[i] == 9 { at = i; found = true; }; } at",
    );
    assert_ne!(law_of(&held), Some(&LawKind::First), "{lines}");
}

/// A chunk run from the unset flag computes the test where the program
/// skips it, so a test that can raise keeps the cycle without a law.
#[test]
fn a_guarded_test_that_can_raise_is_not_first() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 9, 1, 9]); let at = 99u64; let found = false; \
         for i in 0u64..v.len() { if !found && 18 / v[i] == 2 { at = i; found = true; }; } at",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// A print in the arm is an effect a chunk run from the unset flag would
/// issue where the program does not.
#[test]
fn a_guarded_arm_that_prints_is_not_first() {
    let c = Compiled::with_io(
        "let v = vec([5, 9, 1, 9]); let at = 99u64; let found = false; \
         for i in 0u64..v.len() { if !found && v[i] == 9 { at = i; found = true; \
         io::print(\"hit\"); }; } at",
    );
    assert!(!every_law(&c).contains(&LawKind::First), "{}", c.for_lines());
}

/// RFC-0093 rule 7: `pivot` is compared with `6`, which no write sends (the
/// counter is below `6`), so `pivot == 6` is its own `||` guard.
const SENTINEL_FIRST: &str = "let part = [2, 1, 1, 0, 1, 2]; let pivot = 6u64; \
     for v in 0u64..6u64 { if pivot == 6u64 && part[v] == 1 { pivot = v; }; } pivot";

#[test]
fn a_token_compared_with_a_constant_no_write_sends_is_its_own_first_guard() {
    let (held, lines) = the_cycle(SENTINEL_FIRST);
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Carried],
            Order::InOrder,
            exact(LawKind::First)
        ),
        "{lines}"
    );
}

/// `!=` with the sentinel is unset where it fails.
#[test]
fn a_sentinel_tested_by_inequality_is_its_own_first_guard_too() {
    let (held, lines) = the_cycle(
        "let part = [2, 1, 1, 0, 1, 2]; let pivot = 6u64; \
         for v in 0u64..6u64 { if !(pivot != 6u64) && part[v] == 1 { pivot = v; }; } pivot",
    );
    assert_eq!(law_of(&held), Some(&LawKind::First), "{lines}");
}

/// The counter reaches `6`, so an arm can send the sentinel and leave the
/// token unset after a hit.
#[test]
fn a_sentinel_a_write_can_send_is_no_first_guard() {
    let (held, lines) = the_cycle(
        "let part = [2, 1, 1, 0, 1, 2, 1]; let pivot = 6u64; \
         for v in 0u64..7u64 { if pivot == 6u64 && part[v] == 1 { pivot = v; }; } pivot",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// The arm runs where `pivot` is already set, so a later hit replaces the
/// earlier one.
#[test]
fn an_arm_the_sentinel_does_not_guard_is_not_first() {
    let (held, lines) = the_cycle(
        "let part = [2, 1, 1, 0, 1, 2]; let pivot = 6u64; \
         for v in 0u64..6u64 { if pivot != 6u64 || part[v] == 1 { pivot = v; }; } pivot",
    );
    assert_ne!(law_of(&held), Some(&LawKind::First), "{lines}");
}

/// Compared with two constants, the token has no one sentinel.
#[test]
fn a_token_compared_with_two_constants_has_no_sentinel() {
    let (held, lines) = the_cycle(
        "let part = [2, 1, 1, 0, 1, 2]; let pivot = 6u64; \
         for v in 0u64..6u64 { if pivot == 6u64 && pivot != 7u64 && part[v] == 1 \
         { pivot = v; }; } pivot",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// `pivot < n` after the search assumes `n ≤ pivot` on its false side, and
/// that bound on `n` is joined away around the outer loop; `n` is still the
/// constant its `Const` defines, so the counter below it never sends `n`.
#[test]
fn a_sentinel_a_later_test_bounds_is_still_the_constant_its_definition_is() {
    let c = Compiled::of(
        "let n = 6u64; let total = 0u64; \
         for round in 0u64..2u64 { let pivot = n; \
         for v in 0u64..n { if pivot == n && v >= 2u64 { pivot = v; }; } \
         if pivot < n { total = total + pivot + round; }; } total",
    );
    let inner = c
        .listing
        .lines()
        .find(|line| line.contains("cycle Carried") && line.contains("in_order"))
        .unwrap_or_else(|| panic!("the search's cycle:\n{}", c.listing));
    assert!(inner.contains("law(First("), "{}", c.listing);
}

/// App 02:98's shape: `part[v]` into a vec whose length the interval
/// domain does not know can trap, and a chunk run from the unset sentinel
/// computes it where the program skips it.
#[test]
fn a_sentinel_guarding_a_test_that_can_trap_is_no_first_guard() {
    let (held, lines) = the_cycle(
        "let part = vec([2, 1, 1, 0, 1, 2]); let pivot = 6u64; \
         for v in 0u64..6u64 { if pivot == 6u64 && part[v] == 1 { pivot = v; }; } pivot",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// The law of each cycle of the one loop that has one.
fn every_law(c: &Compiled) -> Vec<LawKind> {
    c.shapes()
        .into_iter()
        .flat_map(|shape| match shape {
            Shape::Free => Vec::new(),
            Shape::Cycles(cycles) => cycles,
        })
        .filter_map(|held| held.law.map(|shape| shape.law))
        .collect()
}

const JOIN_WITH_SEPARATOR: &str = "let xs = vec([\"a\".to_string(), \"b\".to_string()]); \
     let s = \"\".to_string(); let first = true; \
     for x in &xs { if first { s = x.to_string(); first = false; } \
     else { s = s + \", \" + x; }; } s";

#[test]
fn an_arm_taken_only_at_the_first_iteration_resets_the_law() {
    let (held, lines) = the_cycle(JOIN_WITH_SEPARATOR);
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::InOrder,
            exact(LawKind::Reset(Box::new(LawKind::Op(LawOp::Concat))))
        ),
        "{lines}"
    );
}

/// A flag entering `false` never takes the arm, so the join needs the
/// entry value.
#[test]
fn an_arm_on_a_flag_that_enters_unset_is_no_reset() {
    let (held, lines) = the_cycle(&JOIN_WITH_SEPARATOR.replace("first = true;", "first = false;"));
    assert_eq!(law_of(&held), None, "{lines}");
}

/// The branch on the flag runs only where the element passes a test, so
/// the first iteration may skip the arm and the join needs the entry value.
#[test]
fn an_arm_under_a_flag_branch_that_not_every_iteration_runs_is_no_reset() {
    let c = Compiled::of(
        "let xs = vec([\"a\".to_string(), \"bc\".to_string()]); \
         let s = \"\".to_string(); let first = true; \
         for x in &xs { if len(x) > 1u64 { if first { s = x.to_string(); } \
         else { s = s + \", \" + x; }; }; first = false; } s",
    );
    assert_eq!(every_law(&c), [], "{}", c.for_lines());
}

#[test]
fn a_select_on_the_sign_of_a_total_order_is_its_maximum() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"fig\".to_string(), \"pear\".to_string()]); \
         let best = \"\".to_string(); \
         for x in &xs { if string::cmp(x, &best) > 0 { best = x.clone(); }; } best",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::AnyOrder,
            exact(LawKind::Ordered(LawOp::Max))
        ),
        "{lines}"
    );
}

#[test]
fn a_select_on_a_negative_sign_with_the_state_on_the_left_is_its_maximum_too() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"fig\".to_string(), \"pear\".to_string()]); \
         let best = \"\".to_string(); \
         for x in &xs { if string::cmp(&best, x) < 0 { best = x.clone(); }; } best",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Ordered(LawOp::Max)), "{lines}");
}

#[test]
fn a_select_on_the_sign_of_a_word_s_total_order_is_its_minimum() {
    let (held, lines) = the_cycle(
        "let v = vec([5, 3, 8]); let best = 100; \
         for x in &v { if cmp(x, &best) < 0 { best = *x; }; } best",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Ordered(LawOp::Min)), "{lines}");
}

/// `total_cmp` orders every `f64`, `-0.0` below `0.0` and each NaN by its
/// bits, and its `Equal` is the bit equality `==` is on a `Float`.
#[test]
fn a_select_on_the_sign_of_the_float_total_order_is_its_maximum() {
    let (held, lines) = the_cycle(
        "let v = vec([1.5, 2.5]); let best = 0.0; \
         for x in &v { if cmp(x, &best) > 0 { best = *x; }; } best",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::AnyOrder,
            exact(LawKind::Ordered(LawOp::Max))
        ),
        "{lines}"
    );
}

/// RFC-0082 rule 2: `string::concat` takes the `str` views of its operands
/// and states its law over `String`, so a call on the view the state lends
/// is read as that law, in order since it does not commute.
#[test]
fn a_call_on_the_view_the_state_lends_is_read_by_the_law_over_its_type() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"a\".to_string(), \"b\".to_string()]); let text = \"\".to_string(); \
         for x in &xs { text = text.concat(x).concat(\";\"); } text",
    );
    assert_eq!(
        held,
        cycle(
            vec![TokenKind::Storage],
            Order::InOrder,
            exact(LawKind::Call)
        ),
        "{lines}"
    );
}

/// The state is `concat`'s right operand, and `concat` does not commute.
#[test]
fn a_call_on_the_view_the_state_lends_as_the_right_operand_has_no_law() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"a\".to_string(), \"b\".to_string()]); let text = \"\".to_string(); \
         for x in &xs { text = string::concat(x, &text); } text",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// `by_magnitude` compares `-3` and `3` equal, two values, and states no
/// `total_order`; the chosen `*x` is the compared value itself.
#[test]
fn a_select_on_the_sign_of_a_comparison_stating_no_total_order_has_no_law() {
    let c = Compiled::with_registries(
        "let v = vec([-3, 3]); let best = 0; \
         for x in &v { if magnitude::by_magnitude(x, &best) > 0 { best = *x; }; } best",
        vec![magnitude_fx::registry()],
    );
    let lines = c.for_lines();
    let cycles: Vec<CycleShape> = c
        .shapes()
        .into_iter()
        .flat_map(|shape| match shape {
            Shape::Free => Vec::new(),
            Shape::Cycles(cycles) => cycles,
        })
        .collect();
    let [held] = &cycles[..] else {
        panic!("the loop holds one cycle:\n{lines}")
    };
    assert_eq!(law_of(held), None, "{lines}");
}

mod magnitude_fx {
    use acvus_extern::{Registry, TypesOnly, extern_fn, extern_registry};

    #[extern_fn(effect = pure, total)]
    fn by_magnitude(a: &i64, b: &i64) -> i64 {
        let _ = (a, b);
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "magnitude",
            fns: [by_magnitude],
        }
    }
}

/// `string::concat(x, "")` holds the bytes `x` lends, but declares no
/// `copies`.
#[test]
fn a_select_choosing_a_value_no_declaration_makes_the_compared_one_has_no_law() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"fig\".to_string(), \"pear\".to_string()]); \
         let best = \"\".to_string(); \
         for x in &xs { let t = string::concat(x, \"\"); \
         if string::cmp(x, &best) > 0 { best = t; }; } best",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

/// The first element is copied before the branch, and the select compares
/// each element but chooses that copy.
#[test]
fn a_select_choosing_a_copy_of_another_value_than_the_compared_one_has_no_law() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"fig\".to_string(), \"pear\".to_string()]); \
         let best = \"\".to_string(); let head = &xs[0u64]; \
         for x in &xs { let t = head.to_string(); \
         if string::cmp(x, &best) > 0 { best = t; }; } best",
    );
    assert_eq!(law_of(&held), None, "{lines}");
}

#[test]
fn a_copy_of_the_state_is_read_as_the_state() {
    let (held, lines) = the_cycle(
        "let v = [5, 3, 8]; let s = 0; for x in &v { let t = s.clone(); s = t + *x; } s",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Op(LawOp::Add)), "{lines}");
}

/// `x.clone()` of a `String` is the language's own clone, which copies the
/// bytes `x` lends.
#[test]
fn a_select_choosing_the_clone_of_the_compared_string_is_its_minimum() {
    let (held, lines) = the_cycle(
        "let xs = vec([\"fig\".to_string(), \"pear\".to_string()]); \
         let least = \"zzz\".to_string(); \
         for x in &xs { if string::cmp(&least, x) > 0 { least = x.clone(); }; } least",
    );
    assert_eq!(law_of(&held), Some(&LawKind::Ordered(LawOp::Min)), "{lines}");
}

// -- Pull loops (RFC-0089 rule 1) ---------------------------------------

impl Compiled {
    fn only_pull_header(&self) -> BlockIdx {
        let found: Vec<BlockIdx> = (0..self.cfg.blocks.len())
            .map(BlockIdx)
            .filter(|at| matches!(self.cfg.blocks[at.0].terminator, Terminator::While { .. }))
            .collect();
        match found[..] {
            [header] => header,
            _ => panic!("one pull loop:\n{}", self.listing),
        }
    }
}

const PULL_PUSH: &str = "let xs = vec([5, 3, 8]); let it = xs.into_iter(); let out = vec([]); \
     while let Some(x) = it.next() { out.push(x * 2); } out.len()";

#[test]
fn a_pull_loop_s_header_is_its_first_stage_and_the_control_token_s_cycle() {
    let c = Compiled::of(PULL_PUSH);
    let header = c.only_pull_header();
    let deps = c.deps_of(header);
    assert_eq!(
        deps.membership.stages()[0].blocks,
        [header],
        "the header alone is the first stage:\n{}",
        c.listing
    );
    let Control::Chained { cycle } = deps.control else {
        panic!("control is chained through the pull:\n{}", c.listing)
    };
    assert_eq!(deps.cycles[cycle].stage(), Some(0));
    assert!(
        deps.cycles[cycle].tokens.contains(&Token::Control),
        "{}",
        c.listing
    );
    let held: Vec<Member> = (0..c.cfg.blocks[header.0].insts.len())
        .map(|at| Member::Inst(acvus_mir::analysis::loop_deps::InstAt { block: header, at }))
        .chain([Member::Term(header)])
        .collect();
    assert!(
        held.iter().all(|member| deps.cycles[cycle].members.contains(member)),
        "the header, its call and its test are the control token's cycle:\n{}",
        c.listing
    );
    assert_eq!(
        c.shapes_of(header),
        [
            one(vec![TokenKind::Storage, TokenKind::Control], Order::InOrder, None),
            Shape::Free,
            one(vec![TokenKind::Storage], Order::InOrder, exact(LawKind::Fold)),
        ],
        "the pull, the free `x * 2`, the push's fold:\n{}",
        c.listing
    );
    assert!(
        c.listing.contains("control chained through L0"),
        "{}",
        c.listing
    );
}

#[test]
fn a_pull_loop_prints_its_header_first_in_its_stages() {
    let c = Compiled::of(PULL_PUSH);
    let line = c
        .listing
        .lines()
        .find(|line| line.contains("while "))
        .unwrap_or_else(|| panic!("a `while` line:\n{}", c.listing));
    assert!(line.contains("stages [L0, L1, "), "{line}");
}

#[test]
fn a_pull_loop_s_effect_after_the_pull_runs_in_a_free_stage() {
    let c = Compiled::with_io(
        "let text = \"a\\nb\".to_string(); let ls = text.lines(); \
         anyorder { while let Some(l) = ls.next() { let u = l.upper(); io::print(&u); } } 0",
    );
    let header = c.only_pull_header();
    let shapes = c.shapes_of(header);
    assert_eq!(shapes[1], Shape::Free, "{}", c.listing);
    assert!(c.deps_of(header).ahead_of_exit().is_empty());
}

#[test]
fn a_pull_loop_costs_in_place_for_no_count_is_known_on_entry() {
    let c = Compiled::of(PULL_PUSH);
    let table = CostTable {
        arithmetic: 1,
        compare: 1,
        load: 1,
        store: 1,
        allocation: 1,
        local_call: 1,
        extern_call: 1,
        heavy: 1,
        spawn: 1,
        merge: 1,
        chunk_dispatch: 1,
        buffered_element: 1,
        k: 1,
    };
    let costs = Costs::of(&c.cfg, &c.laws, &table);
    assert_eq!(
        costs.of_loop(&c.deps_of(c.only_pull_header())),
        LoopCost::InPlace(InPlace::CountUnknown)
    );
}

#[test]
fn a_pull_loop_s_header_holds_no_drop() {
    let c = Compiled::of(
        "let it = vec([\"a\".to_string(), \"bc\".to_string()]).into_iter(); let n = 0u64; \
         while let Some(s) = it.next() { n = n + s.len(); } n",
    );
    let header = c.only_pull_header();
    assert!(
        !c.cfg.blocks[header.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Drop { .. })),
        "a value dying on the body edge is dropped in the body, not at the pull:\n{}",
        c.listing
    );
}

fn refused_shapes(module: &acvus_mir::ir::MirModule, laws: &LawTable) -> Vec<String> {
    acvus_mir::validate::stages::check(module, laws)
        .into_iter()
        .map(|error| format!("{:?}", error.kind))
        .collect()
}

#[test]
fn a_while_whose_header_is_no_pull_is_refused() {
    let interner = Interner::new();
    let mut compiled = compile_script_at(&interner, PULL_PUSH, &FxHashMap::default(), Opt::Full)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(refused_shapes(&compiled.module, &compiled.laws), Vec::<String>::new());
    let mut cfg = promote(compiled.module.main.clone());
    let header = (0..cfg.blocks.len())
        .map(BlockIdx)
        .find(|at| matches!(cfg.blocks[at.0].terminator, Terminator::While { .. }))
        .expect("the pull loop");
    cfg.blocks[header.0].insts.remove(0);
    compiled.module.main = acvus_mir::cfg::demote(cfg);
    let refused = refused_shapes(&compiled.module, &compiled.laws);
    assert!(
        refused.iter().any(|kind| kind.contains("HeaderIsNoPull")),
        "{refused:?}"
    );
}

#[test]
fn a_pull_loop_that_leaves_from_its_body_is_refused() {
    let interner = Interner::new();
    let mut compiled = compile_script_at(&interner, PULL_PUSH, &FxHashMap::default(), Opt::Full)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut cfg = promote(compiled.module.main.clone());
    let header = (0..cfg.blocks.len())
        .map(BlockIdx)
        .find(|at| matches!(cfg.blocks[at.0].terminator, Terminator::While { .. }))
        .expect("the pull loop");
    let Terminator::While { exit, .. } = cfg.blocks[header.0].terminator.clone() else {
        panic!("the header ends in `While`")
    };
    let loops = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg));
    let [latch] = loops
        .iter()
        .find(|loop_| loop_.header == header)
        .expect("the pull loop is a natural loop")
        .latches[..]
    else {
        panic!("one latch")
    };
    let decided = cfg.blocks[header.0].insts[2].kind.clone();
    let InstKind::TestVariant { dst: cond, .. } = decided else {
        panic!("the header tests the pulled option")
    };
    cfg.blocks[latch.0].terminator = Terminator::JumpIf {
        cond,
        then_label: exit,
        then_args: Vec::new(),
        else_label: cfg.blocks[header.0].label,
        else_args: Vec::new(),
    };
    compiled.module.main = acvus_mir::cfg::demote(cfg);
    let refused = refused_shapes(&compiled.module, &compiled.laws);
    assert!(
        refused.iter().any(|kind| kind.contains("PullLeavesElsewhere")),
        "{refused:?}"
    );
}

#[test]
fn a_countdown_by_one_is_read_from_the_counter_and_carries_nothing() {
    let c = Compiled::of(
        "let xs = vec([5, 3, 8, 1]); let i = xs.len(); let s = 0; \
         while i > 0u64 { i = i - 1u64; s = s + xs[i]; } s",
    );
    assert_eq!(c.carried_ivs(), Vec::<ValueId>::new(), "{}", c.listing);
    assert_eq!(
        c.shapes(),
        [
            Shape::Free,
            one(vec![TokenKind::Carried], Order::AnyOrder, exact(LawKind::Op(LawOp::Add))),
        ],
        "{}",
        c.for_lines()
    );
}
