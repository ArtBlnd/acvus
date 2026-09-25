//! `optimize::while_to_for` (RFC-0081): which `while` loops become a range
//! `for`, what the rewrite leaves as it was, and the forms it declines.
//!
//! These tests read the loop as `Promoted::of` leaves it, before IV
//! canonicalization. That pass runs later in the full pipeline and replaces
//! `i` with the counter, which would hide what the promotion itself wrote.

use acvus_mir::analysis::affine::{AffineValues, Derivation};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::inst_info;
use acvus_mir::analysis::loops::{Invariant, Invariants, Loop, LoopKind, LoopNest, Trip};
use acvus_mir::analysis::raise::FunctionSummary;
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ir::{BinOp, ForSource, InstKind, Overflow, ValueId};
use acvus_mir::laws::LawTable;
use acvus_mir::optimize::{dce, fold, reborrow, ssa_pass, while_to_for};
use acvus_mir::ty::{Effect, EffectTerm, Instances, ParamTerm, Poly, PolyTy, Ty, lift_to_poly};
use acvus_mir_test::{LoweredScript, lowered_script};
use acvus_utils::Interner;

struct Promoted {
    cfg: CfgBody,
    nest: LoopNest,
    invariants: Invariants,
    laws: LawTable,
}

impl Promoted {
    fn of(source: &str) -> Self {
        Self::with_externs(source, |_| Vec::new())
    }

    fn with_externs(source: &str, externs: impl FnOnce(&Interner) -> Vec<Function>) -> Self {
        Self::lowered(source, externs, Pass::Runs)
    }

    /// The same pipeline with `while_to_for` left out: the loop as the pass
    /// found it, after the same `dce`.
    fn unconverted(source: &str) -> Self {
        Self::lowered(source, |_| Vec::new(), Pass::Skipped)
    }

    fn lowered(
        source: &str,
        externs: impl FnOnce(&Interner) -> Vec<Function>,
        pass: Pass,
    ) -> Self {
        let i = Interner::new();
        let LoweredScript { module, laws } = lowered_script(&i, source, &externs(&i), vec![])
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let mut cfg = promote(module.main);
        ssa_pass::run(&mut cfg);
        fold::run(&mut cfg);
        reborrow::run(&mut cfg);
        if let Pass::Runs = pass {
            while_to_for::run(&i, &mut cfg, &laws);
        }
        dce::run(&mut cfg, &laws, &FunctionSummary::unknown());
        let invariants = Invariants::of(&cfg);
        let nest = LoopNest::of(&cfg, &DomTree::build(&cfg), &invariants);
        Self {
            cfg,
            nest,
            invariants,
            laws,
        }
    }

    fn sole_loop(&self) -> &Loop {
        let loops: Vec<&Loop> = self.nest.iter().map(|(_, l)| l).collect();
        let [only] = loops[..] else {
            panic!("one loop, found {}", loops.len());
        };
        only
    }

    fn range(&self, loop_: &Loop) -> Range {
        let LoopKind::For {
            source: ForSource::Range { at, hi },
        } = loop_.kind
        else {
            panic!("a range `for`, found {:?}", loop_.kind);
        };
        Range { at, hi }
    }

    fn carried_counter(&self, loop_: &Loop) -> ValueId {
        let affine = AffineValues::of(&self.cfg, loop_, &self.invariants, &self.laws);
        let found: Vec<ValueId> = self.cfg.blocks[loop_.natural.header.0]
            .params
            .iter()
            .copied()
            .filter(|p| {
                matches!(
                    affine.get(*p).map(|a| &a.derivation),
                    Some(Derivation::Carried { .. })
                )
            })
            .collect();
        let [i] = found[..] else {
            panic!("one carried counter among the header's parameters, found {found:?}");
        };
        i
    }

    fn body_block(&self, loop_: &Loop) -> BlockIdx {
        let Terminator::For { stages, .. } = &self.cfg.blocks[loop_.natural.header.0].terminator
        else {
            panic!("the header ends in `For`");
        };
        self.cfg.label_to_block[&stages.body()]
    }

    fn readers(&self, value: ValueId) -> Vec<BlockIdx> {
        self.cfg
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| {
                block
                    .insts
                    .iter()
                    .flat_map(|inst| inst_info::uses(&inst.kind))
                    .chain(inst_info::terminator_uses(&block.terminator))
                    .any(|used| used == value)
            })
            .map(|(b, _)| BlockIdx(b))
            .collect()
    }
}

struct Range {
    at: ValueId,
    hi: ValueId,
}

enum Pass {
    Runs,
    Skipped,
}

/// RFC-0094 rule 3's count: the constant it divides the distance by.
struct TripShape {
    divisor: i128,
}

impl Promoted {
    fn definition(&self, value: ValueId) -> &InstKind {
        self.cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .map(|inst| &inst.kind)
            .find(|kind| inst_info::defs(kind).contains(&value))
            .unwrap_or_else(|| panic!("an instruction defines {value:?}"))
    }

    fn literal(&self, value: ValueId) -> Option<i128> {
        let found = self
            .cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find(|inst| inst_info::defs(&inst.kind).contains(&value))?;
        match &found.kind {
            InstKind::Const { value, .. } => match value.desugared() {
                acvus_ast::Literal::Int(word) => Some(word),
                _ => None,
            },
            _ => None,
        }
    }

    /// The count `hi` joins: `(d − 1) / |s| + 1` on the edge where the
    /// counter has not passed the bound, zero on the other.
    fn trip_count_of(&self, hi: ValueId) -> TripShape {
        let join = self
            .cfg
            .blocks
            .iter()
            .position(|block| block.params == [hi])
            .unwrap_or_else(|| panic!("the count is the parameter of the block it joins at"));
        let label = self.cfg.blocks[join].label;
        let mut sent: Vec<ValueId> = self
            .cfg
            .blocks
            .iter()
            .filter_map(|block| match &block.terminator {
                Terminator::Jump { label: to, args } if *to == label => args.first().copied(),
                _ => None,
            })
            .collect();
        sent.sort_by_key(|value| self.literal(*value).is_some());
        let [counted, zero] = sent[..] else {
            panic!("two edges join the count: {sent:?}");
        };
        assert_eq!(self.literal(zero), Some(0), "one edge sends zero");
        let InstKind::BinOp {
            op: BinOp::Add(Overflow::Wrap),
            left: quotient,
            right: one,
            ..
        } = self.definition(counted)
        else {
            panic!("the other sends the quotient plus one: {:?}", self.definition(counted));
        };
        assert_eq!(self.literal(*one), Some(1));
        let InstKind::BinOp {
            op: BinOp::Div,
            right: divisor,
            ..
        } = self.definition(*quotient)
        else {
            panic!("one constant division: {:?}", self.definition(*quotient));
        };
        TripShape {
            divisor: self
                .literal(*divisor)
                .unwrap_or_else(|| panic!("the divisor is a word constant")),
        }
    }
}

fn assert_converted(source: &str) -> Promoted {
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    let Range { at, hi } = o.range(loop_);
    for bound in [at, hi] {
        assert_eq!(
            o.invariants.above(&loop_.natural, bound),
            Some(Invariant::Outside(bound)),
            "a range's bounds are settled above the header:\n{source}"
        );
    }

    let i = o.carried_counter(loop_);
    let body = o.body_block(loop_);
    let counter = o.cfg.blocks[body.0].params[0];
    assert_ne!(counter, i);
    assert!(
        o.readers(i).contains(&body),
        "the body still reads `i`, the header parameter it read before:\n{source}"
    );
    assert_eq!(
        o.readers(counter),
        [],
        "nothing reads the terminator's counter: no body instruction was \
         rewritten to it:\n{source}"
    );
    o
}

fn assert_declined(source: &str) {
    assert_still_a_while(&Promoted::of(source), source);
}

fn assert_still_a_while(o: &Promoted, source: &str) {
    let loop_ = o.sole_loop();
    assert!(
        matches!(loop_.kind, LoopKind::While),
        "declined, and still a `while`; found {:?}:\n{source}",
        loop_.kind
    );
}

#[test]
fn a_counted_while_is_a_range_for() {
    assert_converted("let n = 10; let s = 0; let i = 0; while i < n { s = s + i; i = i + 1; } s");
}

#[test]
fn i_read_after_the_loop_is_the_header_parameter_the_exit_reads() {
    let o = assert_converted(
        "let n = 10; let s = 0; let i = 0; while i < n { s = s + i; i = i + 1; } s + i",
    );
    let loop_ = o.sole_loop();
    let i = o.carried_counter(loop_);
    assert!(
        o.readers(i)
            .iter()
            .any(|block| !loop_.natural.contains(*block)),
        "after the loop the program reads `i` itself; no exit value was computed"
    );
}

#[test]
fn a_literal_bound_is_written_again_above_the_header() {
    assert_converted("let s = 0; let i = 0; while i < 10 { s = s + i; i = i + 1; } s");
}

#[test]
fn n_greater_than_i_is_the_same_condition() {
    assert_converted("let n = 10; let s = 0; let i = 0; while n > i { s = s + i; i = i + 1; } s");
}

#[test]
fn a_while_nested_in_a_for_is_converted_with_the_outer_counter_as_its_bound() {
    let o = Promoted::of(
        "let s = 0; for j in 0..5 { let i = 0; while i < j { s = s + i; i = i + 1; } } s",
    );
    let loops: Vec<&Loop> = o.nest.iter().map(|(_, l)| l).collect();
    assert_eq!(loops.len(), 2);
    assert!(
        loops.iter().all(|l| matches!(
            l.kind,
            LoopKind::For {
                source: ForSource::Range { .. }
            }
        )),
        "both loops are range `for`s: {:?}",
        loops.iter().map(|l| l.kind).collect::<Vec<_>>()
    );
}

#[test]
fn less_or_equal_with_the_bound_below_the_maximum_is_the_range_to_one_past_it() {
    let source = "let n = 10; let s = 0; let i = 0; while i <= n { s = s + i; i = i + 1; } s";
    let o = Promoted::of(source);
    let Range { at, hi } = o.range(o.sole_loop());
    assert_eq!(o.literal(at), Some(0), "the range starts at `b`:\n{source}");
    let past = o.definition(hi);
    assert!(
        matches!(
            past,
            InstKind::BinOp { op: BinOp::Add(Overflow::Wrap), left, right, .. }
                if o.literal(*left) == Some(10) && o.literal(*right) == Some(1)
        ),
        "the range ends at `n + 1`, the pass's own wrapping add: {past:?}"
    );
}

#[test]
fn less_or_equal_with_the_bound_at_the_maximum_is_declined() {
    assert_declined("let s = 0; let i = 250u8; while i <= 255u8 { s = s + 1; i = i + 1u8; } s");
}

#[test]
fn less_or_equal_with_a_bound_of_unknown_range_is_declined() {
    assert_declined(&format!(
        "{UNFOLDED_N_AND_M} let s = 0; let i = 0u64; while i <= n {{ s = s + 1; i = i + 1u64; }} s"
    ));
}

#[test]
fn counting_down_by_one_is_the_range_from_the_bound_to_the_start() {
    let source = "let s = 0; let i = 9; while i > 2 { i = i - 1; s = s + i; } s";
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    let Range { at, hi } = o.range(loop_);
    assert_eq!(
        (o.literal(at), o.literal(hi)),
        (Some(2), Some(9)),
        "`Range {{ at: n, hi: b }}`:\n{source}"
    );
    let affine = AffineValues::of(&o.cfg, loop_, &o.invariants, &o.laws);
    let counting_down = o.cfg.blocks[loop_.natural.header.0]
        .params
        .iter()
        .filter(|param| {
            matches!(
                affine.get(**param).map(|found| &found.derivation),
                Some(Derivation::CountsDown { .. })
            )
        })
        .count();
    assert_eq!(
        counting_down, 1,
        "`i` stays its own header parameter, `{{b, −1}}`"
    );
}

#[test]
fn counting_down_by_two_without_a_word_constant_step_is_declined() {
    assert_declined("let s = 0; let i = 9u64; while i > 2u64 { i = i - 2u64; s = s + 1; } s");
}

#[test]
fn a_step_of_two_is_a_range_over_its_trip_count() {
    let source = "let n = 10; let s = 0; let i = 0; while i < n { s = s + i; i = i + 2; } s";
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    let Range { at, hi } = o.range(loop_);
    assert_eq!(o.literal(at), Some(0), "the count starts at zero:\n{source}");
    assert_eq!(
        o.cfg.val_types[&hi],
        Ty::U64,
        "the count is read unsigned at the width"
    );
    let trip = o.trip_count_of(hi);
    assert_eq!(
        trip.divisor, 2,
        "one constant division by the step's magnitude"
    );
}

#[test]
fn a_negative_step_counting_down_is_a_range_over_its_trip_count() {
    let o = Promoted::of("let s = 0; let i = 10; while i > 0 { s = s + i; i = i + -3; } s");
    let Range { hi, .. } = o.range(o.sole_loop());
    assert_eq!(o.trip_count_of(hi).divisor, 3);
}

#[test]
fn a_step_moving_away_from_the_bound_is_declined() {
    assert_declined("let s = 0; let i = 0; while i < 10 { s = s + 1; i = i + -2; } s");
}

#[test]
fn a_step_of_zero_is_declined() {
    assert_declined("let s = 0; let i = 0; while i < 10 { s = s + 1; i = i + 0; if s > 5 { i = 10; }; } s");
}

#[test]
fn an_offset_compare_is_the_range_from_the_first_visit_s_sum() {
    let source =
        "let n = 10; let s = 0; let i = 0; while i + 1 < n { s = s + i; i = i + 1; } s";
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    let Range { at, .. } = o.range(loop_);
    let start = o.definition(at);
    assert!(
        matches!(start, InstKind::BinOp { op: BinOp::Add(_), .. }),
        "the range starts at `b + c₀`, the header's own step moved to the entry: {start:?}"
    );
    assert!(
        o.cfg.blocks[loop_.natural.header.0].insts.is_empty(),
        "the header holds no instruction:\n{source}"
    );
    let body = o.body_block(loop_);
    assert!(
        o.cfg.blocks[body.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::BinOp { op: BinOp::Add(Overflow::Trap), .. })),
        "the program's `i + 1` runs at the body's head"
    );
}

#[test]
fn an_offset_compare_whose_offset_is_no_word_is_declined() {
    assert_declined(&format!(
        "{UNFOLDED_N_AND_M} let s = 0; let i = 0u64; while i + m < n {{ s = s + 1; i = i + 1u64; }} s"
    ));
}

#[test]
fn a_bound_the_body_writes_is_declined() {
    assert_declined(
        "let n = 10; let s = 0; let i = 0; while i < n { s = s + i; n = n - 1; i = i + 1; } s",
    );
}

// -- An exit from the body (RFC-0094 rule 7) ------------------------------

/// The `For`'s own exit block and the block it jumps to, which the body's
/// edges out of the loop reach too.
struct SharedExit {
    own: BlockIdx,
    joined: BlockIdx,
}

impl Promoted {
    fn shared_exit(&self, loop_: &Loop) -> SharedExit {
        let Terminator::For { exit, exit_args, .. } =
            &self.cfg.blocks[loop_.natural.header.0].terminator
        else {
            panic!("the header ends in `For`");
        };
        assert_eq!(exit_args, &[], "the `For` leaves through a block of its own");
        let own = self.cfg.label_to_block[exit];
        assert_eq!(
            self.cfg.predecessors()[&own][..],
            [loop_.natural.header],
            "the `For`'s own exit block is entered from the header alone"
        );
        let Terminator::Jump { label, .. } = &self.cfg.blocks[own.0].terminator else {
            panic!("the `For`'s exit block jumps to the block the body leaves for");
        };
        SharedExit {
            own,
            joined: self.cfg.label_to_block[label],
        }
    }

    /// The blocks other than the `For`'s own exit that enter `joined`.
    fn edges_from_the_body(&self, exit: &SharedExit) -> Vec<BlockIdx> {
        self.cfg.predecessors()[&exit.joined]
            .iter()
            .copied()
            .filter(|pred| *pred != exit.own)
            .collect()
    }

    fn jump_into(&self, from: BlockIdx, to: BlockIdx) -> Vec<ValueId> {
        let label = self.cfg.blocks[to.0].label;
        match &self.cfg.blocks[from.0].terminator {
            Terminator::Jump { label: target, args } if *target == label => args.clone(),
            other => panic!("{from:?} jumps to {label:?}, found {other:?}"),
        }
    }
}

const BROKEN_COUNT: &str = "let n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].len() as i64; \
     let s = 0; let r = 0; let i = 0; \
     while i < n { if s > 20 { r = s * 2; break; }; s = s + i; i = i + 1; } r + s + i";

#[test]
fn a_break_in_a_counted_while_is_converted_with_its_edge_and_arguments_kept() {
    let o = Promoted::of(BROKEN_COUNT);
    let loop_ = o.sole_loop();
    o.range(loop_);
    let exit = o.shared_exit(loop_);
    let [break_block] = o.edges_from_the_body(&exit)[..] else {
        panic!("one edge from the body joins the exit:\n{BROKEN_COUNT}");
    };
    let before = Promoted::unconverted(BROKEN_COUNT);
    let before_loop = before.sole_loop();
    let Terminator::JumpIf {
        else_label,
        else_args,
        ..
    } = &before.cfg.blocks[before_loop.natural.header.0].terminator
    else {
        panic!("the unconverted header branches");
    };
    let joined = before.cfg.label_to_block[else_label];
    assert_eq!(
        o.cfg.blocks[exit.joined.0].label, *else_label,
        "the block the header left for is the one the `For`'s exit jumps to"
    );
    assert_eq!(
        o.jump_into(exit.own, exit.joined),
        *else_args,
        "the `For`'s exit sends what the header's exit edge sent"
    );
    let break_label = o.cfg.blocks[break_block.0].label;
    let before_break = before.cfg.label_to_block[&break_label];
    assert_eq!(
        o.jump_into(break_block, exit.joined),
        before.jump_into(before_break, joined),
        "the edge out of the body keeps its arguments:\n{BROKEN_COUNT}"
    );
}

#[test]
fn a_return_in_a_counted_while_is_declined() {
    assert_declined(
        "let n = [1, 2, 3].len() as i64; let s = 0; let i = 0; \
         while i < n { if s > 20 { return s; }; s = s + i; i = i + 1; } s + i",
    );
}

// -- A back edge whose arguments decide the test (RFC-0094 rule 6) -------

/// S11 of the parallel-loop corpus, whose found arm also moves `i`, so the
/// value it sends the exit is not the header's.
const SEARCH: &str = "let xs = vec([5, 3, 9, -1, 9]); let i = 0u64; let found = false; \
     while i < xs.len() && !found { \
         if xs[i] < 0 { found = true; i = i + 100u64; } else { i = i + 1u64; }; \
     } i";

#[test]
fn a_back_edge_whose_flag_decides_the_test_goes_to_the_exit() {
    let o = Promoted::of(SEARCH);
    let loop_ = o.sole_loop();
    o.range(loop_);
    let header = &o.cfg.blocks[loop_.natural.header.0];
    assert!(
        header
            .params
            .iter()
            .all(|param| o.cfg.val_types[param] != Ty::Bool),
        "the flag is folded, and no header parameter carries it:\n{SEARCH}"
    );
    let exit = o.shared_exit(loop_);
    let i = o.carried_counter(loop_);
    assert_eq!(
        o.jump_into(exit.own, exit.joined),
        [i],
        "the `For`'s exit sends `i`, which the exit read"
    );
    let [found_arm] = o.edges_from_the_body(&exit)[..] else {
        panic!("the found arm's edge goes to the exit:\n{SEARCH}");
    };
    let [sent] = o.jump_into(found_arm, exit.joined)[..] else {
        panic!("the found arm sends the exit one value");
    };
    let InstKind::BinOp {
        op: BinOp::Add(_),
        left,
        right,
        ..
    } = o.definition(sent)
    else {
        panic!("the found arm sends the `i + 100` it sent the header, found {:?}", o.definition(sent));
    };
    assert_eq!((*left, o.literal(*right)), (i, Some(100)));
}

fn pure_scale(i: &Interner) -> Vec<Function> {
    vec![Function {
        qref: QualifiedRef::root(i.intern("pure_scale")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty: PolyTy::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: EffectTerm::<Poly>::Known(Effect::PURE),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }]
}

fn opaque_and_pure_scale(i: &Interner) -> Vec<Function> {
    opaque(i).into_iter().chain(pure_scale(i)).collect()
}

fn flag_search(bound: &str) -> String {
    format!(
        "{WORD_N_AND_M} let d = opaque(m); let found = false; let i = 0; \
         while i < {bound} && !found {{ if i * i > 7 {{ found = true; }} else {{ i = i + 1; }}; }} i"
    )
}

/// The control: the same search whose bound skips nothing that can trap.
#[test]
fn a_flag_search_over_a_bound_that_cannot_trap_is_converted() {
    let source = flag_search("n");
    let o = Promoted::with_externs(&source, opaque_and_pure_scale);
    o.range(o.sole_loop());
}

#[test]
fn a_flag_whose_edge_skips_a_division_by_a_parameter_is_declined() {
    let source = flag_search("n / d");
    assert_still_a_while(&Promoted::with_externs(&source, opaque_and_pure_scale), &source);
}

#[test]
fn a_flag_whose_edge_skips_a_pure_call_not_declared_total_is_declined() {
    let source = flag_search("pure_scale(n)");
    assert_still_a_while(&Promoted::with_externs(&source, opaque_and_pure_scale), &source);
}

/// `found` true leaves `!found || i < 3` to `i < 3`, so the edge that sets
/// it decides nothing; with `found` false the chain is `i < n`, which the
/// ranges read.
#[test]
fn a_flag_whose_constant_does_not_decide_the_chain_is_declined() {
    assert_declined(
        "let n = [1, 2, 3, 4, 5].len() as i64; let found = false; let i = 0; \
         while i < n && (!found || i < 3) { if i == 2 { found = true; } else { i = i + 1; }; } i",
    );
}

#[test]
fn a_flag_another_edge_sets_to_a_test_is_declined() {
    assert_declined(
        "let n = [1, 2, 3, 4, 5].len() as i64; let found = false; let i = 0; \
         while i < n && !found { \
             if i == 2 { found = true; } else { found = i > 3; i = i + 1; }; \
         } i",
    );
}

#[test]
fn a_condition_on_a_value_that_is_not_an_induction_variable_is_declined() {
    assert_declined("let n = 100; let x = 1; while x < n { x = x * 2; } x");
}

fn snapshot(cfg: &CfgBody) -> Vec<BlockText> {
    cfg.blocks
        .iter()
        .map(|block| BlockText {
            params: block.params.clone(),
            insts: block
                .insts
                .iter()
                .map(|inst| format!("{:?}", inst.kind))
                .collect(),
            terminator: format!("{:?}", block.terminator),
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct BlockText {
    params: Vec<ValueId>,
    insts: Vec<String>,
    terminator: String,
}

fn assert_computation_alone_moves(source: &str, added: &[&str]) {
    let i = Interner::new();
    let LoweredScript { module, laws } =
        lowered_script(&i, source, &[], vec![]).unwrap_or_else(|e| panic!("{e}"));
    let mut cfg = promote(module.main);
    ssa_pass::run(&mut cfg);
    dce::run(&mut cfg, &laws, &FunctionSummary::unknown());
    let before = snapshot(&cfg);
    let header = {
        let invariants = Invariants::of(&cfg);
        let nest = LoopNest::of(&cfg, &DomTree::build(&cfg), &invariants);
        let [(_, loop_)] = nest.iter().collect::<Vec<_>>()[..] else {
            panic!("one loop");
        };
        loop_.natural.header
    };

    while_to_for::run(&i, &mut cfg, &laws);
    let after = snapshot(&cfg);

    let Terminator::For { stages, exit, .. } = &cfg.blocks[header.0].terminator else {
        panic!(
            "the header ends in `For`: {:?}",
            cfg.blocks[header.0].terminator
        );
    };
    let body = cfg.label_to_block[&stages.body()];
    let exit = cfg.label_to_block[exit];
    let mut grew = Vec::new();
    let mut body_head: Vec<String> = Vec::new();
    let mut exit_head: Vec<String> = Vec::new();
    for (b, (was, is)) in before.iter().zip(&after).enumerate() {
        if b == header.0 {
            assert_eq!(was.params, is.params);
            assert!(is.insts.is_empty(), "a `for` header holds no instruction");
            continue;
        }
        assert_eq!(was.terminator, is.terminator, "block {b}'s terminator");
        match b == body.0 {
            true => assert_eq!(
                was.params[..],
                is.params[1..],
                "the body gains one leading parameter"
            ),
            false => assert_eq!(was.params, is.params, "block {b}'s parameters"),
        }
        if was.insts == is.insts {
            continue;
        }
        if b == body.0 || b == exit.0 {
            let (head, rest) = is.insts.split_at(is.insts.len() - was.insts.len());
            assert_eq!(rest, &was.insts[..], "block {b} only gained at its head");
            match b == body.0 {
                true => body_head = head.to_vec(),
                false => exit_head = head.to_vec(),
            }
        } else {
            assert_eq!(
                was.insts[..],
                is.insts[..was.insts.len()],
                "block {b} only gained"
            );
            grew.push(is.insts[was.insts.len()..].to_vec());
        }
    }
    let grew_insts = grew.concat();
    let lost = &before[header.0].insts;
    let (to_entry, to_heads): (Vec<&String>, Vec<&String>) =
        lost.iter().partition(|inst| grew_insts.contains(inst));
    assert_eq!(
        grew_insts.iter().collect::<Vec<_>>(),
        to_entry,
        "what the entering block gained the header lost, in order"
    );
    assert_eq!(
        body_head.iter().collect::<Vec<_>>(),
        to_heads,
        "the rest of the header runs at the body's head, in order"
    );
    assert_eq!(
        kinds(&exit_head),
        kinds(&body_head),
        "the exit's head runs the same instructions"
    );
    assert_eq!(
        kinds(&grew_insts),
        added,
        "the bound's computation moves and nothing else"
    );
}

fn kinds(insts: &[String]) -> Vec<&str> {
    insts
        .iter()
        .map(|inst| {
            inst.split(|c: char| !c.is_alphanumeric())
                .next()
                .expect("a `Debug` of an instruction starts with its kind")
        })
        .collect()
}

#[test]
fn the_pass_rewrites_the_terminator_alone() {
    assert_computation_alone_moves(
        "let n = 10; let s = 0; let i = 0; while i < n { s = s + i; i = i + 1; } s + i",
        &[],
    );
}

#[test]
fn a_literal_bound_moves_and_nothing_else() {
    assert_computation_alone_moves(
        "let s = 0; let i = 0; while i < 10 { s = s + i; i = i + 1; } s + i",
        &["Const"],
    );
}

const UNFOLDED_N_AND_M: &str = "let n = [1, 2, 3, 4, 5].len(); let m = [1, 2, 3].len();";

#[test]
fn a_bound_the_header_computes_from_invariant_operands_is_a_range_for() {
    for bound in ["n * 2", "n + m", "(n - 1) * m"] {
        assert_converted(&format!(
            "{UNFOLDED_N_AND_M} let s = 0; let i = 0; while i < {bound} {{ s = s + i; i = i + 1; }} s"
        ));
    }
}

#[test]
fn a_computed_bound_nested_in_a_for_reads_the_outer_counter() {
    let o = Promoted::of(
        "let s = 0; for j in 0..5 { let i = 0; while i < j * 2 + 1 { s = s + i; i = i + 1; } } s",
    );
    assert!(
        o.nest.iter().all(|(_, l)| matches!(
            l.kind,
            LoopKind::For {
                source: ForSource::Range { .. }
            }
        )),
        "both loops are range `for`s: {:?}",
        o.nest.iter().map(|(_, l)| l.kind).collect::<Vec<_>>()
    );
}

#[test]
fn a_computed_bound_over_an_operand_the_loop_writes_is_declined() {
    assert_declined(&format!(
        "{UNFOLDED_N_AND_M} let s = 0; let i = 0; \
         while i < n * 2 {{ s = s + i; n = n - 1; i = i + 1; }} s"
    ));
}

#[test]
fn a_len_bound_over_a_vector_the_loop_writes_is_declined() {
    assert_declined(
        "let v = vec([1]); let s = 0; let i = 0; \
         while i < v.len() { if i < 3 { v.push(i); }; s = s + i; i = i + 1; } s",
    );
}

#[test]
fn the_pass_moves_the_computation_of_the_bound_and_nothing_else() {
    assert_computation_alone_moves(
        &format!(
            "{UNFOLDED_N_AND_M} let s = 0; let i = 0; while i < n * 2 + m {{ s = s + i; i = i + 1; }} s + i"
        ),
        &["Const", "BinOp", "BinOp"],
    );
}

#[test]
fn a_bound_that_divides_is_a_range_for() {
    for bound in ["n / 2", "n % m", "n / m + n % m"] {
        assert_converted(&format!(
            "{UNFOLDED_N_AND_M} let s = 0; let i = 0; while i < {bound} {{ s = s + i; i = i + 1; }} s"
        ));
    }
}

#[test]
fn a_len_bound_over_a_slice_made_above_the_loop_is_a_range_for() {
    assert_converted(
        "let v = [1, 2, 3]; let s = v.as_slice(); let t = 0; let i = 0; \
         while i < s.len() { t = t + i; i = i + 1; } t",
    );
}

#[test]
fn a_len_bound_over_a_reference_made_above_the_loop_is_a_range_for() {
    assert_converted(
        "let v = [1, 2, 3]; let r = &v; let t = 0; let i = 0; \
         while i < r.len() { t = t + i; i = i + 1; } t",
    );
}

#[test]
fn a_pure_call_combined_with_word_operations_is_a_range_for() {
    assert_converted(&format!(
        "{UNFOLDED_N_AND_M} let v = [1, 2, 3]; let s = v.as_slice(); let t = 0; let i = 0; \
         while i < s.len() * 2 + n {{ t = t + i; i = i + 1; }} t"
    ));
}

#[test]
fn a_pure_call_moves_in_the_order_of_the_header() {
    assert_computation_alone_moves(
        "let v = [1, 2, 3]; let s = v.as_slice(); let t = 0; let i = 0; \
         while i < s.len() / 2 { t = t + i; i = i + 1; } t + i",
        &["FunctionCall", "Const", "BinOp"],
    );
}

/// The header makes `&v` on every visit, so the receiver is not defined
/// outside the loop, and RFC-0094 rule 5 makes it a step.
#[test]
fn a_len_bound_over_a_receiver_the_header_makes_is_a_range_for() {
    assert_converted(
        "let v = [1, 2, 3]; let s = 0; let i = 0; \
         while i < v.len() { s = s + i; i = i + 1; } s",
    );
}

#[test]
fn a_reference_a_step_computes_is_declined_as_an_argument() {
    assert_declined(
        "let v = [1, 2, 3]; let r = &v; let t = 0; let i = 0; \
         while i < r.as_slice().len() { t = t + i; i = i + 1; } t",
    );
}

#[test]
fn a_pure_call_given_a_mutable_reference_is_declined() {
    assert_declined(
        "let v = vec([5, 6, 7]); let r = &mut v; let t = 0; let i = 0; \
         while i < r.remove(0) { t = t + i; i = i + 1; } t",
    );
}

fn opaque(i: &Interner) -> Vec<Function> {
    vec![Function {
        qref: QualifiedRef::root(i.intern("opaque")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty: PolyTy::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: EffectTerm::<Poly>::Known(Effect::OPAQUE),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }]
}

const WORD_N_AND_M: &str = "let n = [1, 2, 3, 4, 5].len() as i64; let m = [1, 2, 3].len() as i64;";

#[test]
fn a_bound_that_calls_an_extern_not_declared_pure_is_declined() {
    let source = format!(
        "{WORD_N_AND_M} let s = 0; let i = 0; while i < opaque(n) {{ s = s + i; i = i + 1; }} s"
    );
    assert_still_a_while(&Promoted::with_externs(&source, opaque), &source);
}

#[test]
fn an_effect_before_a_step_that_traps_is_declined() {
    let source = format!(
        "{WORD_N_AND_M} let d = 0; let s = 0; let i = 0; \
         while i < {{ d = opaque(n); n / m }} {{ s = s + i + d; i = i + 1; }} s"
    );
    assert_still_a_while(&Promoted::with_externs(&source, opaque), &source);
}

#[test]
fn a_trap_before_a_step_that_traps_is_declined() {
    assert_declined(&format!(
        "{UNFOLDED_N_AND_M} let d = 0; let s = 0; let i = 0; \
         while i < {{ d = n / m; n % m }} {{ s = s + i + d; i = i + 1; }} s"
    ));
}

/// A bound with no step that traps moves ahead of a call. A product is not
/// such a bound: the program's `*` traps (RFC-0037 rule 3), and the call
/// before it can trap too, so that bound is declined.
#[test]
fn an_effect_before_a_bound_that_cannot_trap_is_no_obstacle() {
    let source = format!(
        "{WORD_N_AND_M} let d = 0; let s = 0; let i = 0; \
         while i < {{ d = opaque(n); 10 }} {{ s = s + i + d; i = i + 1; }} s"
    );
    let o = Promoted::with_externs(&source, opaque);
    let loop_ = o.sole_loop();
    o.range(loop_);
}

#[test]
fn a_call_before_a_product_is_declined() {
    let source = format!(
        "{WORD_N_AND_M} let d = 0; let s = 0; let i = 0; \
         while i < {{ d = opaque(n); n * m }} {{ s = s + i + d; i = i + 1; }} s"
    );
    assert_still_a_while(&Promoted::with_externs(&source, opaque), &source);
}

// -- A header instruction that is no step (RFC-0081 rule 2) -------------

fn exit_block(o: &Promoted, loop_: &Loop) -> BlockIdx {
    let Terminator::For { exit, .. } = &o.cfg.blocks[loop_.natural.header.0].terminator else {
        panic!("the header ends in `For`");
    };
    o.cfg.label_to_block[exit]
}

fn binops(o: &Promoted, block: BlockIdx) -> Vec<(BinOp, ValueId)> {
    o.cfg.blocks[block.0]
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::BinOp { op, dst, .. } => Some((*op, *dst)),
            _ => None,
        })
        .collect()
}

fn call_args(o: &Promoted, block: BlockIdx) -> Vec<Vec<ValueId>> {
    o.cfg.blocks[block.0]
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::FunctionCall { args, .. } => Some(args.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_header_instruction_that_is_no_step_runs_at_the_heads_of_the_body_and_the_exit() {
    let source = "let n = [1, 2, 3, 4, 5].len() as i64; let i = 0; \
                  while { let z = 10 / (7 - i); i < n } { i = i + 1; } i";
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    o.range(loop_);
    assert!(
        o.cfg.blocks[loop_.natural.header.0].insts.is_empty(),
        "a `for` header holds no instruction:\n{source}"
    );
    let body = binops(&o, o.body_block(loop_));
    let exit = binops(&o, exit_block(&o, loop_));
    let ops = |found: &[(BinOp, ValueId)]| found.iter().map(|(op, _)| *op).collect::<Vec<_>>();
    let header_order = [BinOp::Sub(Overflow::Trap), BinOp::Div];
    assert_eq!(ops(&body[..2]), header_order, "the body's head:\n{source}");
    assert_eq!(ops(&exit), header_order, "the exit's head:\n{source}");
    assert_ne!(
        body[1].1, exit[1].1,
        "the exit's copy defines its own value"
    );
}

#[test]
fn an_effect_before_a_bound_runs_at_the_heads_of_the_body_and_the_exit_in_the_header_s_order() {
    let source = format!(
        "{WORD_N_AND_M} let d = 0; let e = 0; let s = 0; let i = 0; \
         while i < {{ d = opaque(n); e = opaque(m); 10 }} {{ s = s + i + d + e; i = i + 1; }} s"
    );
    let o = Promoted::with_externs(&source, opaque);
    let loop_ = o.sole_loop();
    o.range(loop_);
    let before = Promoted::lowered(&source, opaque, Pass::Skipped);
    let header_calls = call_args(&before, before.sole_loop().natural.header);
    assert_eq!(header_calls.len(), 2, "two calls in the header:\n{source}");
    assert_eq!(call_args(&o, o.body_block(loop_)), header_calls, "{source}");
    assert_eq!(
        call_args(&o, exit_block(&o, loop_)),
        header_calls,
        "{source}"
    );
}

#[test]
fn a_header_value_read_where_the_exit_and_a_break_meet_is_a_parameter_there() {
    let source = format!(
        "{WORD_N_AND_M} let d = 0; let i = 0; \
         while {{ d = d + 1; i < n }} {{ if i == 2 {{ break; }}; i = i + 1; }} i * 10 + d"
    );
    let before = Promoted::lowered(&source, |_| Vec::new(), Pass::Skipped);
    let before_loop = before.sole_loop();
    let [(_, counted)] = binops(&before, before_loop.natural.header)[..1] else {
        panic!("the header counts `d` first:\n{source}");
    };
    let join = before
        .readers(counted)
        .into_iter()
        .find(|block| !before_loop.natural.contains(*block));
    assert!(
        join.is_some(),
        "the block after the loop reads the header's `d + 1`:\n{source}"
    );

    let o = Promoted::of(&source);
    let loop_ = o.sole_loop();
    o.range(loop_);
    let own_exit = exit_block(&o, loop_);
    let [(BinOp::Add(_), copy)] = binops(&o, own_exit)[..] else {
        panic!("the exit's own block counts `d`:\n{source}");
    };
    let Terminator::Jump { label, args } = &o.cfg.blocks[own_exit.0].terminator else {
        panic!("the exit's own block jumps to the exit");
    };
    assert!(args.contains(&copy), "the exit sends its copy:\n{source}");
    let [(BinOp::Add(_), counted)] = binops(&o, o.body_block(loop_))[..1] else {
        panic!("the body counts `d` first:\n{source}");
    };
    let joined = o.cfg.label_to_block[label];
    assert!(
        !o.readers(counted).contains(&joined),
        "the join reads a parameter, not the body's copy:\n{source}"
    );
}

// -- Pull loops (RFC-0089 rule 1) ---------------------------------------

fn header_of(o: &Promoted) -> &Terminator {
    &o.cfg.blocks[o.sole_loop().natural.header.0].terminator
}

fn assert_a_pull(source: &str) -> Promoted {
    let o = Promoted::of(source);
    let loop_ = o.sole_loop();
    assert!(
        matches!(loop_.kind, LoopKind::While) && loop_.trip == Trip::Unknown,
        "a pull loop is a `While` in the nest with an unknown trip:\n{source}"
    );
    let Terminator::While { stages, .. } = header_of(&o) else {
        panic!("the header ends in `While`: {:?}\n{source}", header_of(&o));
    };
    assert_eq!(stages.len(), 1, "the lowered chain is the body alone");
    o
}

fn assert_a_plain_branch_loop(source: &str) {
    let o = Promoted::of(source);
    assert!(
        matches!(header_of(&o), Terminator::JumpIf { .. }),
        "the loop stays a plain branch loop: {:?}\n{source}",
        header_of(&o)
    );
}

#[test]
fn a_pull_over_an_owning_iterator_is_a_while_terminator() {
    assert_a_pull(
        "let xs = vec([3, 4]); let it = xs.into_iter(); let s = 0; \
         while let Some(x) = it.next() { s = s + x; } s",
    );
}

#[test]
fn a_pull_whose_body_reads_its_payload_by_value_is_a_while_terminator() {
    assert_a_pull(
        "let text = \"abc\"; let cs = text.chars(); let n = 0; \
         while let Some(c) = cs.next() { n = n + (c as i64); } n",
    );
}

/// `Refs::next` states its result at the collection's lifetime, not the
/// `&mut` it takes (RFC-0096 rules 1 and 4), so the payload holds the
/// vector's loan and reading it touches no pulled storage.
#[test]
fn a_pull_whose_payload_borrows_the_collection_is_a_while_terminator() {
    assert_a_pull(
        "let xs = vec([3, 4]); let it = xs.as_iter(); let s = 0; \
         while let Some(x) = it.next() { s = s + *x; } s",
    );
}

#[test]
fn a_pull_by_an_extern_other_than_next_stays_a_plain_branch_loop() {
    assert_a_plain_branch_loop(
        "let v = vec([3, 4]); let s = 0; while let Some(x) = v.pop() { s = s + x; } s",
    );
}

#[test]
fn a_pull_loop_the_body_leaves_stays_a_plain_branch_loop() {
    assert_a_plain_branch_loop(
        "let xs = vec([3, 4]); let it = xs.into_iter(); let s = 0; \
         while let Some(x) = it.next() { if x > 3 { break; }; s = s + x; } s",
    );
}

#[test]
fn a_body_that_pulls_again_stays_a_plain_branch_loop() {
    assert_a_plain_branch_loop(
        "let xs = vec([3, 4, 5]); let it = xs.into_iter(); let s = 0; \
         while let Some(x) = it.next() { let y = it.next(); s = s + x; } s",
    );
}
