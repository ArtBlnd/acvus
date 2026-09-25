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
        let i = Interner::new();
        let LoweredScript { module, laws } = lowered_script(&i, source, &externs(&i), vec![])
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let mut cfg = promote(module.main);
        ssa_pass::run(&mut cfg);
        fold::run(&mut cfg);
        reborrow::run(&mut cfg);
        while_to_for::run(&i, &mut cfg, &laws);
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

#[test]
fn a_break_is_declined() {
    assert_declined(
        "let n = 10; let s = 0; let i = 0; \
         while i < n { if s > 20 { break; }; s = s + i; i = i + 1; } s",
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

    let Terminator::For { stages, .. } = &cfg.blocks[header.0].terminator else {
        panic!(
            "the header ends in `For`: {:?}",
            cfg.blocks[header.0].terminator
        );
    };
    let body = cfg.label_to_block[&stages.body()];
    let mut grew = Vec::new();
    let mut moved = Vec::new();
    for (b, (was, is)) in before.iter().zip(&after).enumerate() {
        if b == header.0 {
            assert_eq!(was.params, is.params);
            let (kept, left): (Vec<&String>, Vec<&String>) =
                was.insts.iter().partition(|inst| is.insts.contains(inst));
            assert_eq!(kept, is.insts.iter().collect::<Vec<_>>(), "the header only lost");
            moved = left.into_iter().cloned().collect();
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
        if was.insts != is.insts {
            assert_eq!(
                was.insts[..],
                is.insts[..was.insts.len()],
                "block {b} only gained"
            );
            grew.push(is.insts[was.insts.len()..].to_vec());
        }
    }
    assert_eq!(
        grew.concat(),
        moved,
        "what the header lost is what the entering block gained, in order"
    );
    let grew: Vec<&str> = grew
        .iter()
        .flatten()
        .map(|inst| {
            inst.split(|c: char| !c.is_alphanumeric())
                .next()
                .expect("a `Debug` of an instruction starts with its kind")
        })
        .collect();
    assert_eq!(
        grew, added,
        "the bound's computation moves and nothing else"
    );
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

/// `Refs::next` declares its result at the lifetime of the `&mut` it takes,
/// so the payload holds a loan on the iterator's own storage and reading it
/// touches the pulled storage outside the header.
#[test]
fn a_pull_whose_payload_borrows_the_iterator_stays_a_plain_branch_loop() {
    assert_a_plain_branch_loop(
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
