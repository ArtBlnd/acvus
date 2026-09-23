//! What `optimize::iv_canon` leaves (RFC-0066 rule 7), and the trip count a
//! `for`'s exit edge defines for it (RFC-0057 rule 9).
//!
//! Every program here is compiled at both levels, and each one also stands
//! in `acvus-interpreter-test/tests/soundness/iv-canon/`, where it runs at
//! both levels to its pinned value. This file says what shape the full
//! level gives it; that corpus says the shape computes the same number.

use acvus_ast::{Literal, Span};
use acvus_mir::ir::BinOp;
use acvus_mir::analysis::affine::{AffineValues, for_body};
use acvus_mir::analysis::carried::{Carried, CarriedState, MergeOp, Strength};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loans::{Loans, Summaries};
use acvus_mir::analysis::loops::{Invariants, Loop, LoopKind, LoopNest};
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{ExitTrip, ForSource, Inst, InstKind, Label, MirBody, MirModule, ValueId};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{CastTy, IntTy, Ty};
use acvus_mir::validate::type_check::{ValidationErrorKind, check_types};
use acvus_mir::laws::LawTable;
use acvus_mir_test::compile_script_at;
use acvus_utils::{Astr, Interner, LocalFactory};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner) -> FxHashMap<Astr, Ty> {
    ["base", "n", "m"]
        .iter()
        .map(|n| (i.intern(n), Ty::I64))
        .collect()
}

struct Compiled {
    interner: Interner,
    listing: String,
    cfg: CfgBody,
    nest: LoopNest,
    invariants: Invariants,
    loans: Loans,
    laws: LawTable,
}

impl Compiled {
    fn of(source: &str, opt: Opt) -> Self {
        let interner = Interner::new();
        let compiled = compile_script_at(&interner, source, &ctx(&interner), opt)
            .unwrap_or_else(|e| panic!("{source} at {opt:?}\n{e}"));
        let module = compiled.module;
        let listing = dump_with(&interner, &module);
        let cfg = promote(module.main);
        let invariants = Invariants::of(&cfg);
        let nest = LoopNest::of(&cfg, &DomTree::build(&cfg), &invariants);
        let loans = Loans::build(&cfg, Summaries::NONE);
        Self {
            interner,
            listing,
            cfg,
            nest,
            invariants,
            loans,
            laws: compiled.laws,
        }
    }

    fn loops_by_header(&self) -> Vec<&Loop> {
        let mut loops: Vec<&Loop> = self.nest.iter().map(|(_, l)| l).collect();
        loops.sort_by_key(|l| l.natural.header.0);
        loops
    }

    fn state(&self, loop_: &Loop) -> CarriedState {
        let affine = AffineValues::of(&self.cfg, loop_, &self.invariants);
        CarriedState::of(&self.cfg, loop_, &affine, &self.loans, &self.laws)
    }

    fn carried(&self, loop_: &Loop) -> Vec<Carried> {
        self.state(loop_).params.iter().map(|p| p.carried).collect()
    }

    fn ivs(&self, loop_: &Loop) -> usize {
        self.carried(loop_)
            .iter()
            .filter(|c| **c == Carried::Iv)
            .count()
    }

    fn exit_trip(&self, loop_: &Loop) -> ExitTrip {
        match &self.cfg.blocks[loop_.natural.header.0].terminator {
            Terminator::For { exit_trip, .. } => *exit_trip,
            other => panic!("a `for` header ends in {other:?}"),
        }
    }

    fn exit_block(&self, loop_: &Loop) -> BlockIdx {
        match &self.cfg.blocks[loop_.natural.header.0].terminator {
            Terminator::For { exit, .. } => self.cfg.label_to_block[exit],
            other => panic!("a `for` header ends in {other:?}"),
        }
    }

    fn insts(&self, block: BlockIdx) -> &[Inst] {
        &self.cfg.blocks[block.0].insts
    }

    fn word(&self, value: ValueId) -> Option<i128> {
        self.cfg
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find_map(|inst| match &inst.kind {
                InstKind::Const {
                    dst,
                    value: Literal::Int(word),
                } if *dst == value => Some(*word),
                _ => None,
            })
    }

    fn fetched(&self, name: &str) -> ValueId {
        let wanted = QualifiedRef::root(self.interner.intern(name));
        self.cfg
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find_map(|inst| match &inst.kind {
                InstKind::Fetch { dst, context } if *context == wanted => Some(*dst),
                _ => None,
            })
            .unwrap_or_else(|| panic!("the body fetches @{name}:\n{}", self.listing))
    }
}

fn exact_add() -> Carried {
    Carried::Merge {
        op: MergeOp::Add,
        exact: true,
    }
}

fn binop(insts: &[Inst], op: BinOp, left: ValueId, right: ValueId) -> Option<ValueId> {
    insts.iter().find_map(|inst| match &inst.kind {
        InstKind::BinOp {
            dst,
            op: found,
            left: l,
            right: r,
        } if *found == op && *l == left && *r == right => Some(*dst),
        _ => None,
    })
}

fn cast(insts: &[Inst], src: ValueId, to: IntTy) -> Option<ValueId> {
    insts.iter().find_map(|inst| match &inst.kind {
        InstKind::Cast {
            dst,
            src: from,
            to: CastTy::Int(width),
        } if *from == src && *width == to => Some(*dst),
        _ => None,
    })
}

struct Scaled {
    product: ValueId,
    factor: i128,
}

fn scaled(c: &Compiled, insts: &[Inst], count: ValueId) -> Option<Scaled> {
    insts.iter().find_map(|inst| match &inst.kind {
        InstKind::BinOp {
            dst,
            op: BinOp::Mul,
            left,
            right,
        } if *left == count => Some(Scaled {
            product: *dst,
            factor: c.word(*right)?,
        }),
        _ => None,
    })
}

// -- A second counter ------------------------------------------------

const SECOND_COUNTER: &str =
    "let j = @base; let s = 0; for i in 0..@n { s = s + j; j = j + 2; } s + j";

#[test]
fn a_second_counter_is_computed_from_the_first() {
    let none = Compiled::of(SECOND_COUNTER, Opt::None);
    let [before] = none.loops_by_header()[..] else {
        panic!("one loop:\n{}", none.listing);
    };
    assert_eq!(
        none.ivs(before),
        1,
        "`j` is carried at none:\n{}",
        none.listing
    );
    assert_eq!(none.exit_trip(before), ExitTrip::Absent);

    let full = Compiled::of(SECOND_COUNTER, Opt::Full);
    let [loop_] = full.loops_by_header()[..] else {
        panic!("one loop:\n{}", full.listing);
    };
    assert_eq!(
        full.carried(loop_),
        [exact_add()],
        "the header carries the sum and not `j`:\n{}",
        full.listing
    );
    assert_eq!(full.state(loop_).strength(), Strength::Weak);

    let LoopKind::For {
        source: ForSource::Range { at, .. },
    } = loop_.kind
    else {
        panic!("a range `for`:\n{}", full.listing);
    };
    let base = full.fetched("base");
    let body = for_body(&full.cfg, loop_.natural.header);
    let counter = full.cfg.blocks[body.0].params[0];
    let insts = full.insts(body);
    assert_eq!(full.word(at), Some(0), "the range starts at 0:\n{}", full.listing);
    let subtracts = insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::BinOp { op: BinOp::Sub, .. }));
    assert!(
        !subtracts,
        "`k = counter − 0` is the counter (RFC-0083):\n{}",
        full.listing
    );
    let advanced =
        scaled(&full, insts, counter).unwrap_or_else(|| panic!("`k · 2`:\n{}", full.listing));
    assert_eq!(advanced.factor, 2);
    assert!(
        binop(insts, BinOp::Add, base, advanced.product).is_some(),
        "the body computes `base + k·2`:\n{}",
        full.listing
    );

    assert_eq!(full.exit_trip(loop_), ExitTrip::Defined);
    let exit = full.exit_block(loop_);
    let trip = full.cfg.blocks[exit.0].params[0];
    assert_eq!(full.cfg.val_types[&trip], Ty::U64);
    let insts = full.insts(exit);
    let trip_i64 = cast(insts, trip, IntTy::I64)
        .unwrap_or_else(|| panic!("the exit converts the trip count:\n{}", full.listing));
    let advanced =
        scaled(&full, insts, trip_i64).unwrap_or_else(|| panic!("`trip · 2`:\n{}", full.listing));
    assert_eq!(advanced.factor, 2);
    let exit_j = binop(insts, BinOp::Add, base, advanced.product)
        .unwrap_or_else(|| panic!("the exit computes `base + trip·2`:\n{}", full.listing));
    let reads_exit_j = insts.iter().any(|inst| {
        matches!(&inst.kind, InstKind::BinOp { op: BinOp::Add, right, .. } if *right == exit_j)
    });
    assert!(
        reads_exit_j,
        "`s + j` after the loop reads the exit value:\n{}",
        full.listing
    );
    assert!(
        full.listing.contains("else L2(trip)"),
        "the printout names the count the exit edge defines:\n{}",
        full.listing
    );
}

// -- What stays carried ----------------------------------------------

const MERGE: &str = "let s = 0; let j = 1; for i in 0..@n { s = s + j * 3; j = j + 1; } s";

#[test]
fn a_merge_is_still_carried_and_an_iv_read_only_inside_asks_for_no_count() {
    let full = Compiled::of(MERGE, Opt::Full);
    let [loop_] = full.loops_by_header()[..] else {
        panic!("one loop:\n{}", full.listing);
    };
    assert_eq!(full.carried(loop_), [exact_add()], "{}", full.listing);
    assert_eq!(
        full.exit_trip(loop_),
        ExitTrip::Absent,
        "no read of `j` follows the loop, so nothing asked for the count:\n{}",
        full.listing
    );
}

const RECURRENCE: &str = "let acc = 0; let j = 0; \
     for i in 0..6 { acc = acc * 2 + (j * @base + 1); j = j + 1; } acc";

const BREAK: &str = "let j = 0; let s = 0; \
     for i in 0..@n { if i == 4 { break; }; s = s + j; j = j + 5; } s * 1000 + j";

#[test]
fn a_strong_loop_is_left_to_lsr() {
    let none = Compiled::of(RECURRENCE, Opt::None);
    let full = Compiled::of(RECURRENCE, Opt::Full);
    let [before] = none.loops_by_header()[..] else {
        panic!("one loop:\n{}", none.listing);
    };
    let [loop_] = full.loops_by_header()[..] else {
        panic!("one loop:\n{}", full.listing);
    };
    assert_eq!(full.state(loop_).strength(), Strength::Strong);
    assert_eq!(none.ivs(before), 1, "`j`:\n{}", none.listing);
    assert_eq!(
        full.ivs(loop_),
        1,
        "`lsr` carries `j·base + 1` in `j`'s place: once the product is \
         reduced nothing reads `j` but its own `j + 1`, and the `dce` after \
         GVN sweeps both (RFC-0083):\n{}",
        full.listing
    );
    let body = for_body(&full.cfg, loop_.natural.header);
    let products = full
        .insts(body)
        .iter()
        .filter(|inst| matches!(inst.kind, InstKind::BinOp { op: BinOp::Mul, .. }))
        .count();
    assert_eq!(products, 1, "only `acc * 2` multiplies:\n{}", full.listing);
    assert_eq!(full.exit_trip(loop_), ExitTrip::Absent);

    let full = Compiled::of(BREAK, Opt::Full);
    let [loop_] = full.loops_by_header()[..] else {
        panic!("one loop:\n{}", full.listing);
    };
    assert_eq!(full.state(loop_).strength(), Strength::Strong);
    assert_eq!(
        full.ivs(loop_),
        1,
        "a loop a `break` leaves keeps `j`:\n{}",
        full.listing
    );
    assert_eq!(full.exit_trip(loop_), ExitTrip::Absent);
}

const BOTH: &str = "let j = 0; let s = 0; for i in 0..@n { s = s + j; j = j + 2; } \
     let acc = 0; let q = 0; \
     for i in 0..@n { acc = acc * 2 + (q * @base + 1); q = q + 1; } \
     s + j + acc";

#[test]
fn no_loop_is_rewritten_by_both_passes() {
    let none = Compiled::of(BOTH, Opt::None);
    let full = Compiled::of(BOTH, Opt::Full);
    let before = none.loops_by_header();
    let after = full.loops_by_header();
    assert_eq!(before.len(), 2, "{}", none.listing);
    assert_eq!(after.len(), 2, "{}", full.listing);
    for (before, after) in before.into_iter().zip(after) {
        match full.state(after).strength() {
            Strength::Weak => {
                assert_eq!(
                    full.ivs(after),
                    0,
                    "a weak loop carries only merges:\n{}",
                    full.listing
                );
                assert_eq!(full.exit_trip(after), ExitTrip::Defined);
            }
            Strength::Strong => {
                assert_eq!(
                    full.ivs(after),
                    none.ivs(before),
                    "`lsr` adds one `Iv` to a strong loop, and the `dce` after \
                     GVN sweeps the one it replaced, which nothing reads but \
                     its own advance (RFC-0083):\n{}",
                    full.listing
                );
                assert_eq!(full.exit_trip(after), ExitTrip::Absent);
            }
        }
    }
}

// -- Nests -----------------------------------------------------------

/// Each inner loop also sums `r`, a merge it keeps carrying, so it stays a
/// loop once `q` is canonicalized. A body that does nothing is removed
/// (RFC-0087), which would leave one loop to judge.
const WEAK_IN_WEAK: &str = "let v = vec([0, 0, 0]); let j = 1; \
     for x in &mut v { let q = j; let r = 0; for k in 0..4 { q = q + 3; r = r + k; } \
     *x = q + r; j = j + 2; } \
     j * 1000000 + v[0u64] * 10000 + v[1u64] * 100 + v[2u64]";

const WEAK_IN_STRONG: &str = "let t = 0; \
     for i in 0..@n { let q = 0; let r = 0; for k in 0..@m { q = q + 3; r = r + k; } \
     t = t * 2 + q + r; } t";

#[test]
fn nested_loops_are_each_judged_by_their_own_state() {
    let full = Compiled::of(WEAK_IN_WEAK, Opt::Full);
    let [outer, inner] = full.loops_by_header()[..] else {
        panic!("two loops:\n{}", full.listing);
    };
    assert!(outer.natural.contains(inner.natural.header));
    for (loop_, carried) in [(outer, vec![]), (inner, vec![exact_add()])] {
        assert_eq!(full.state(loop_).strength(), Strength::Weak);
        assert_eq!(
            full.carried(loop_),
            carried,
            "the outer loop carries nothing and the inner one its sum:\n{}",
            full.listing
        );
        assert_eq!(
            full.exit_trip(loop_),
            ExitTrip::Defined,
            "`q` is read after the inner loop and `j` after the outer:\n{}",
            full.listing
        );
    }

    let full = Compiled::of(WEAK_IN_STRONG, Opt::Full);
    let [outer, inner] = full.loops_by_header()[..] else {
        panic!("two loops:\n{}", full.listing);
    };
    assert_eq!(full.state(outer).strength(), Strength::Strong);
    assert_eq!(full.exit_trip(outer), ExitTrip::Absent);
    assert_eq!(full.state(inner).strength(), Strength::Weak);
    assert_eq!(full.carried(inner), vec![exact_add()], "{}", full.listing);
    assert_eq!(full.exit_trip(inner), ExitTrip::Defined);
}

// -- Declined --------------------------------------------------------

const WHILE: &str = "let j = 0; let i = 0; while i <= 4 { j = j + 2; i = i + 1; } j";

#[test]
fn a_weak_while_is_untouched() {
    let none = Compiled::of(WHILE, Opt::None);
    let full = Compiled::of(WHILE, Opt::Full);
    let [before] = none.loops_by_header()[..] else {
        panic!("one loop:\n{}", none.listing);
    };
    let [loop_] = full.loops_by_header()[..] else {
        panic!("one loop:\n{}", full.listing);
    };
    assert!(matches!(loop_.kind, LoopKind::While));
    assert_eq!(full.state(loop_).strength(), Strength::Weak);
    assert_eq!(full.carried(loop_), none.carried(before));
    assert_eq!(full.ivs(loop_), 2, "`i` and `j`:\n{}", full.listing);
}

#[test]
fn no_level_below_full_asks_for_a_count() {
    for source in [SECOND_COUNTER, MERGE, BOTH, WEAK_IN_WEAK, WEAK_IN_STRONG] {
        let none = Compiled::of(source, Opt::None);
        for loop_ in none.loops_by_header() {
            if matches!(loop_.kind, LoopKind::For { .. }) {
                assert_eq!(none.exit_trip(loop_), ExitTrip::Absent, "{}", none.listing);
            }
        }
        assert!(!none.listing.contains("trip"), "{}", none.listing);
    }
}

// -- The validator ---------------------------------------------------

enum ExitEntries {
    ExitEdgeAlone,
    AlsoABranch,
}

fn hand_built(trip_ty: Ty, entries: ExitEntries) -> MirModule {
    let mut factory = LocalFactory::<ValueId>::new();
    let mut val_types = FxHashMap::default();
    let mut value = |ty: Ty| {
        let v = factory.next();
        val_types.insert(v, ty);
        v
    };
    let at = value(Ty::I64);
    let hi = value(Ty::I64);
    let counter = value(Ty::I64);
    let trip = value(trip_ty);
    let cond = value(Ty::Bool);
    let other = value(Ty::U64);
    let inst = |kind| Inst {
        span: Span::ZERO,
        kind,
    };
    let header = Label(0);
    let body = Label(1);
    let exit = Label(2);
    let mut insts = vec![
        inst(InstKind::Const {
            dst: at,
            value: Literal::Int(0),
        }),
        inst(InstKind::Const {
            dst: hi,
            value: Literal::Int(3),
        }),
    ];
    match entries {
        ExitEntries::AlsoABranch => insts.extend([
            inst(InstKind::Const {
                dst: cond,
                value: Literal::Bool(true),
            }),
            inst(InstKind::Const {
                dst: other,
                value: Literal::Int(7),
            }),
            inst(InstKind::JumpIf {
                cond,
                then_label: header,
                then_args: vec![],
                else_label: exit,
                else_args: vec![other],
            }),
        ]),
        ExitEntries::ExitEdgeAlone => insts.push(inst(InstKind::Jump {
            label: header,
            args: vec![],
        })),
    }
    insts.extend([
        inst(InstKind::BlockLabel {
            label: header,
            params: vec![],
        }),
        inst(InstKind::For {
            source: ForSource::Range { at, hi },
            body,
            body_args: vec![],
            exit,
            exit_trip: ExitTrip::Defined,
            exit_args: vec![],
        }),
        inst(InstKind::BlockLabel {
            label: body,
            params: vec![counter],
        }),
        inst(InstKind::Jump {
            label: header,
            args: vec![],
        }),
        inst(InstKind::BlockLabel {
            label: exit,
            params: vec![trip],
        }),
        inst(InstKind::Return {
            value: trip,
            order: None,
        }),
    ]);
    MirModule {
        main: MirBody {
            insts,
            val_types,
            val_factory: factory,
            label_count: 3,
            ..MirBody::new()
        },
        closures: FxHashMap::default(),
        ret: Ty::U64,
    }
}

#[test]
fn the_validator_admits_a_count_the_exit_edge_alone_defines() {
    let errors = check_types(&hand_built(Ty::U64, ExitEntries::ExitEdgeAlone));
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn the_validator_refuses_a_count_another_edge_also_defines() {
    let errors = check_types(&hand_built(Ty::U64, ExitEntries::AlsoABranch));
    assert!(
        errors.iter().any(|e| matches!(
            e.kind,
            ValidationErrorKind::TripBesideExitEdge {
                exit: Label(2),
                entries: 2
            }
        )),
        "{errors:?}"
    );
}

#[test]
fn the_validator_refuses_a_count_that_is_not_a_u64() {
    let errors = check_types(&hand_built(Ty::I64, ExitEntries::ExitEdgeAlone));
    assert!(
        errors.iter().any(|e| matches!(
            &e.kind,
            ValidationErrorKind::TypeMismatch { inst_name, desc, .. }
                if inst_name == "For(exit)" && desc == "trip"
        )),
        "{errors:?}"
    );
}
