//! Integer `min` and `max` as the passes read them, and the `for` whose body
//! does nothing, which `optimize::empty_loop` removes (RFC-0084).
//!
//! The source cannot write `min` or `max`, so the fold and value-numbering
//! tests build their bodies by hand. The programs compiled from source also
//! stand in `acvus-interpreter-test/tests/soundness/empty-loop/`, where they
//! run at both levels to their pinned values.

use acvus_ast::{Literal, Span};
use acvus_mir::analysis::inst_info;
use acvus_mir::cfg::{Block, CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{
    BinOp, ExitTrip, ForSource, Inst, InstKind, Label, MirBody, MirModule, ValueId,
};
use acvus_mir::optimize::{empty_loop, fold, gvn};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{CastTy, IntTy, LenTerm, Ty};
use acvus_mir_test::compile_script_module_at;
use acvus_utils::{Interner, LocalFactory};
use rustc_hash::FxHashMap;

// -- Bodies built by hand ----------------------------------------------

struct Hand {
    factory: LocalFactory<ValueId>,
    val_types: FxHashMap<ValueId, Ty>,
    insts: Vec<Inst>,
    params: Vec<ValueId>,
}

impl Hand {
    fn new() -> Self {
        Self {
            factory: LocalFactory::new(),
            val_types: FxHashMap::default(),
            insts: Vec::new(),
            params: Vec::new(),
        }
    }

    fn value(&mut self, ty: Ty) -> ValueId {
        let value = self.factory.next();
        self.val_types.insert(value, ty);
        value
    }

    fn param(&mut self, ty: Ty) -> ValueId {
        let value = self.value(ty);
        self.params.push(value);
        value
    }

    fn push(&mut self, kind: InstKind) {
        self.insts.push(Inst {
            span: Span::ZERO,
            kind,
        });
    }

    fn constant(&mut self, ty: Ty, value: i128) -> ValueId {
        let dst = self.value(ty);
        self.push(InstKind::Const {
            dst,
            value: Literal::Int(value),
        });
        dst
    }

    fn binop(&mut self, op: BinOp, left: ValueId, right: ValueId) -> ValueId {
        let dst = self.value(self.val_types[&left].clone());
        self.push(InstKind::BinOp {
            dst,
            op,
            left,
            right,
        });
        dst
    }

    fn returns(&mut self, value: ValueId) {
        self.push(InstKind::Return { value, order: None });
    }

    fn cfg(self, interner: &Interner, label_count: u32) -> CfgBody {
        let params = self
            .params
            .iter()
            .enumerate()
            .map(|(at, value)| (interner.intern(&format!("p{at}")), *value))
            .collect();
        promote(MirBody {
            insts: self.insts,
            val_types: self.val_types,
            params,
            val_factory: self.factory,
            label_count,
            ..MirBody::new()
        })
    }
}

fn defining(cfg: &CfgBody, value: ValueId) -> &InstKind {
    cfg.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .map(|inst| &inst.kind)
        .find(|kind| inst_info::defs(kind).contains(&value))
        .unwrap_or_else(|| panic!("{value:?} has no definition"))
}

#[derive(Debug, PartialEq)]
struct Operands {
    left: ValueId,
    right: ValueId,
}

fn operands_of(cfg: &CfgBody, value: ValueId) -> Operands {
    let InstKind::BinOp { left, right, .. } = defining(cfg, value) else {
        panic!("{value:?} is not a `BinOp`: {:?}", defining(cfg, value))
    };
    Operands {
        left: *left,
        right: *right,
    }
}

// -- fold --------------------------------------------------------------

struct Extremes {
    width: IntTy,
    left: i128,
    right: i128,
    min: i128,
    max: i128,
}

macro_rules! extremes {
    ($($t:ty => $k:ident),* $(,)?) => {{
        let mut cases = Vec::new();
        $(
            let pairs: [($t, $t); 5] = [
                (<$t>::MIN, <$t>::MAX),
                (<$t>::MAX, <$t>::MIN),
                (<$t>::MIN, <$t>::MIN),
                (<$t>::MAX, 1),
                (0, <$t>::MIN),
            ];
            for (left, right) in pairs {
                cases.push(Extremes {
                    width: IntTy::$k,
                    left: i128::from(left),
                    right: i128::from(right),
                    min: i128::from(left.min(right)),
                    max: i128::from(left.max(right)),
                });
            }
        )*
        cases
    }};
}

fn folded(width: IntTy, op: BinOp, left: i128, right: i128) -> Literal {
    let interner = Interner::new();
    let mut hand = Hand::new();
    let left = hand.constant(Ty::Int(width), left);
    let right = hand.constant(Ty::Int(width), right);
    let dst = hand.binop(op, left, right);
    hand.returns(dst);
    let mut cfg = hand.cfg(&interner, 0);
    fold::run(&mut cfg);
    match defining(&cfg, dst) {
        InstKind::Const { value, .. } => value.clone(),
        other => panic!("{op:?} at {width:?} is not folded: {other:?}"),
    }
}

#[test]
fn min_and_max_of_two_constants_fold_at_every_width() {
    let cases = extremes!(
        i8 => I8, i16 => I16, i32 => I32, i64 => I64,
        u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    );
    for case in cases {
        let Extremes {
            width,
            left,
            right,
            min,
            max,
        } = case;
        assert_eq!(
            folded(width, BinOp::Min, left, right),
            Literal::Int(min),
            "min({left}, {right}) at {width:?}"
        );
        assert_eq!(
            folded(width, BinOp::Max, left, right),
            Literal::Int(max),
            "max({left}, {right}) at {width:?}"
        );
    }
}

// -- gvn ---------------------------------------------------------------

#[derive(Clone, Copy)]
enum Order {
    AB,
    BA,
}

#[derive(Clone, Copy)]
struct Operation {
    op: BinOp,
    order: Order,
}

struct Numbered {
    first: ValueId,
    second: ValueId,
    sum_reads: Operands,
}

fn summed_after_numbering(first: Operation, second: Operation) -> Numbered {
    let interner = Interner::new();
    let mut hand = Hand::new();
    let a = hand.param(Ty::I64);
    let b = hand.param(Ty::I64);
    let mut apply = |operation: Operation| {
        let (left, right) = match operation.order {
            Order::AB => (a, b),
            Order::BA => (b, a),
        };
        hand.binop(operation.op, left, right)
    };
    let first = apply(first);
    let second = apply(second);
    let sum = hand.binop(BinOp::Add, first, second);
    hand.returns(sum);
    let mut cfg = hand.cfg(&interner, 0);
    gvn::run(&mut cfg);
    Numbered {
        first,
        second,
        sum_reads: operands_of(&cfg, sum),
    }
}

#[test]
fn min_or_max_with_its_operands_swapped_is_one_value() {
    for op in [BinOp::Max, BinOp::Min] {
        let Numbered {
            first, sum_reads, ..
        } = summed_after_numbering(
            Operation {
                op,
                order: Order::AB,
            },
            Operation {
                op,
                order: Order::BA,
            },
        );
        assert_eq!(
            sum_reads,
            Operands {
                left: first,
                right: first
            },
            "{op:?}"
        );
    }
}

#[test]
fn min_and_max_of_the_same_operands_are_two_values() {
    let Numbered {
        first,
        second,
        sum_reads,
    } = summed_after_numbering(
        Operation {
            op: BinOp::Min,
            order: Order::AB,
        },
        Operation {
            op: BinOp::Max,
            order: Order::AB,
        },
    );
    assert_eq!(
        sum_reads,
        Operands {
            left: first,
            right: second
        }
    );
}

#[test]
fn min_or_max_of_one_value_is_that_value() {
    for op in [BinOp::Min, BinOp::Max] {
        let interner = Interner::new();
        let mut hand = Hand::new();
        let a = hand.param(Ty::I64);
        let b = hand.param(Ty::I64);
        let same = hand.binop(op, a, a);
        let sum = hand.binop(BinOp::Add, same, b);
        hand.returns(sum);
        let mut cfg = hand.cfg(&interner, 0);
        gvn::run(&mut cfg);
        assert_eq!(
            operands_of(&cfg, sum),
            Operands { left: a, right: b },
            "{op:?}(a, a)"
        );
    }
}

// -- empty_loop on bodies built by hand ----------------------------------

const HEADER: Label = Label(0);
const BODY: Label = Label(1);
const EXIT: Label = Label(2);
const OUT: Label = Label(3);

enum Source {
    Range(IntTy),
    Array(usize),
}

enum BodyHolds {
    Nothing,
    AnInstruction,
    AnEdgeOut,
}

struct Shape {
    source: Source,
    exit_trip: ExitTrip,
    body: BodyHolds,
    header_carries: bool,
}

struct Bounds {
    at: ValueId,
    hi: ValueId,
}

struct Built {
    cfg: CfgBody,
    bounds: Option<Bounds>,
}

struct SourceParts {
    source: ForSource,
    bounds: Option<Bounds>,
    element: Ty,
}

fn built(shape: Shape) -> Built {
    let interner = Interner::new();
    let mut hand = Hand::new();
    let parts = match shape.source {
        Source::Range(width) => {
            let at = hand.param(Ty::Int(width));
            let hi = hand.param(Ty::Int(width));
            SourceParts {
                source: ForSource::Range { at, hi },
                bounds: Some(Bounds { at, hi }),
                element: Ty::Int(width),
            }
        }
        Source::Array(len) => {
            let array = hand.param(Ty::Array(Box::new(Ty::I64), LenTerm::Known(len)));
            SourceParts {
                source: ForSource::Array(array),
                bounds: None,
                element: Ty::I64,
            }
        }
    };
    let leaves = hand.param(Ty::Bool);
    let carried: Vec<ValueId> = match shape.header_carries {
        true => vec![hand.param(Ty::I64)],
        false => Vec::new(),
    };
    let header_params: Vec<ValueId> = carried.iter().map(|_| hand.value(Ty::I64)).collect();
    let supplied: Vec<ValueId> = match parts.source {
        ForSource::Range { .. } => vec![hand.value(parts.element.clone())],
        ForSource::Array(_) | ForSource::Slice(_) | ForSource::SliceMut(_) => {
            vec![hand.value(parts.element.clone()), hand.value(Ty::U64)]
        }
    };
    let body_carried: Vec<ValueId> = header_params.iter().map(|_| hand.value(Ty::I64)).collect();
    let trip: Vec<ValueId> = match shape.exit_trip {
        ExitTrip::Defined => vec![hand.value(Ty::U64)],
        ExitTrip::Absent => Vec::new(),
    };

    hand.push(InstKind::Jump {
        label: HEADER,
        args: carried,
    });
    hand.push(InstKind::BlockLabel {
        label: HEADER,
        params: header_params.clone(),
    });
    hand.push(InstKind::For {
        source: parts.source,
        body: BODY,
        body_args: header_params,
        exit: EXIT,
        exit_trip: shape.exit_trip,
        exit_args: vec![],
    });
    hand.push(InstKind::BlockLabel {
        label: BODY,
        params: supplied.into_iter().chain(body_carried.clone()).collect(),
    });
    match shape.body {
        BodyHolds::Nothing => hand.push(InstKind::Jump {
            label: HEADER,
            args: body_carried,
        }),
        BodyHolds::AnInstruction => {
            hand.constant(Ty::I64, 1);
            hand.push(InstKind::Jump {
                label: HEADER,
                args: body_carried,
            });
        }
        BodyHolds::AnEdgeOut => {
            hand.push(InstKind::JumpIf {
                cond: leaves,
                then_label: OUT,
                then_args: vec![],
                else_label: HEADER,
                else_args: body_carried,
            });
            hand.push(InstKind::BlockLabel {
                label: OUT,
                params: vec![],
            });
            let early = hand.constant(Ty::U64, 9);
            hand.returns(early);
        }
    }
    hand.push(InstKind::BlockLabel {
        label: EXIT,
        params: trip.clone(),
    });
    let done = match trip.as_slice() {
        [trip] => *trip,
        _ => hand.constant(Ty::U64, 0),
    };
    hand.returns(done);

    let mut cfg = hand.cfg(&interner, 4);
    empty_loop::run(&mut cfg);
    Built {
        cfg,
        bounds: parts.bounds,
    }
}

fn fors(cfg: &CfgBody) -> usize {
    cfg.blocks
        .iter()
        .filter(|block| matches!(block.terminator, Terminator::For { .. }))
        .count()
}

fn header(cfg: &CfgBody) -> &Block {
    &cfg.blocks[cfg.label_to_block[&HEADER].0]
}

struct EmptyRange {
    cfg: CfgBody,
    bounds: Bounds,
}

fn empty_range(width: IntTy, exit_trip: ExitTrip) -> EmptyRange {
    let Built { cfg, bounds } = built(Shape {
        source: Source::Range(width),
        exit_trip,
        body: BodyHolds::Nothing,
        header_carries: false,
    });
    let Some(bounds) = bounds else {
        panic!("a range has bounds")
    };
    EmptyRange { cfg, bounds }
}

#[test]
fn an_empty_range_that_defines_its_count_is_max_casts_a_subtraction_and_a_jump() {
    let EmptyRange {
        cfg,
        bounds: Bounds { at, hi },
    } = empty_range(IntTy::I8, ExitTrip::Defined);
    assert_eq!(fors(&cfg), 0);
    assert!(
        !cfg.label_to_block.contains_key(&BODY),
        "the body is pruned"
    );
    let header = header(&cfg);
    let kinds: Vec<&InstKind> = header.insts.iter().map(|inst| &inst.kind).collect();
    let [
        InstKind::BinOp {
            dst: reached,
            op: BinOp::Max,
            left: max_left,
            right: max_right,
        },
        InstKind::Cast {
            dst: reached_u64,
            src: reached_src,
            to: CastTy::Int(IntTy::U64),
        },
        InstKind::Cast {
            dst: at_u64,
            src: at_src,
            to: CastTy::Int(IntTy::U64),
        },
        InstKind::BinOp {
            dst: trip,
            op: BinOp::Sub,
            left: sub_left,
            right: sub_right,
        },
    ] = kinds.as_slice()
    else {
        panic!("{kinds:?}")
    };
    assert_eq!((*max_left, *max_right), (hi, at));
    assert_eq!(cfg.val_types[reached], Ty::I8);
    assert_eq!((*reached_src, *at_src), (*reached, at));
    assert_eq!((*sub_left, *sub_right), (*reached_u64, *at_u64));
    assert_eq!(cfg.val_types[trip], Ty::U64);
    let Terminator::Jump { label, args } = &header.terminator else {
        panic!("{:?}", header.terminator)
    };
    assert_eq!((*label, args.as_slice()), (EXIT, &[*trip][..]));
}

#[test]
fn a_u64_range_is_not_cast() {
    let EmptyRange {
        cfg,
        bounds: Bounds { at, .. },
    } = empty_range(IntTy::U64, ExitTrip::Defined);
    let kinds: Vec<&InstKind> = header(&cfg).insts.iter().map(|inst| &inst.kind).collect();
    let [
        InstKind::BinOp {
            dst: reached,
            op: BinOp::Max,
            ..
        },
        InstKind::BinOp {
            op: BinOp::Sub,
            left,
            right,
            ..
        },
    ] = kinds.as_slice()
    else {
        panic!("{kinds:?}")
    };
    assert_eq!((*left, *right), (*reached, at));
}

#[test]
fn an_empty_range_that_does_not_define_its_count_is_a_jump() {
    let EmptyRange { cfg, .. } = empty_range(IntTy::I64, ExitTrip::Absent);
    assert_eq!(fors(&cfg), 0);
    let header = header(&cfg);
    assert!(header.insts.is_empty(), "{:?}", header.insts);
    let Terminator::Jump { label, args } = &header.terminator else {
        panic!("{:?}", header.terminator)
    };
    assert_eq!((*label, args.len()), (EXIT, 0));
}

#[test]
fn an_empty_array_loop_counts_its_length() {
    let Built { cfg, .. } = built(Shape {
        source: Source::Array(4),
        exit_trip: ExitTrip::Defined,
        body: BodyHolds::Nothing,
        header_carries: false,
    });
    assert_eq!(fors(&cfg), 0);
    let header = header(&cfg);
    let kinds: Vec<&InstKind> = header.insts.iter().map(|inst| &inst.kind).collect();
    let [
        InstKind::Const {
            dst,
            value: Literal::Int(4),
        },
    ] = kinds.as_slice()
    else {
        panic!("{kinds:?}")
    };
    assert_eq!(cfg.val_types[dst], Ty::U64);
    let Terminator::Jump { args, .. } = &header.terminator else {
        panic!("{:?}", header.terminator)
    };
    assert_eq!(args.as_slice(), &[*dst][..]);
}

#[test]
fn a_body_that_holds_an_instruction_stays() {
    let Built { cfg, .. } = built(Shape {
        source: Source::Range(IntTy::I64),
        exit_trip: ExitTrip::Defined,
        body: BodyHolds::AnInstruction,
        header_carries: false,
    });
    assert_eq!(fors(&cfg), 1);
}

#[test]
fn a_body_with_an_edge_out_stays() {
    let Built { cfg, .. } = built(Shape {
        source: Source::Range(IntTy::I64),
        exit_trip: ExitTrip::Defined,
        body: BodyHolds::AnEdgeOut,
        header_carries: false,
    });
    assert_eq!(fors(&cfg), 1);
}

#[test]
fn a_header_that_carries_a_value_stays() {
    let Built { cfg, .. } = built(Shape {
        source: Source::Range(IntTy::I64),
        exit_trip: ExitTrip::Absent,
        body: BodyHolds::Nothing,
        header_carries: true,
    });
    assert_eq!(fors(&cfg), 1);
}

// -- The full pipeline ---------------------------------------------------

struct Compiled {
    listing: String,
    cfg: CfgBody,
}

impl Compiled {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let contexts = [(interner.intern("n"), Ty::I64)].into_iter().collect();
        let module: MirModule = compile_script_module_at(&interner, source, &contexts, Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let listing = dump_with(&interner, &module);
        Self {
            listing,
            cfg: promote(module.main),
        }
    }

    fn insts(&self) -> impl Iterator<Item = &InstKind> {
        self.cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .map(|inst| &inst.kind)
    }

    fn binops(&self, op: BinOp) -> usize {
        self.insts()
            .filter(|kind| matches!(kind, InstKind::BinOp { op: found, .. } if *found == op))
            .count()
    }
}

#[test]
fn a_range_whose_accumulator_is_its_count_loses_its_loop() {
    let compiled = Compiled::of("let s = 0; for i in 0..@n { s = s + 3; } s");
    let listing = &compiled.listing;
    assert_eq!(fors(&compiled.cfg), 0, "{listing}");
    assert_eq!(compiled.binops(BinOp::Max), 1, "{listing}");
    assert!(listing.contains("= max("), "{listing}");
}

#[test]
fn a_range_nothing_counts_loses_its_loop() {
    let compiled = Compiled::of("for i in 0..@n { } 7");
    let listing = &compiled.listing;
    assert_eq!(fors(&compiled.cfg), 0, "{listing}");
    assert_eq!(compiled.binops(BinOp::Max), 0, "{listing}");
}

#[test]
fn an_array_loop_is_its_length() {
    let compiled = Compiled::of("let a = [2, 3, 4, 5]; let s = 0; for x in a { s = s + 3; } s");
    let listing = &compiled.listing;
    assert_eq!(fors(&compiled.cfg), 0, "{listing}");
    assert_eq!(compiled.binops(BinOp::Max), 0, "{listing}");
    let length = compiled.insts().any(|kind| {
        matches!(kind, InstKind::Const { dst, value: Literal::Int(4) }
            if compiled.cfg.val_types[dst] == Ty::U64)
    });
    assert!(length, "{listing}");
}

#[test]
fn an_empty_loop_inside_an_empty_loop_goes_with_it() {
    let compiled = Compiled::of("let t = 0; for i in 0..@n { for k in 0..@n { } t = t + 1; } t");
    let listing = &compiled.listing;
    assert_eq!(fors(&compiled.cfg), 0, "{listing}");
}

#[test]
fn a_slice_loop_stays() {
    let compiled = Compiled::of("let v = vec([1, 2, 3]); let s = 0; for x in &v { s = s + 3; } s");
    let listing = &compiled.listing;
    assert_eq!(fors(&compiled.cfg), 1, "{listing}");
    assert_eq!(compiled.binops(BinOp::Max), 0, "{listing}");
}

#[test]
fn a_loop_that_carries_a_merge_stays() {
    let compiled = Compiled::of("let s = 0; for i in 0..@n { s = s + i; } s");
    let listing = &compiled.listing;
    let headers: Vec<&Block> = compiled
        .cfg
        .blocks
        .iter()
        .filter(|block| matches!(block.terminator, Terminator::For { .. }))
        .collect();
    let [header] = headers.as_slice() else {
        panic!("one loop stays: {listing}")
    };
    assert_eq!(header.params.len(), 1, "{listing}");
}
