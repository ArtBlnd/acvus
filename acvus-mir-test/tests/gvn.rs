//! What `optimize::gvn` merges and simplifies, and what it leaves (RFC-0083).
//!
//! Each program also stands in `acvus-interpreter-test/tests/soundness/gvn/`,
//! where it runs at both levels to its pinned value.

use acvus_ast::{BinOp, Literal};
use acvus_mir::analysis::affine::for_body;
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{Callee, InstKind};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_module_at;
use acvus_utils::Interner;

struct Compiled {
    listing: String,
    cfg: CfgBody,
}

impl Compiled {
    fn of(source: &str, contexts: &[(&str, Ty)], opt: Opt) -> Self {
        let interner = Interner::new();
        let contexts = contexts
            .iter()
            .map(|(name, ty)| (interner.intern(name), ty.clone()))
            .collect();
        let module = compile_script_module_at(&interner, source, &contexts, opt)
            .unwrap_or_else(|e| panic!("{source} at {opt:?}\n{e}"));
        let listing = dump_with(&interner, &module);
        Self {
            listing,
            cfg: promote(module.main),
        }
    }

    fn binops_in(&self, block: BlockIdx, op: BinOp) -> usize {
        self.cfg.blocks[block.0]
            .insts
            .iter()
            .filter(|inst| matches!(&inst.kind, InstKind::BinOp { op: found, .. } if *found == op))
            .count()
    }

    fn binops(&self, op: BinOp) -> usize {
        (0..self.cfg.blocks.len())
            .map(|block| self.binops_in(BlockIdx(block), op))
            .sum()
    }

    fn count(&self, wanted: fn(&InstKind) -> bool) -> usize {
        self.cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| wanted(&inst.kind))
            .count()
    }

    fn blocks_with(&self, op: BinOp) -> Vec<BlockIdx> {
        (0..self.cfg.blocks.len())
            .map(BlockIdx)
            .filter(|block| self.binops_in(*block, op) > 0)
            .collect()
    }
}

fn int(name: &str) -> (&str, Ty) {
    (name, Ty::I64)
}

/// The `accum` bench's `collatz while`, whose `for` IV canonicalization
/// rewrites (RFC-0066 rule 7) with a start of 0 and a step of 1.
const COLLATZ: &str = "let i = 0; let acc = 0; \
     while i < @n { let d = if i % 2 == 0 { i / 2 } else { i * 3 + 1 }; \
     acc = acc + d; i = i + 1; } acc";

#[test]
fn a_canonicalized_counter_is_the_counter() {
    let full = Compiled::of(COLLATZ, &[int("n")], Opt::Full);
    let [header] = (0..full.cfg.blocks.len())
        .map(BlockIdx)
        .filter(|block| matches!(full.cfg.blocks[block.0].terminator, Terminator::For { .. }))
        .collect::<Vec<_>>()[..]
    else {
        panic!("one `for`:\n{}", full.listing);
    };
    let body = for_body(&full.cfg, header);
    let counter = full.cfg.blocks[body.0].params[0];
    let operations: Vec<&InstKind> = full.cfg.blocks[body.0]
        .insts
        .iter()
        .map(|inst| &inst.kind)
        .filter(|kind| !matches!(kind, InstKind::Const { .. }))
        .collect();
    let [
        InstKind::BinOp {
            op: BinOp::Mod,
            left,
            ..
        },
        InstKind::BinOp { op: BinOp::Eq, .. },
    ] = operations[..]
    else {
        panic!(
            "the body is `i % 2 == 0` and nothing else: IV canonicalization's \
             `0 + (c − 0) * 1` is the counter `c`:\n{}",
            full.listing
        );
    };
    assert_eq!(*left, counter, "`i` is the counter:\n{}", full.listing);
}

#[test]
fn an_expression_a_dominating_block_computed_is_read_from_there() {
    let source = "let x = @a; let y = @b; let p = x * y; \
         let q = if @c { x * y + 1 } else { x * y - 1 }; p + q";
    let contexts = [int("a"), int("b"), ("c", Ty::Bool)];
    let none = Compiled::of(source, &contexts, Opt::None);
    let full = Compiled::of(source, &contexts, Opt::Full);
    assert_eq!(none.binops(BinOp::Mul), 3, "{}", none.listing);
    assert_eq!(
        full.binops(BinOp::Mul),
        1,
        "the entry's `x * y` dominates both arms:\n{}",
        full.listing
    );
}

#[test]
fn expressions_in_sibling_branches_stay_apart() {
    let source = "let x = @a; let y = @b; if @c { x * y + 1 } else { x * y - 1 }";
    let full = Compiled::of(source, &[int("a"), int("b"), ("c", Ty::Bool)], Opt::Full);
    let arms = full.blocks_with(BinOp::Mul);
    assert_eq!(
        arms.len(),
        2,
        "each arm keeps its own `x * y`, since neither arm dominates the \
         other:\n{}",
        full.listing
    );
}

#[test]
fn commuted_operands_are_one_expression() {
    let source = "let x = @a; let y = @b; (x + y) * (y + x) - x * y + y * x";
    let full = Compiled::of(source, &[int("a"), int("b")], Opt::Full);
    assert_eq!(full.binops(BinOp::Add), 2, "`x + y` once, and the last `+`:\n{}", full.listing);
    assert_eq!(
        full.binops(BinOp::Mul),
        2,
        "the square, and `x * y` once:\n{}",
        full.listing
    );
}

#[test]
fn a_float_identity_is_not_simplified() {
    let source = "let z = -@f; let s = z + 0.0; let p = z * 1.0; let q = z * 0.0; \
         1.0 / (s + p + q) > 0.0";
    let full = Compiled::of(source, &[("f", Ty::Float)], Opt::Full);
    assert_eq!(
        full.binops(BinOp::Mul),
        2,
        "`z * 1.0` and `z * 0.0` stand:\n{}",
        full.listing
    );
    assert_eq!(
        full.binops(BinOp::Add),
        3,
        "`z + 0.0` is `0.0` at `z = -0.0`, so it stands beside the two sums:\n{}",
        full.listing
    );
}

#[test]
fn an_integer_identity_is_its_operand() {
    let source = "let x = @a; (x + 0) * 1 + (0 * x) - 0";
    let full = Compiled::of(source, &[int("a")], Opt::Full);
    let arithmetic = [BinOp::Add, BinOp::Sub, BinOp::Mul]
        .into_iter()
        .map(|op| full.binops(op))
        .sum::<usize>();
    assert_eq!(
        arithmetic, 0,
        "`x + 0`, `· 1`, `0 * x`, `+ 0` and `− 0` each leave their operand:\n{}",
        full.listing
    );
}

#[test]
fn a_call_repeated_is_two_calls() {
    let source = "let x = @f; x.sqrt() + x.sqrt()";
    let full = Compiled::of(source, &[("f", Ty::Float)], Opt::Full);
    let calls = full.count(|kind| {
        matches!(
            kind,
            InstKind::FunctionCall {
                callee: Callee::Extern { .. },
                ..
            }
        )
    });
    assert_eq!(calls, 2, "{}", full.listing);
}

#[test]
fn a_read_through_a_reference_repeated_is_two_reads() {
    let source = "let v = vec([1, 2, 3]); let a = v[1]; let b = v[1]; a * 10 + b";
    let full = Compiled::of(source, &[], Opt::Full);
    let reads = full.count(|kind| matches!(kind, InstKind::Index { .. }));
    assert_eq!(reads, 2, "{}", full.listing);
}

#[test]
fn an_equal_constant_is_written_again_where_it_is_read() {
    let source = "let x = @a; let p = x + 7; let q = if @c { x + 7 } else { 7 }; p * q";
    let full = Compiled::of(source, &[int("a"), ("c", Ty::Bool)], Opt::Full);
    let sevens: Vec<BlockIdx> = (0..full.cfg.blocks.len())
        .map(BlockIdx)
        .flat_map(|block| {
            full.cfg.blocks[block.0]
                .insts
                .iter()
                .filter(|inst| {
                    matches!(&inst.kind, InstKind::Const { value, .. } if value.desugared() == Literal::Int(7))
                })
                .map(move |_| block)
        })
        .collect();
    assert_eq!(
        full.binops(BinOp::Add),
        1,
        "the arm's `x + 7` is the entry's:\n{}",
        full.listing
    );
    assert!(
        sevens.len() == 2 && sevens[0] != sevens[1],
        "the `else` arm writes its own `7` rather than reading the entry's \
         (RFC-0083 rule 1):\n{}",
        full.listing
    );
}
