//! A constant expression folds (RFC-0055).
//!
//! The values this pass computes are the values `acvus-interpreter`'s
//! `ops::arith` computes at run time, at the operand's width: `+`, `-` and
//! `*` wrap, a shift takes its amount modulo the width, a comparison reads
//! both operands at the width's own signedness, and `/` and `%` panic on a
//! zero divisor and on a quotient that leaves the width. Change an
//! operation there and the matching function here moves with it;
//! `acvus-interpreter-test/tests/fold_agreement.rs` runs both.
//!
//! The pass introduces no operator the body did not already hold. It
//! replaces a binary operation with its constant, or moves a constant
//! from one operand to another; `prepare::arith_of` claims the five
//! arithmetic operators, and an operator outside them breaks the chain
//! that reads the site.
//!
//! A `f64` operation whose result is NaN is not folded. `==` on floats
//! compares bit patterns here, so a folded NaN would carry the compiler's
//! pattern where the machine's belongs.

use acvus_ast::{Literal, UnaryOp};
use rustc_hash::FxHashMap;

use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{BinOp, InstKind, ValueId};
use crate::ty::{CastTy, IntTy, Ty, WordTy};

pub fn run(cfg: &mut CfgBody) {
    while rewrite_once(cfg) {}
}

const ASSOCIATIVE: [BinOp; 5] = [
    BinOp::Add,
    BinOp::Mul,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::Xor,
];

/// Where an instruction sits in the body.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Site {
    block: BlockIdx,
    at: usize,
}

/// A binary operation's operands split by constness, where exactly one of
/// the two is constant.
#[derive(Clone, Copy)]
struct Split {
    constant: ValueId,
    variable: ValueId,
}

/// The constants, the use counts and the definitions of the body as it
/// stands now.
struct Facts {
    consts: FxHashMap<ValueId, Literal>,
    texts: FxHashMap<ValueId, String>,
    uses: FxHashMap<ValueId, usize>,
    defs: FxHashMap<ValueId, Site>,
}

/// Apply the first rewrite any rule finds; `true` when one was applied.
fn rewrite_once(cfg: &mut CfgBody) -> bool {
    let facts = Facts::of(cfg);
    let sites = (0..cfg.blocks.len()).flat_map(|block| {
        (0..cfg.blocks[block].insts.len()).map(move |at| Site {
            block: BlockIdx(block),
            at,
        })
    });
    for site in sites.collect::<Vec<Site>>() {
        if fold_constant(cfg, &facts, site)
            || fold_cast(cfg, &facts, site)
            || fold_not(cfg, &facts, site)
            || fold_string_eq(cfg, &facts, site)
            || join_constants(cfg, &facts, site)
        {
            return true;
        }
    }
    false
}

impl Facts {
    fn of(cfg: &CfgBody) -> Facts {
        let mut consts = FxHashMap::default();
        let mut texts = FxHashMap::default();
        let mut uses: FxHashMap<ValueId, usize> = FxHashMap::default();
        let mut defs = FxHashMap::default();
        for (block, held) in cfg.blocks.iter().enumerate() {
            for (at, inst) in held.insts.iter().enumerate() {
                let site = Site {
                    block: BlockIdx(block),
                    at,
                };
                match &inst.kind {
                    InstKind::Const { dst, value } => {
                        consts.insert(*dst, value.clone());
                    }
                    InstKind::ConstStr { dst, text } => {
                        texts.insert(*dst, text.clone());
                    }
                    _ => {}
                }
                for def in inst_info::defs(&inst.kind) {
                    defs.insert(def, site);
                }
                for read in inst_info::uses(&inst.kind) {
                    *uses.entry(read).or_default() += 1;
                }
            }
            for read in terminator_uses(&held.terminator) {
                *uses.entry(read).or_default() += 1;
            }
        }
        Facts {
            consts,
            texts,
            uses,
            defs,
        }
    }

    fn literal(&self, v: ValueId) -> Option<&Literal> {
        self.consts.get(&v)
    }

    fn integer(&self, v: ValueId) -> Option<i128> {
        match self.consts.get(&v) {
            Some(Literal::Int(n)) => Some(*n),
            _ => None,
        }
    }

    /// How many times the body reads `v`. A value no instruction and no
    /// terminator names is read zero times.
    fn reads(&self, v: ValueId) -> usize {
        self.uses.get(&v).copied().unwrap_or_default()
    }

    /// The operands split by constness, where exactly one is constant.
    fn split(&self, left: ValueId, right: ValueId) -> Option<Split> {
        match (self.literal(left), self.literal(right)) {
            (Some(_), None) => Some(Split {
                constant: left,
                variable: right,
            }),
            (None, Some(_)) => Some(Split {
                constant: right,
                variable: left,
            }),
            (Some(_), Some(_)) | (None, None) => None,
        }
    }
}

/// Every value a terminator reads, block arguments included.
fn terminator_uses(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => std::iter::once(*cond)
            .chain(then_args.iter().copied())
            .chain(else_args.iter().copied())
            .collect(),
        Terminator::Switch { tag, arms, default } => std::iter::once(*tag)
            .chain(arms.iter().flat_map(|(_, _, args)| args.iter().copied()))
            .chain(default.iter().flat_map(|(_, args)| args.iter().copied()))
            .collect(),
        Terminator::For {
            source, exit_args, ..
        } => source
            .uses()
            .into_iter()
            .chain(exit_args.iter().copied())
            .collect(),
        Terminator::Return { value, order, .. } => std::iter::once(*value).chain(*order).collect(),
        Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

// -- A binary operation on two constants is the constant --------------

fn fold_constant(cfg: &mut CfgBody, facts: &Facts, site: Site) -> bool {
    let InstKind::BinOp {
        dst,
        op,
        left,
        right,
    } = kind_at(cfg, site)
    else {
        return false;
    };
    let (Some(a), Some(b)) = (facts.literal(left), facts.literal(right)) else {
        return false;
    };
    let Some(ty) = cfg.val_types.get(&left) else {
        return false;
    };
    let Some(value) = evaluate(op, ty, a, b) else {
        return false;
    };
    *kind_mut(cfg, site) = InstKind::Const { dst, value };
    true
}

// -- A negation of a constant is the constant -------------------------

fn fold_not(cfg: &mut CfgBody, facts: &Facts, site: Site) -> bool {
    let InstKind::UnaryOp {
        dst,
        op: UnaryOp::Not,
        operand,
    } = kind_at(cfg, site)
    else {
        return false;
    };
    let Some(Literal::Bool(held)) = facts.literal(operand) else {
        return false;
    };
    *kind_mut(cfg, site) = InstKind::Const {
        dst,
        value: Literal::Bool(!held),
    };
    true
}

// -- A comparison of two constant strings is the constant --------------

fn fold_string_eq(cfg: &mut CfgBody, facts: &Facts, site: Site) -> bool {
    let InstKind::StringEq { dst, a, b } = kind_at(cfg, site) else {
        return false;
    };
    let (Some(left), Some(right)) = (facts.texts.get(&a), facts.texts.get(&b)) else {
        return false;
    };
    *kind_mut(cfg, site) = InstKind::Const {
        dst,
        value: Literal::Bool(left == right),
    };
    true
}

// -- A cast of a constant is the constant (RFC-0049) -------------------

fn fold_cast(cfg: &mut CfgBody, facts: &Facts, site: Site) -> bool {
    let InstKind::Cast { dst, src, to } = kind_at(cfg, site) else {
        return false;
    };
    let Some(from) = cfg.val_types.get(&src).and_then(CastTy::of_ty) else {
        return false;
    };
    let Some(value) = facts
        .literal(src)
        .and_then(|held| cast_result(from, to, held))
    else {
        return false;
    };
    *kind_mut(cfg, site) = InstKind::Const { dst, value };
    true
}

/// The value `src as to` has, where `src` is a constant of type `from`.
///
/// Every arm is the Rust `as` expression `ops::cast`'s instance for the
/// same pair runs, so the two agree by construction;
/// `acvus-interpreter-test/tests/fold_agreement.rs` runs both.
fn cast_result(from: CastTy, to: CastTy, held: &Literal) -> Option<Literal> {
    let source = match (from, held) {
        (CastTy::Int(k), Literal::Int(a)) => Source::Int(k.read(register_word(*a))),
        (CastTy::Char, Literal::Char(c)) => Source::Int(i128::from(u32::from(*c))),
        (CastTy::F64, Literal::Float(x)) => Source::Float(*x),
        _ => return None,
    };
    match (to, source) {
        (CastTy::Char, Source::Int(a)) => u8::try_from(a)
            .ok()
            .map(|byte| Literal::Char(char::from(byte))),
        (CastTy::Char, Source::Float(_)) => None,
        (num, Source::Int(a)) => Some(int_cast(a, num.word())),
        (num, Source::Float(x)) => Some(float_cast(x, num.word())),
    }
}

/// The word a cast reads, at the width its source type named.
#[derive(Clone, Copy)]
enum Source {
    Int(i128),
    Float(f64),
}

fn int_cast(a: i128, to: WordTy) -> Literal {
    match to {
        WordTy::Int(j) => Literal::Int(j.read(register_word(a))),
        WordTy::F64 => Literal::Float(a as f64),
    }
}

fn float_cast(x: f64, to: WordTy) -> Literal {
    let WordTy::Int(j) = to else {
        return Literal::Float(x);
    };
    Literal::Int(match j {
        IntTy::I8 => i128::from(x as i8),
        IntTy::I16 => i128::from(x as i16),
        IntTy::I32 => i128::from(x as i32),
        IntTy::I64 => i128::from(x as i64),
        IntTy::U8 => i128::from(x as u8),
        IntTy::U16 => i128::from(x as u16),
        IntTy::U32 => i128::from(x as u32),
        IntTy::U64 => i128::from(x as u64),
    })
}

// -- Two constants under one associative operator join ----------------

fn join_constants(cfg: &mut CfgBody, facts: &Facts, site: Site) -> bool {
    let InstKind::BinOp {
        dst,
        op,
        left,
        right,
    } = kind_at(cfg, site)
    else {
        return false;
    };
    let Some(k) = int_width(cfg, left) else {
        return false;
    };
    if !ASSOCIATIVE.contains(&op) {
        return false;
    }
    let Some(outer) = facts.split(left, right) else {
        return false;
    };
    if facts.reads(outer.variable) != 1 {
        return false;
    }
    let Some(&inner_site) = facts.defs.get(&outer.variable) else {
        return false;
    };
    let InstKind::BinOp {
        op: inner_op,
        left: inner_left,
        right: inner_right,
        ..
    } = kind_at(cfg, inner_site)
    else {
        return false;
    };
    if inner_op != op || int_width(cfg, inner_left) != Some(k) {
        return false;
    }
    let Some(inner) = facts.split(inner_left, inner_right) else {
        return false;
    };
    let (Some(a), Some(b)) = (facts.integer(outer.constant), facts.integer(inner.constant)) else {
        return false;
    };
    let Some(value) = int_result(op, k, a, b) else {
        return false;
    };
    *kind_mut(cfg, inner_site) = InstKind::Const {
        dst: outer.variable,
        value,
    };
    *kind_mut(cfg, site) = InstKind::BinOp {
        dst,
        op,
        left: inner.variable,
        right: outer.variable,
    };
    true
}

fn kind_at(cfg: &CfgBody, site: Site) -> InstKind {
    cfg.blocks[site.block.0].insts[site.at].kind.clone()
}

fn kind_mut(cfg: &mut CfgBody, site: Site) -> &mut InstKind {
    &mut cfg.blocks[site.block.0].insts[site.at].kind
}

/// The integer width an operand is read at, where it is an integer.
fn int_width(cfg: &CfgBody, v: ValueId) -> Option<IntTy> {
    match cfg.val_types.get(&v) {
        Some(Ty::Int(k)) => Some(*k),
        _ => None,
    }
}

// -- The value a binary operation on two constants has ----------------

/// The constant `op` gives on `left` and `right` at the operand type `ty`,
/// or nothing where the run-time operation panics instead of giving one.
fn evaluate(op: BinOp, ty: &Ty, left: &Literal, right: &Literal) -> Option<Literal> {
    match (ty, left, right) {
        (Ty::Int(k), Literal::Int(a), Literal::Int(b)) => int_result(op, *k, *a, *b),
        (Ty::Float, Literal::Float(a), Literal::Float(b)) => float_result(op, *a, *b),
        (Ty::Bool, Literal::Bool(a), Literal::Bool(b)) => bool_result(op, *a, *b),
        _ => None,
    }
}

fn int_result(op: BinOp, k: IntTy, a: i128, b: i128) -> Option<Literal> {
    let word = |bits: u64| Some(Literal::Int(k.read(bits)));
    let wrap = |v: i128| word(register_word(v));
    let held = |v: bool| Some(Literal::Bool(v));
    match op {
        BinOp::Add => wrap(a.wrapping_add(b)),
        BinOp::Sub => wrap(a.wrapping_sub(b)),
        BinOp::Mul => wrap(a.wrapping_mul(b)),
        BinOp::Div => quotient(k, a, b).map(Literal::Int),
        BinOp::Mod => quotient(k, a, b).map(|_| Literal::Int(a % b)),
        BinOp::BitAnd => wrap(a & b),
        BinOp::BitOr => wrap(a | b),
        BinOp::Xor => wrap(a ^ b),
        BinOp::Shl => word(register_word(a).wrapping_shl(shift_amount(k, b))),
        BinOp::Shr if k.signed() => wrap(a >> shift_amount(k, b)),
        BinOp::Shr => word(register_word(a).wrapping_shr(shift_amount(k, b))),
        BinOp::Eq => held(a == b),
        BinOp::Neq => held(a != b),
        BinOp::Lt => held(a < b),
        BinOp::Gt => held(a > b),
        BinOp::Lte => held(a <= b),
        BinOp::Gte => held(a >= b),
        BinOp::Min => wrap(a.min(b)),
        BinOp::Max => wrap(a.max(b)),
        BinOp::And | BinOp::Or => None,
    }
}

/// The quotient, where `/` and `%` have one. Both panic on a zero divisor
/// and both panic where the quotient leaves the width, so `%` asks for the
/// quotient it does not keep.
fn quotient(k: IntTy, a: i128, b: i128) -> Option<i128> {
    let q = a.checked_div(b)?;
    k.holds(q).then_some(q)
}

/// The register word a literal occupies, as `prepare::constant` writes it:
/// the low 64 bits, which `IntTy::read` reads back at the width.
fn register_word(v: i128) -> u64 {
    v as u64
}

/// The shift amount `ops::arith::shift_amount` takes: the right operand's
/// register word modulo the width. A width is at most 64, so the mask
/// keeps only the low six bits of that word.
fn shift_amount(k: IntTy, b: i128) -> u32 {
    (register_word(b) as u32) & (k.bits() - 1)
}

fn float_result(op: BinOp, a: f64, b: f64) -> Option<Literal> {
    let num = |v: f64| (!v.is_nan()).then_some(Literal::Float(v));
    let held = |v: bool| Some(Literal::Bool(v));
    match op {
        BinOp::Add => num(a + b),
        BinOp::Sub => num(a - b),
        BinOp::Mul => num(a * b),
        BinOp::Div => num(a / b),
        BinOp::Mod => num(a % b),
        BinOp::Eq => held(a.to_bits() == b.to_bits()),
        BinOp::Neq => held(a.to_bits() != b.to_bits()),
        BinOp::Lt => held(a.total_cmp(&b).is_lt()),
        BinOp::Gt => held(a.total_cmp(&b).is_gt()),
        BinOp::Lte => held(a.total_cmp(&b).is_le()),
        BinOp::Gte => held(a.total_cmp(&b).is_ge()),
        BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::And
        | BinOp::Or
        | BinOp::Min
        | BinOp::Max => None,
    }
}

/// The operators and the operands `acvus-interpreter-test`'s agreement
/// test cannot reach: the surface grammar writes no shift and no bitwise
/// operator, and the checker rejects a signed `MIN` literal. These pin
/// them against the Rust operators `ops::arith`'s `Int` delegates to.
#[cfg(test)]
mod unreachable_from_source {
    use super::*;

    fn int(v: i128) -> Option<Literal> {
        Some(Literal::Int(v))
    }

    #[test]
    fn a_shift_takes_its_amount_modulo_the_width() {
        assert_eq!(
            int_result(BinOp::Shl, IntTy::U8, 200, 9),
            int(i128::from(200u8.wrapping_shl(9)))
        );
        assert_eq!(
            int_result(BinOp::Shr, IntTy::I8, -8, 1),
            int(i128::from((-8i8).wrapping_shr(1)))
        );
        assert_eq!(
            int_result(BinOp::Shr, IntTy::I8, -8, -1),
            int(i128::from((-8i8).wrapping_shr(7)))
        );
        assert_eq!(
            int_result(BinOp::Shr, IntTy::U8, 200, 1),
            int(i128::from(200u8.wrapping_shr(1)))
        );
    }

    #[test]
    fn a_quotient_that_leaves_the_width_does_not_fold() {
        assert_eq!(int_result(BinOp::Div, IntTy::I8, -128, -1), None);
        assert_eq!(int_result(BinOp::Mod, IntTy::I8, -128, -1), None);
        assert_eq!(int_result(BinOp::Div, IntTy::I64, 1, 0), None);
        assert_eq!(int_result(BinOp::Mod, IntTy::I64, 1, 0), None);
    }

    #[test]
    fn a_bitwise_operator_folds_at_the_width() {
        assert_eq!(
            int_result(BinOp::BitAnd, IntTy::I8, -8, 6),
            int(i128::from(-8i8 & 6))
        );
        assert_eq!(
            int_result(BinOp::BitOr, IntTy::U8, 200, 6),
            int(i128::from(200u8 | 6))
        );
        assert_eq!(
            int_result(BinOp::Xor, IntTy::I8, -8, 6),
            int(i128::from(-8i8 ^ 6))
        );
    }
}

fn bool_result(op: BinOp, a: bool, b: bool) -> Option<Literal> {
    match op {
        BinOp::Eq => Some(Literal::Bool(a == b)),
        BinOp::Neq => Some(Literal::Bool(a != b)),
        BinOp::Xor => Some(Literal::Bool(a ^ b)),
        BinOp::And => Some(Literal::Bool(a && b)),
        BinOp::Or => Some(Literal::Bool(a || b)),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::Min
        | BinOp::Max => None,
    }
}
